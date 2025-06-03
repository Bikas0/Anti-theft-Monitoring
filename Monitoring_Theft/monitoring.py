import os
import cv2
import json
import uuid
import base64
import torch
import uvicorn
import tempfile
import numpy as np
from database import get_db
from ultralytics import YOLO
from pydantic import BaseModel
from sqlalchemy.sql import text
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta, time
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, APIRouter, UploadFile, File, Depends, HTTPException
import os
from dotenv import load_dotenv
import httpx
from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

load_dotenv()
FEATURE_EXTRACTOR_URL = os.getenv("FEATURE_EXTRACTOR_URL")

# FastAPI app instance
app = FastAPI()

# Load YOLO model for face detection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO("model/model_YOLO11.pt")#.to(device)


def features_list(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT file_name, embedding FROM anti_theft"))
    rows = result.fetchall()
    
    filenames = [row.file_name for row in rows]

    # Handles both JSON string or native Python list
    embeddings = []
    for row in rows:
        emb = row.embedding
        if isinstance(emb, str):
            emb = json.loads(emb)
        embeddings.append(np.array(emb, dtype=np.float32))

    return filenames, np.array(embeddings)

def process_image(img_data, db: Session):
    filenames, embeddings = features_list(db)
    img_array = np.array(img_data, dtype=np.float32)
    similarity = cosine_similarity(img_array.reshape(1, -1), embeddings)
    print("="*70)
    print(similarity)
    print("="*70)
    max_similarity = np.max(similarity)

    print("="*70)
    print(max_similarity)
    print("="*70)
    if max_similarity >= 0.45:
        index = np.argmax(similarity)
        return {'status': 'Success', 'Tracking Number': filenames[index].split(".")[0]}
    else:
        return {'status': 'Success', 'Tracking Number': "Unknown"}
    

def update_face_in_db(file_name, face_array, embedding, detection_time, status, db: Session):
    try:
        # Convert numpy array to list of floats
        if isinstance(embedding, np.ndarray):
            embedding = embedding.astype(float).flatten().tolist()  # Ensure flat float list

        db.execute(text("""
            INSERT INTO anti_theft (file_name, face_image, embedding, create_date, detection_time, status) 
            VALUES (:file_name, :face_image, :embedding, :create_date, :detection_time, :status)
        """), {
            "file_name": file_name,
            "face_image": face_array,
            "embedding": embedding,
            "create_date": db.execute(text("SELECT NOW() AT TIME ZONE 'Asia/Dhaka';")).fetchone()[0],
            "detection_time": detection_time,
            "status": status
        })

        db.commit()
        print("Face data updated successfully.")

    except Exception as e:
        print(f"Error updating face data: {e}")


@app.get("/api")
async def index():
    return {"message": "API health check"}


@app.post("/upload-image")
async def detect_faces(image: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = yolo_model(frame, device=device)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    async with httpx.AsyncClient() as client:
        for i, confidence in enumerate(confidences):
            if confidence >= 0.7:
                x1, y1, x2, y2 = map(int, boxes[i])
                face_crop = frame[y1:y2, x1:x2]
                face_crop_list = face_crop.tolist()
                payload = {"face_crop": face_crop_list}

                response = await client.post(FEATURE_EXTRACTOR_URL, json=payload)

                if response.status_code == 200:
                    result = response.json()
                else:
                    continue
                    # Encode the full original image as base64 once

                _, buffer = cv2.imencode('.jpg', frame)
                face_base64 = base64.b64encode(buffer).decode('utf-8')
                # face_base64 = base64.b64encode(cv2.imencode('.jpg', face_crop)[1]).decode('utf-8')
                timestamp = db.execute(text("SELECT NOW() AT TIME ZONE 'Asia/Dhaka';")).fetchone()[0]
                update_face_in_db(image.filename, face_base64, result, timestamp, status=1, db=db)

    return {"image": image.filename, "message": "Cropped face processed and saved"}


@app.get("/get-all-data", response_model=List[Dict])
def get_all_anti_theft_data(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("SELECT id, embedding FROM anti_theft"))
        rows = result.fetchall()
        # Convert rows to list of dictionaries
        return [dict(row._mapping) for row in rows]
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []


@app.post("/video-detect-faces")
async def detect_faces(video: UploadFile = File(...), db: Session = Depends(get_db)):
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        content = await video.read()  # Async read from UploadFile
        tmpfile.write(content)
        tmpfile_path = tmpfile.name

    cap = cv2.VideoCapture(tmpfile_path)
    frame_count = 0

    try:
        async with httpx.AsyncClient() as client:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % 3 == 0:
                    # Run detection (assuming yolo_model is a synchronous call)
                    results = yolo_model(frame, device=device)
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()

                    for i, confidence in enumerate(confidences):
                        if confidence >= 0.7:
                            x1, y1, x2, y2 = map(int, boxes[i])

                            # Draw bounding box and label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{confidence:.2f}"
                            cv2.putText(frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                            # Crop face and prepare payload
                            face_crop = frame[y1:y2, x1:x2]
                            face_crop_list = face_crop.tolist()
                            payload = {"face_crop": face_crop_list}

                            # Async HTTP POST to feature extractor
                            response = await client.post(FEATURE_EXTRACTOR_URL, json=payload)

                            if response.status_code == 200:
                                result = response.json()
                                embedding = np.array(result["features"]).astype(float).flatten().tolist()
                                user_identity = process_image(embedding, db)
                                print("="*70)
                                print(user_identity)
                                print("="*70)
                                print("\n")
                                tracking_number = user_identity.get("Tracking Number", "Unknown")
                                if tracking_number != "Unknown":
                                    save_dir = "SEGMENT_FACE"
                                    os.makedirs(save_dir, exist_ok=True)
                                    save_path = os.path.join(save_dir, f"{tracking_number}.jpg")
                                    cv2.imwrite(save_path, face_crop)
                            else:
                                continue

                    # # Show frame with bounding boxes and scores
                    # cv2.imshow("Face Detection", frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break

                    # Show frame with bounding boxes and scores
                    cv2.imshow("Face Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        os.remove(tmpfile_path)

    return {"video": video.filename, "message": "Faces processed, scores shown, and results saved"}

# To run the app with uvicorn when the script is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8089)
