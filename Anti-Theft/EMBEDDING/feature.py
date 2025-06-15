import numpy as np
import cv2
from flask import Flask, request, jsonify
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import tensorflow as tf

# Configure GPU memory usage
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu_devices[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(0.5 * 1024))]  # 0.5 GB
    )
else:
    print("No GPU devices found.")

# Load VGGFace model (ResNet50)
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Flask app
app = Flask(__name__)


# Feature extraction function
def extract_features_from_crop(face_crop_array: np.ndarray) -> np.ndarray:
    if face_crop_array.shape[-1] != 3:
        raise ValueError("Cropped face image must have 3 channels (RGB).")

    resized_img = cv2.resize(face_crop_array, (224, 224))
    img_array = image.img_to_array(resized_img).astype('float32')
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    features = model.predict(preprocessed_img).flatten()
    return features

# Route for feature extraction
@app.route('/extract-features', methods=['POST'])
def extract_features():
    try:
        data = request.get_json()
        if 'face_crop' not in data:
            return jsonify({"error": "Missing 'face_crop' in request."}), 400

        np_face = np.array(data['face_crop'], dtype=np.float32)
        features = extract_features_from_crop(np_face)
        return jsonify({"features": features.tolist()})
    except Exception as e:
        return jsonify({"error": f"Feature extraction failed: {str(e)}"}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
