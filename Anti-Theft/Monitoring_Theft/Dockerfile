# Use the official Python image as a base
FROM python:3.11.11-slim

RUN apt-get update && \
    apt-get install -y git libgl1-mesa-glx libglib2.0-0 && \
    python -m pip install --upgrade pip setuptools wheel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files to the container
COPY . /app

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "monitoring:app", "--host", "0.0.0.0", "--port", "8000"]