# Use the official Python image from the Docker Hub
FROM python:3.7.9-slim

# Upgrade pip to the latest version and install dependencies in one RUN command
RUN apt-get update && \
    apt-get install -y git libgl1-mesa-glx libglib2.0-0 && \
    python -m pip install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Define environment variable (corrected the syntax for environment variable name and value)
# ENV FLASK_APP=app.py

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8081", "feature:app"]