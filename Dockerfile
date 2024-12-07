# Use an official Python runtime as the base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the app will run on
EXPOSE 8024

# Command to run the application
CMD ["python", "main.py"]
