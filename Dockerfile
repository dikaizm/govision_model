# Use the official Python image from Docker Hub
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory (including app.py and requirements.txt) into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the .env file to the container
COPY .env /app/.env

# Expose the port to make the app accessible
EXPOSE 8020

# Set the command to run the Flask app
CMD ["python", "main.py"]
