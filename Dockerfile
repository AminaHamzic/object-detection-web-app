FROM python:3.9-slim

# Install dependencies for OpenCV
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-dev

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . /app
WORKDIR /app

# Command to run the application
CMD uvicorn main:app --host 0.0.0.0 --port $PORT

