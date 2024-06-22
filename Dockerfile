FROM python:3.8-slim

# Set work directory
WORKDIR /app

# Install dependencies
RUN pip install numpy
RUN pip install opencv-python-headless==4.5.2.54

# Copy your application code
COPY . /app

# Run your application
CMD ["python", "main.py"]
