
# Object Detection Web App

Object Detection Web App is designed to help users detect objects within images using machine learning models. Users can easily upload images through the web interface and receive detection results in real-time.

## Features
- Upload images for object detection
- Real-time object detection results

## Technologies
- Frontend: HTML, CSS, JavaScript
- Backend: Python, FastAPI
- Libraries: OpenCV
- Build Tools: Node.js, npm

## Installation
To set up and run the ObjectDetectionWebApp locally, follow these steps:

### Prerequisites
- Ensure you have the following installed on your system:
  - [Python 3.7+](https://www.python.org/downloads/)
  - [Node.js and npm](https://nodejs.org/)
  - [Git](https://git-scm.com/)

### Project Setup
1. **Clone the repository**
```sh
   git clone https://github.com/AminaHamzic/object-detection-web-app.git
```
2. **Navigate to the project directory:**
```sh
    cd ObjectDetectionWebApp/backend
```
3. **Create a virtual environment and activate it:**
```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
4. **Install Python dependencies:**
```sh
    pip install -r requirements.txt
```
5. **Run the FastAPI server:**
```sh
    uvicorn main:app --reload
```
The project should now be running at http://127.0.0.1:8000.
