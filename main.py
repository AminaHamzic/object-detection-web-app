import base64
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles
from pathlib import Path
import cv2
import numpy as np
import logging
import shutil
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()
'''
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)'''

# Global model variables - holding model
net = None
classes = None

@app.on_event("startup")
async def load_model():
    global net, classes
    model_path = Path("./yolo model")
    weights_path = Path(os.getenv('MODEL_WEIGHTS', './yolo model/yolov3.weights'))
    config_path = Path(os.getenv('MODEL_CONFIG', './yolo model/yolov3.cfg'))
    classes_file = Path(os.getenv('MODEL_CLASSES', './yolo model/coco.names'))

    if not weights_path.exists() or not config_path.exists() or not classes_file.exists():
        raise FileNotFoundError("Model files are missing.")

    net = cv2.dnn.readNet(str(weights_path), str(config_path))
    with open(classes_file, 'r') as f:
        classes = f.read().strip().split('\n')

frontend_dir = Path(__file__).resolve().parent / 'frontend'
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
def read_root():
    return FileResponse(frontend_dir / 'index.html')

logging.basicConfig(level=logging.DEBUG)

@app.post("/upload_image/")
async def upload_image(file_upload: UploadFile = File(...), confidence: float = Form(default=0.5)):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        with open(temp_file.name, "wb") as f:
            shutil.copyfileobj(file_upload.file, f)

        img = cv2.imread(temp_file.name)
        results, processed_img = detect_objects(img, confidence)
        _, buffer = cv2.imencode('.png', processed_img)
        base64_img = base64.b64encode(buffer).decode("utf-8")
        return {"base64_image": base64_img, "detection_results": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "An error occurred during image processing",
                                                      "error": str(e)})
    finally:
        temp_file.close()
        os.unlink(temp_file.name)


@app.post("/upload_video/")
async def upload_video(file_upload: UploadFile = File(...)):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with open(temp_file.name, "wb") as f:
            shutil.copyfileobj(file_upload.file, f)

        video_id = temp_file.name.split(os.sep)[-1]
        return {"video_id": video_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "An error occurred during video processing",
                                                      "error": str(e)})


def detect_objects(image, confidence_threshold):
    global net, classes
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                w = int(detection[2] * image.shape[1])
                h = int(detection[3] * image.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    results = []
    if len(indexes) > 0:
        indexes = indexes.flatten()
        for i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            results.append({"label": label, "confidence": confidences[i], "box": [x, y, w, h]})

    return results, image


def detect_objects_in_frame(frame, net, classes, confidence_threshold):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{classes[class_ids[i]]} {confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


def generate_video_stream(video_path, net, classes, confidence_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_objects_in_frame(frame, net, classes, confidence_threshold)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.get("/process_video/")
async def process_video(video_id: str, confidence: float = Query(0.5)):
    temp_file_path = tempfile.gettempdir() + os.sep + video_id
    if not os.path.exists(temp_file_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return StreamingResponse(generate_video_stream(temp_file_path, net, classes, confidence),
                             media_type="multipart/x-mixed-replace; boundary=frame")


