import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, StreamingResponse
from starlette.staticfiles import StaticFiles
from pathlib import Path
import cv2
import numpy as np
import base64
import logging
import shutil
import tempfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables - holding model
net = None
classes = None

@app.on_event("startup")
async def load_model():
    global net, classes
    model_path = Path("./yolo model")
    weights_path = model_path / "yolov3.weights"
    config_path = model_path / "yolov3.cfg"
    classes_file = model_path / "coco.names"

    if not weights_path.exists() or not config_path.exists() or not classes_file.exists():
        raise FileNotFoundError("Model files are missing.")

    net = cv2.dnn.readNet(str(weights_path), str(config_path))
    with open(classes_file, 'r') as f:
        classes = f.read().strip().split('\n')

frontend_dir = Path(__file__).resolve().parent.parent / 'frontend'
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def read_root():
    return FileResponse(frontend_dir / 'index.html')

logging.basicConfig(level=logging.DEBUG)

@app.post("/stream_video/")
async def stream_video(file: UploadFile = File(...), confidence: float = Query(0.5)):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        with open(temp_file.name, "wb") as f:
            shutil.copyfileobj(file.file, f)
        video_id = temp_file.name.split(os.sep)[-1]
        return {"video_id": video_id}
    finally:
        temp_file.close()

@app.post("/uploadfile/")
async def create_upload_file(file_upload: UploadFile = File(...), confidence: float = Form(default=0.5)):
    file_type = file_upload.content_type
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        with open(temp_file.name, "wb") as f:
            shutil.copyfileobj(file_upload.file, f)

        if 'image' in file_type:
            img = cv2.imread(temp_file.name)
            results, processed_img = detect_objects(img, confidence)
            _, buffer = cv2.imencode('.png', processed_img)
            base64_img = base64.b64encode(buffer).decode("utf-8")
            return {"base64_image": base64_img, "detection_results": results}
        elif 'video' in file_type:
            video_id = temp_file.name.split(os.sep)[-1]
            return {"video_id": video_id}
    finally:
        temp_file.close()

@app.get("/process_video/")
async def process_video(video_id: str, confidence: float = Query(0.5)):
    temp_file_path = tempfile.gettempdir() + os.sep + video_id
    if not os.path.exists(temp_file_path):
        raise HTTPException(status_code=404, detail="Video not found")

    return StreamingResponse(generate_video_stream(temp_file_path, confidence), media_type="video/mp4")

def detect_objects(image, confidence_threshold):
    global net
    classes_file = "./yolo model/coco.names"
    with open(classes_file, 'r') as file:
        classes = [line.strip() for line in file.readlines()]

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

    if len(indexes) > 0:
        indexes = indexes.flatten()

    results = []
    for i in indexes:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        results.append({"label": label, "confidence": confidences[i], "box": [x, y, w, h]})

    print(f"Total boxes before NMS: {len(boxes)}")
    print(f"Boxes retained after NMS: {len(indexes)}")

    return results, image

def generate_video_stream(video_path, confidence_threshold):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_objects(frame, confidence_threshold)[1]
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

