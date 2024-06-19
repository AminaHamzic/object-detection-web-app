from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles
from pathlib import Path
import cv2
import numpy as np
import base64
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as necessary for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to hold the loaded model
net = None

@app.on_event("startup")
async def load_model():
    global net
    weights_path = "./yolo model/yolov3.weights"
    config_path = "./yolo model/yolov3.cfg"
    classes_file = "./yolo model/coco.names"
    net = cv2.dnn.readNet(weights_path, config_path)
    if not Path(weights_path).exists() or not Path(config_path).exists() or not Path(classes_file).exists():
        raise FileNotFoundError("Model files are missing.")

frontend_dir = Path(__file__).resolve().parent.parent / 'frontend'
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def read_root():
    return FileResponse(frontend_dir / 'index.html')

@app.post("/uploadfile/")
async def create_upload_file(file_upload: UploadFile = File(...), confidence: float = Form()):
    logging.info(f"Confidence level: {confidence}")
    try:
        data = await file_upload.read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        results, processed_img = detect_objects(img, confidence)
        _, buffer = cv2.imencode('.png', processed_img)
        base64_str = base64.b64encode(buffer).decode("utf-8")
        return {"base64_image": base64_str, "detection_results": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "An unexpected error occurred", "error": str(e)})


def detect_objects(image, confidence_threshold):
    global net
    classes_file = "./yolo model/coco.names"

    # Ensure early exit if the threshold is set to zero
    if confidence_threshold == 0:
        return [], image

    with open(classes_file, 'r') as file:
        classes = [line.strip() for line in file.readlines()]

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    boxes, confidences, class_ids = [], [], []

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]  # get score for each class
            class_id = np.argmax(scores)  # get the class with the highest score
            confidence = scores[class_id]  # get the highest score as confidence

            if confidence >= confidence_threshold:
                print(f"Processing detection with confidence {confidence} and threshold {confidence_threshold}")
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                w = int(detection[2] * image.shape[1])
                h = int(detection[3] * image.shape[0])

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.5)
    if indexes is not None and len(indexes) > 0:
        indexes = indexes.flatten()

    results = []
    # Prepare final results with bounding box coordinates, labels, and confidence
    print(f"Total boxes before NMS: {len(boxes)}")
    if len(indexes) > 0:
        print(f"Boxes retained after NMS: {len(indexes.flatten())}")
        indexes = indexes.flatten()
        for i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), [0, 255, 0], 2)
            cv2.putText(image, f"{label} {confidences[i]:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0],
                        2)
            results.append({"label": label, "confidence": confidences[i], "box": [x, y, w, h]})
    else:
        print("No bounding boxes meet the confidence threshold or NMS criteria.")

    return results, image
