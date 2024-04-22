import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

UPLOAD_DIR = Path() / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR = UPLOAD_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # temporary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).resolve().parent.parent / 'frontend'
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")



@app.get("/")
def read_root():
    return FileResponse(frontend_dir / 'index.html')

@app.post("/uploadfile/")
async def create_upload_file(file_upload: UploadFile = File(...)):
    data = await file_upload.read()

    #print("Received upload:", file_upload.filename)

    save_to = UPLOAD_DIR / file_upload.filename
    with open(save_to, 'wb') as f:
        f.write(data)

    processed_image_path = process_and_save_image(str(save_to))
    return {"url": f"/uploads/processed/{processed_image_path.name}"}


def process_and_save_image(image_path):
    results, processed_img = detect_objects(image_path)
    processed_image_path = PROCESSED_DIR / f"processed_{Path(image_path).name}"
    cv2.imwrite(str(processed_image_path), processed_img)
    return processed_image_path


# New route for object detection
@app.get("/detect/{filename}")
async def detect_objects_in_image(filename: str):
    image_path = UPLOAD_DIR / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    results = detect_objects(str(image_path))
    return {"filename": filename, "detection_results": results}


def detect_objects(image_path):
    weights_path = Path("./yolo model/yolov3.weights").resolve()
    config_path = Path("./yolo model/yolov3.cfg").resolve()
    classes_file = Path("./yolo model/coco.names").resolve()

    if not weights_path.exists() or not config_path.exists() or not classes_file.exists():
        raise FileNotFoundError("YOLO model files are missing, please check the paths.")

    # Load the YOLO model
    net = cv2.dnn.readNet(str(weights_path), str(config_path))

    # Load the classes
    with open(str(classes_file), 'r') as file:
        classes = [line.strip() for line in file.readlines()]

    # Get the output layer names
    layer_names = net.getLayerNames()
    out_layer_indices = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i[0] - 1] for i in out_layer_indices] if out_layer_indices.ndim > 1 else [
        layer_names[i - 1] for i in out_layer_indices]

    #print("Output layers:", output_layers)


    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image file is missing, please check the path.")

    height, width, channels = img.shape

    # Construct a blob from the image
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    results = []

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img, f"{classes[class_ids[i]]} {confidences[i]:.2f}",
            (x, y - 5),
            cv2.FONT_ITALIC,
            2,
            (0, 255, 0), 5)

        results.append({"label": classes[class_ids[i]], "confidence": confidences[i], "box": [x, y, w, h]})

    return results, img


