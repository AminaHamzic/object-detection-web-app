import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles



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

    # Convert image data to numpy array
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"message": "Invalid image file"})

    # Process the image
    results, processed_img = detect_objects(img)

    # Try to convert processed image to Base64, specify .png as the encoding format
    try:
        _, buffer = cv2.imencode('.png', processed_img)
        base64_str = base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "Failed to encode image", "error": str(e)})

    return {"base64_image": base64_str, "detection_results": results, "url": ""}

#@app.get("/detect/{filename}")
#async def detect_objects_in_image(filename: str):
 #   image_path = UPLOAD_DIR / filename
  #  if not image_path.exists():
   #     raise HTTPException(status_code=404, detail="File not found")

    #results = detect_objects(str(image_path))
    #return {"filename": filename, "detection_results": results}

def detect_objects(image):
    weights_path = "./yolo model/yolov3.weights"
    config_path = "./yolo model/yolov3.cfg"
    classes_file = "./yolo model/coco.names"

    # Ensure model files exist
    if not Path(weights_path).exists() or not Path(config_path).exists() or not Path(classes_file).exists():
        raise FileNotFoundError("YOLO model files are missing, please check the paths.")

    net = cv2.dnn.readNet(weights_path, config_path)

    with open(classes_file, 'r') as file:
        classes = [line.strip() for line in file.readlines()]

    layer_names = net.getLayerNames()
    out_layer_indices = net.getUnconnectedOutLayers()

    flat_indices = [i[0] - 1 if isinstance(i, np.ndarray) else i - 1 for i in out_layer_indices.flatten()]
    output_layers = [layer_names[i] for i in flat_indices]

    if not isinstance(image, np.ndarray):
        # Attempt to read the image if a path is provided instead of an image array
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise FileNotFoundError("Image file could not be loaded, check the path.")
        else:
            raise ValueError("Invalid input: image must be a path (str) or an ndarray.")

    height, width, channels = image.shape

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    score_threshold = 0.5
    nms_threshold = 0.4
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)

    results = []
    if indexes is not None and len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{classes[class_ids[i]]} {confidences[i]:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            results.append({"label": classes[class_ids[i]], "confidence": confidences[i], "box": [x, y, w, h]})

    return results, image
