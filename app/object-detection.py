
import cv2
import numpy as np
from pathlib import Path

"""
def detect_objects(image_path):

    weights_path = Path("../yolo model/yolov3.weights")
    config_path = Path("../yolo model/yolov3.cfg")
    net = cv2.dnn.readNet(str(weights_path), str(config_path))

    classes = []
    with open("../yolo model/coco.names", "r") as file:
        classes = [line.strip() for line in file.readlines()]

    # Get output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Load image
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # Create blob and do forward pass
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Parse the outs array
    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Scale the bounding boxes back to the size of the image
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    results = []
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        results.append({"label": label, "confidence": confidence, "box": [x, y, w, h]})

    return results


results = detect_objects("app/uploads/IMG_20201116_002136.jpg")
print(results)

"""