import cv2
import numpy as np


weights_path = "yolo model/yolov3.weights"
config_path = "yolo model/yolov3.cfg"
net = cv2.dnn.readNet(weights_path, config_path)


file_handle = open("yolo model/coco.names", "r")
file_lines = file_handle.readlines()


classes = []


for line in file_lines:
   classes.append(line.strip())


file_handle.close()


#print(classes)


layer_names = net.getLayerNames()


output_layers = []
output_layer_indices = net.getUnconnectedOutLayers()


for index in output_layer_indices:
   layer_name = layer_names[index - 1]


   output_layers.append(layer_name)


# Load image
img = cv2.imread("images/slow_streets_on_lake.png")
height, width, channels = img.shape


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


indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


for i in range(len(boxes)):
   if i in indexes:
       x, y, w, h = boxes[i]
       label = str(classes[class_ids[i]])
       confidence = confidences[i]
       color = [0, 255, 0]  # Green for the boxes
       cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
       cv2.putText(img, f"{label} {int(confidence * 100)}%", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)


cv2.imshow("Detect object", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

