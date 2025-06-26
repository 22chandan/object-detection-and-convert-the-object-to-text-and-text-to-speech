import cv2 # opencv
import numpy as np
from gtts import gTTS
import os

# Load YOLO model
weights_path = "yolo-coco/yolov3.weights"
config_path = "yolo-coco/yolov3.cfg"
labels_path = "yolo-coco/coco.names"

# Load YOLO
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load labels
with open(labels_path, "r") as f:
    labels = f.read().strip().split("\n")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Function for text-to-speech
import tempfile
import pygame
from gtts import gTTS

def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name
            tts.save(temp_path)

        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pass

    except PermissionError as e:
        print(f"Permission error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Detection Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Threshold
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(idxs) > 0:
        detected_objects = []
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            color = [int(c) for c in np.random.randint(0, 255, size=(3,), dtype="uint8")]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detected_objects.append(labels[class_ids[i]])

        if detected_objects:
            unique_objects = ", ".join(set(detected_objects))
            speak(f"I see: {unique_objects}")

    cv2.imshow("YOLO Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

