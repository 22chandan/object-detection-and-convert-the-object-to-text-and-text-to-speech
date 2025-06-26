"""YOLO object detection module"""
import cv2
import numpy as np
import time
from collections import deque


class YOLODetector:
    def __init__(self, config):
        self.config = config
        self._load_model()
        self._setup_detection_state()

    def _load_model(self):
        """Load YOLO model and configure backend"""
        print("Loading YOLO model...")
        self.net = cv2.dnn.readNetFromDarknet(
            self.config.CONFIG_PATH,
            self.config.WEIGHTS_PATH
        )

        # Configure backend (GPU if available)
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using GPU acceleration")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU backend")

        # Get output layers
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # Load class labels
        with open(self.config.LABELS_PATH, "r") as f:
            self.labels = f.read().strip().split("\n")

    def _setup_detection_state(self):
        """Initialize detection state and colors"""
        self.last_detections = deque(maxlen=self.config.DETECTION_HISTORY_SIZE)

        # Pre-generate colors for consistent visualization
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")

    def detect_objects(self, frame):
        """Perform object detection on frame"""
        H, W = frame.shape[:2]

        # Create blob
        blob = cv2.dnn.blobFromImage(
            frame,
            1 / 255.0,
            (self.config.INPUT_SIZE, self.config.INPUT_SIZE),
            swapRB=True,
            crop=False
        )

        # Run inference
        self.net.setInput(blob)
        start_time = time.time()
        outputs = self.net.forward(self.output_layers)
        inference_time = time.time() - start_time

        # Process detections
        boxes, confidences, class_ids = self._process_detections(outputs, W, H)

        # Apply Non-Maximum Suppression
        idxs = []
        if boxes:
            idxs = cv2.dnn.NMSBoxes(
                boxes, confidences,
                self.config.CONFIDENCE_THRESHOLD,
                self.config.NMS_THRESHOLD
            )

        return boxes, confidences, class_ids, idxs, inference_time

    def _process_detections(self, outputs, width, height):
        """Process raw YOLO outputs"""
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.config.CONFIDENCE_THRESHOLD:
                    box = detection[0:4] * np.array([width, height, width, height])
                    centerX, centerY, w, h = box.astype("int")

                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids

    def draw_detections(self, frame, boxes, confidences, class_ids, idxs):
        """Draw bounding boxes and labels on frame"""
        detected_objects = []

        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                confidence = confidences[i]

                # Get color and label
                color = [int(c) for c in self.colors[class_id]]
                label = self.labels[class_id]

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Draw label
                text = f"{label} {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                detected_objects.append(label)

        return detected_objects

    def should_announce(self, current_objects):
        """Determine if current detections should be announced"""
        if not current_objects:
            return False

        # Add to detection history
        self.last_detections.append(set(current_objects))

        # Need at least 3 frames of history
        if len(self.last_detections) < 3:
            return False

        # Check if objects are stable across recent frames
        recent_sets = list(self.last_detections)[-3:]
        stable_objects = set.intersection(*recent_sets)

        return len(stable_objects) > 0


