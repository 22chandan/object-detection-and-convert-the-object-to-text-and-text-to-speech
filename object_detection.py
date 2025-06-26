import cv2
import numpy as np
import threading
import queue
import time
from collections import deque
import subprocess
import platform


# Optimized YOLO Configuration
class FastYOLODetector:
    def __init__(self):
        # YOLO paths
        self.weights_path = "yolo-coco/yolov3.weights"
        self.config_path = "yolo-coco/yolov3.cfg"
        self.labels_path = "yolo-coco/coco.names"

        # Load YOLO model once
        print("Loading YOLO model...")
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)

        # Use GPU if available (CUDA)
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using GPU acceleration")
        else:
            # Use optimized CPU backend
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU (install OpenCV with CUDA for GPU acceleration)")

        # Get output layers
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # Load labels
        with open(self.labels_path, "r") as f:
            self.labels = f.read().strip().split("\n")

        # Performance settings
        self.input_size = 320  # Reduced from 416 for speed (320, 416, 608)
        self.confidence_threshold = 0.6  # Slightly higher for fewer false positives
        self.nms_threshold = 0.4

        # Detection state
        self.last_detections = deque(maxlen=10)  # Store recent detections
        self.detection_queue = queue.Queue(maxsize=2)  # Audio queue
        self.audio_thread = None
        self.running = True

        # Frame processing optimization
        self.process_every_n_frames = 3  # Process every 3rd frame for speed
        self.frame_count = 0

        # Pre-generate colors for speed
        np.random.seed(42)  # Consistent colors
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")

        self.start_audio_thread()

    def start_audio_thread(self):
        """Start background thread for audio processing"""

        def audio_worker():
            while self.running:
                try:
                    text = self.detection_queue.get(timeout=1.0)
                    if text:
                        self.speak_fast(text)
                except queue.Empty:
                    continue
                except:
                    break

        self.audio_thread = threading.Thread(target=audio_worker, daemon=True)
        self.audio_thread.start()

    def speak_fast(self, text):
        """Optimized offline TTS"""
        try:
            system = platform.system().lower()

            if system == "windows":
                # Faster Windows TTS using PowerShell
                subprocess.run([
                    "powershell", "-Command",
                    f"Add-Type -AssemblyName System.Speech; "
                    f"$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$speak.Rate = 2; $speak.Speak('{text}')"
                ], capture_output=True, timeout=3)

            elif system == "darwin":  # macOS
                subprocess.run(["say", "-r", "200", text], timeout=3)  # Faster rate

            elif system == "linux":
                subprocess.run(["espeak", "-s", "180", text], timeout=3)  # Faster speed

        except:
            pass  # Silent fail for speed

    def detect_objects(self, frame):
        """Fast object detection"""
        (H, W) = frame.shape[:2]

        # Create blob with reduced size for speed
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self.input_size, self.input_size),
            swapRB=True, crop=False
        )

        self.net.setInput(blob)

        # Forward pass
        start_time = time.time()
        outputs = self.net.forward(self.output_layers)
        inference_time = time.time() - start_time

        # Process detections
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply NMS
        if boxes:
            idxs = cv2.dnn.NMSBoxes(boxes, confidences,
                                    self.confidence_threshold, self.nms_threshold)
        else:
            idxs = []

        return boxes, confidences, class_ids, idxs, inference_time

    def draw_detections(self, frame, boxes, confidences, class_ids, idxs):
        """Fast drawing of detections"""
        detected_objects = []

        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                class_id = class_ids[i]
                confidence = confidences[i]

                # Use pre-generated colors
                color = [int(c) for c in self.colors[class_id]]

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Draw label (simplified)
                label = self.labels[class_id]
                text = f"{label} {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                detected_objects.append(label)

        return detected_objects

    def should_announce(self, current_objects):
        """Smart announcement logic"""
        if not current_objects:
            return False

        # Add to recent detections
        self.last_detections.append(set(current_objects))

        # Only announce if objects are consistently detected
        if len(self.last_detections) < 3:
            return False

        # Check if objects are stable across recent frames
        recent_sets = list(self.last_detections)[-3:]
        intersection = set.intersection(*recent_sets)

        return len(intersection) > 0

    def run(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)

        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time

        print("Starting fast detection... Press 'q' to quit")

        fps_counter = deque(maxlen=30)

        try:
            while True:
                loop_start = time.time()

                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                detected_objects = []

                # Process only every Nth frame for speed
                if self.frame_count % self.process_every_n_frames == 0:
                    boxes, confidences, class_ids, idxs, inf_time = self.detect_objects(frame)
                    detected_objects = self.draw_detections(frame, boxes, confidences, class_ids, idxs)

                    # Smart audio announcement
                    if self.should_announce(detected_objects):
                        unique_objects = list(set(detected_objects))
                        if unique_objects and not self.detection_queue.full():
                            objects_text = ", ".join(unique_objects[:3])  # Limit to 3 objects
                            try:
                                self.detection_queue.put_nowait(f"Detected: {objects_text}")
                            except queue.Full:
                                pass

                # Calculate and display FPS
                fps_counter.append(time.time() - loop_start)
                avg_fps = len(fps_counter) / sum(fps_counter)

                # Status display
                status = f"FPS: {avg_fps:.1f} | Objects: {len(detected_objects)}"
                cv2.putText(frame, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Fast YOLO Detection", frame)

                # Non-blocking key check
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup(cap)

    def cleanup(self, cap):
        """Clean shutdown"""
        self.running = False
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")


# Run the fast detector
if __name__ == "__main__":
    detector = FastYOLODetector()
    detector.run()