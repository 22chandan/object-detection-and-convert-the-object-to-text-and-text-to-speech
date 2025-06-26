# camera_manager.py
"""Camera capture and management"""
import cv2


class CameraManager:
    def __init__(self, config):
        self.config = config
        self.cap = None
        self._setup_camera()

    def _setup_camera(self):
        """Initialize and configure camera"""
        self.cap = cv2.VideoCapture(0)

        # Configure camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.CAMERA_BUFFER_SIZE)

    def read_frame(self):
        """Read frame from camera"""
        if self.cap is None:
            return False, None
        return self.cap.read()

    def is_opened(self):
        """Check if camera is open"""
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None


# ================================================================

# detection_system.py
"""Main detection system that coordinates all components"""
import cv2
import time
from collections import deque


class DetectionSystem:
    def __init__(self):
        from config import Config
        from yolo_detector import YOLODetector
        from audio_manager import AudioManager
        from camera_manager import CameraManager

        self.config = Config()
        self.detector = YOLODetector(self.config)
        self.audio_manager = AudioManager(self.config.AUDIO_QUEUE_SIZE)
        self.camera_manager = CameraManager(self.config)

        # Performance tracking
        self.frame_count = 0
        self.fps_counter = deque(maxlen=self.config.FPS_COUNTER_SIZE)

    def run(self):
        """Main detection loop"""
        if not self.camera_manager.is_opened():
            print("Error: Could not open camera")
            return

        print("Starting detection... Press 'q' to quit")

        try:
            while True:
                loop_start = time.time()

                # Read frame
                ret, frame = self.camera_manager.read_frame()
                if not ret:
                    print("Failed to read frame")
                    break

                self.frame_count += 1
                detected_objects = []

                # Process frame (skip frames for performance)
                if self.frame_count % self.config.PROCESS_EVERY_N_FRAMES == 0:
                    detected_objects = self._process_frame(frame)

                # Update display
                self._update_display(frame, detected_objects)

                # Handle user input
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Update FPS counter
                self.fps_counter.append(time.time() - loop_start)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self._cleanup()

    def _process_frame(self, frame):
        """Process a single frame for object detection"""
        # Run detection
        boxes, confidences, class_ids, idxs, inf_time = self.detector.detect_objects(frame)

        # Draw detections
        detected_objects = self.detector.draw_detections(
            frame, boxes, confidences, class_ids, idxs
        )

        # Handle audio announcements
        if self.detector.should_announce(detected_objects):
            unique_objects = list(set(detected_objects))
            if unique_objects:
                objects_text = ", ".join(unique_objects[:self.config.MAX_ANNOUNCED_OBJECTS])
                self.audio_manager.announce(f"Detected: {objects_text}")

        return detected_objects

    def _update_display(self, frame, detected_objects):
        """Update the display window with current frame and stats"""
        # Calculate FPS
        avg_fps = 0
        if self.fps_counter:
            avg_fps = len(self.fps_counter) / sum(self.fps_counter)

        # Display status
        status = f"FPS: {avg_fps:.1f} | Objects: {len(detected_objects)}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLO Object Detection", frame)

    def _cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.audio_manager.stop()
        self.camera_manager.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")
