"""Configuration settings for YOLO detection system"""


class Config:
    # YOLO Model Paths
    WEIGHTS_PATH = "../yolo-coco/yolov3.weights"
    CONFIG_PATH = "../yolo-coco/yolov3.cfg"
    LABELS_PATH = "../yolo-coco/coco.names"

    # Detection Parameters
    INPUT_SIZE = 320  # 320, 416, or 608
    CONFIDENCE_THRESHOLD = 0.6
    NMS_THRESHOLD = 0.4
    PROCESS_EVERY_N_FRAMES = 3

    # Camera Settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    CAMERA_BUFFER_SIZE = 1

    # Audio Settings
    DETECTION_HISTORY_SIZE = 10
    AUDIO_QUEUE_SIZE = 2
    MAX_ANNOUNCED_OBJECTS = 3

    # Performance
    FPS_COUNTER_SIZE = 30


