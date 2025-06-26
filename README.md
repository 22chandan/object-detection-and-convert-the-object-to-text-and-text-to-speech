# 🎯 Real-Time Object Detection with YOLO & Voice Feedback

A high-performance Python application for real-time object detection using YOLOv3 (You Only Look Once) deep learning model with integrated voice feedback system. This project provides both basic and optimized implementations for different performance requirements.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=YOLO&logoColor=black)

## ✨ Features

### 🔍 Detection Capabilities
- **Real-time Object Detection**: Live camera feed processing with YOLOv3
- **High Accuracy**: Detects 80+ object classes from COCO dataset
- **Bounding Box Visualization**: Color-coded rectangles around detected objects
- **Confidence Scoring**: Percentage-based accuracy for each detection
- **Non-Maximum Suppression**: Eliminates duplicate detections

### 🎵 Audio Features
- **Text-to-Speech Integration**: Voice announcements of detected objects
- **Smart Announcement Logic**: Prevents audio spam with intelligent filtering
- **Cross-Platform TTS**: Works on Windows, macOS, and Linux
- **Offline Voice Synthesis**: No internet required for voice feedback

### ⚡ Performance Optimizations
- **GPU Acceleration**: CUDA support for faster inference
- **Frame Skipping**: Process every Nth frame for better FPS
- **Multi-threading**: Background audio processing
- **Memory Optimization**: Efficient queue management and cleanup
- **Adaptive Processing**: Dynamic frame rate adjustment

## 📁 Project Structure

```
d:\ObjectDetection\
├── README.md                              # This file
├── object_detection.py                    # Optimized high-performance version
├── object_detection_with_voice.py         # Basic version with voice feedback
├── detection-system/                      # Modular detection system
│   ├── yolo_detector.py                   # YOLO detection module
│   └── config.py                          # Configuration settings
├── App/                                   # Flutter mobile application
│   └── object_detection/
│       └── pubspec.yaml                   # Flutter dependencies
├── yolo-coco/                             # YOLO model files (to be downloaded)
│   ├── yolov3.weights                     # Pre-trained weights (248MB)
│   ├── yolov3.cfg                         # Network configuration
│   └── coco.names                         # Object class labels
└── docs/                                  # Documentation (optional)
    ├── images/                            # Sample images for testing
    └── screenshots/                       # Application screenshots
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.7+** (recommended: Python 3.8 or higher)
- **Webcam** or camera device
- **Minimum 4GB RAM** (8GB recommended for optimal performance)
- **GPU with CUDA support** (optional, for acceleration)

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd ObjectDetection
   ```

2. **Install Python dependencies**
   ```bash
   pip install opencv-python numpy gtts pygame
   
   # For optimized performance:
   pip install opencv-contrib-python
   ```

3. **Download YOLO model files**
   
   Create the `yolo-coco` directory and download required files:
   ```bash
   mkdir yolo-coco
   cd yolo-coco
   
   # Download YOLOv3 weights (248MB)
   wget https://pjreddie.com/media/files/yolov3.weights
   
   # Download configuration file
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
   
   # Download class labels
   wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
   ```

4. **Install system-specific TTS dependencies**

   **Windows:** PowerShell TTS is built-in
   
   **macOS:** `say` command is built-in
   
   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt-get install espeak espeak-data libespeak1 libespeak-dev
   ```

### Running the Application

#### Basic Version (Recommended for beginners)
```bash
python object_detection_with_voice.py
```

#### Optimized Version (High-performance)
```bash
python object_detection.py
```

#### Modular Version (Advanced)
```bash
cd detection-system
python -c "from yolo_detector import YOLODetector; from config import Config; detector = YOLODetector(Config()); print('Modular system ready')"
```

**Controls:**
- Press `q` to quit the application
- Press `Ctrl+C` for emergency stop

## 🔧 Configuration Options

### Performance Tuning

#### Basic Version (`object_detection_with_voice.py`)
```python
# Adjust detection confidence threshold
if confidence > 0.5:  # Change this value (0.0 - 1.0)

# Modify NMS parameters
idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # confidence, nms_threshold
```

#### Optimized Version (`object_detection.py`)
```python
# Performance settings in FastYOLODetector.__init__
self.input_size = 320              # YOLO input size (320, 416, 608)
self.confidence_threshold = 0.6    # Detection confidence
self.nms_threshold = 0.4           # Non-maximum suppression
self.process_every_n_frames = 3    # Frame processing interval
```

#### Modular Version (`detection-system/config.py`)
```python
# Edit configuration parameters
INPUT_SIZE = 320
CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.4
PROCESS_EVERY_N_FRAMES = 3
```

## 📊 Performance Comparison

| Feature | Basic Version | Optimized Version | Modular Version |
|---------|---------------|-------------------|-----------------|
| **FPS** | 5-10 FPS | 15-25+ FPS | 15-25+ FPS |
| **CPU Usage** | High | Medium | Medium |
| **Memory Usage** | High | Optimized | Optimized |
| **GPU Support** | No | Yes (CUDA) | Yes (CUDA) |
| **Audio Threading** | Blocking | Non-blocking | Configurable |
| **Maintainability** | Basic | Medium | High |

## 🎯 Detected Object Classes

The YOLO model can detect 80 different object classes from the COCO dataset:

**Common Objects:**
- Person, Car, Truck, Bus, Motorcycle, Bicycle
- Dog, Cat, Bird, Horse, Sheep, Cow
- Chair, Couch, Table, Bed, TV, Laptop
- Cell phone, Book, Clock, Vase, Scissors
- And 60+ more classes...

## 📱 Flutter Mobile App

The project includes a Flutter mobile application for object detection:

**Location:** `App/object_detection/`

**Features:**
- Camera integration
- Google ML Kit object detection
- Text-to-speech feedback
- Image picker functionality

**Dependencies:**
- `camera: ^0.10.5+3`
- `google_mlkit_object_detection: ^0.14.0`
- `flutter_tts: ^3.8.5`
- `image_picker: ^1.0.4`

## 🔍 Troubleshooting

### Common Issues

1. **"No module named 'cv2'" Error**
   ```bash
   pip install opencv-python
   # If still failing:
   pip uninstall opencv-python opencv-contrib-python
   pip install opencv-contrib-python
   ```

2. **YOLO files not found**
   ```bash
   # Ensure files are in correct location:
   ls yolo-coco/
   # Should show: yolov3.weights, yolov3.cfg, coco.names
   ```

3. **Camera not detected**
   ```python
   # Try different camera indices
   cap = cv2.VideoCapture(0)  # Try 0, 1, 2, etc.
   ```

4. **Low FPS performance**
   ```python
   # Reduce input size for faster processing
   self.input_size = 320  # Instead of 416 or 608
   
   # Increase frame skipping
   self.process_every_n_frames = 5  # Process every 5th frame
   ```

5. **Audio not working**
   
   **Windows:** Ensure PowerShell execution policy allows scripts
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   
   **Linux:** Install espeak if missing
   ```bash
   sudo apt-get install espeak
   ```

## 🛠️ Development

### Code Structure

#### Basic Version Flow
```
1. Initialize YOLO model and camera
2. Capture frame
3. Process with YOLO
4. Draw detections
5. Generate speech (blocking)
6. Display frame
7. Repeat
```

#### Optimized Version Flow
```
1. Initialize optimized YOLO detector
2. Start background audio thread
3. Capture frame
4. Smart frame processing (skip frames)
5. Queue audio announcements
6. Non-blocking display
7. Repeat with performance monitoring
```

## 📈 Expected Performance

| Hardware | Basic Version FPS | Optimized Version FPS |
|----------|------------------|----------------------|
| Intel i5 + Integrated GPU | 3-6 FPS | 8-12 FPS |
| Intel i7 + GTX 1060 | 6-10 FPS | 15-20 FPS |
| Intel i9 + RTX 3070 | 8-12 FPS | 20-30+ FPS |
| Apple M1/M2 | 5-8 FPS | 12-18 FPS |

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Test your changes thoroughly
4. Submit a pull request with detailed description

### Areas for Contribution
- Performance optimizations
- Additional TTS language support
- GUI interface development
- Mobile deployment improvements
- Custom model training utilities

## 🙏 Acknowledgments

- **Joseph Redmon** - Original YOLO algorithm creator
- **OpenCV Community** - Computer vision library
- **Google** - gTTS (Google Text-to-Speech) API
- **COCO Dataset** - Object detection dataset and annotations

## 🗺️ Roadmap

### Short-term Goals
- [ ] GUI interface with tkinter/PyQt
- [ ] Configuration file support
- [ ] Batch image processing mode
- [ ] Recording and playback features

### Long-term Goals
- [ ] YOLOv4/YOLOv5 model support
- [ ] Custom model training pipeline
- [ ] Web-based interface
- [ ] Enhanced mobile app features

---

**Made with ❤️ using Python, OpenCV, and YOLO**

*For more information about YOLO and object detection, visit the [official YOLO website](https://pjreddie.com/darknet/yolo/).*