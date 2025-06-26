# 🔍 Smart Vision - AI Object Detection App

A sophisticated Flutter application that leverages Google ML Kit for real-time object detection and identification. The app features an intuitive interface with camera integration, voice feedback, and advanced AI-powered object recognition capabilities.

![Flutter](https://img.shields.io/badge/Flutter-02569B?style=for-the-badge&logo=flutter&logoColor=white)
![Dart](https://img.shields.io/badge/Dart-0175C2?style=for-the-badge&logo=dart&logoColor=white)
![ML Kit](https://img.shields.io/badge/Google_ML_Kit-4285F4?style=for-the-badge&logo=google&logoColor=white)

## ✨ Features

### 🎯 Core Functionality
- **Real-time Object Detection**: Live camera feed with continuous object identification
- **Capture Mode**: Single-shot photo analysis with detailed results
- **Gallery Integration**: Analyze existing photos from device gallery
- **Voice Feedback**: Text-to-speech announcements of detected objects
- **Confidence Scoring**: Percentage-based accuracy for each detection

### 🎨 User Experience
- **Modern UI Design**: Beautiful gradient-based interface with smooth animations
- **Multiple Detection Modes**: Switch between capture, real-time, and gallery modes
- **Interactive Results**: Visual progress indicators and confidence meters
- **Responsive Design**: Optimized for various screen sizes and orientations

### 🔧 Technical Features
- **Advanced ML Integration**: Google ML Kit Image Labeling with custom confidence thresholds
- **Camera Controls**: Full camera integration with preview and capture capabilities
- **Permission Management**: Automatic handling of camera, storage, and microphone permissions
- **Error Handling**: Robust error management with user-friendly feedback
- **Performance Optimized**: Efficient image processing and memory management

## 📱 Screenshots

| Home Screen | Real-time Detection | Results View |
|------------|-------------------|--------------|
| ![Home](docs/screenshots/home.png) | ![Detection](docs/screenshots/detection.png) | ![Results](docs/screenshots/results.png) |

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Flutter SDK**: Version 3.6.0 or higher
- **Dart SDK**: Version 3.6.0 or higher
- **Android Studio** or **VS Code** with Flutter extensions
- **iOS Development** (for iOS deployment):
  - Xcode 12.0 or higher
  - iOS deployment target 12.0+
- **Android Development**:
  - Android SDK with API level 21+
  - Android device or emulator

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/object-detection-app.git
   cd object-detection-app
   ```

2. **Install dependencies**
   ```bash
   flutter pub get
   ```

3. **Configure platform-specific settings**

   #### Android Configuration
   - Ensure `minSdkVersion` is set to 21 or higher in `android/app/build.gradle`
   - Camera and storage permissions are already configured in `AndroidManifest.xml`

   #### iOS Configuration
   - Update `ios/Runner/Info.plist` with camera and microphone usage descriptions:
   ```xml
   <key>NSCameraUsageDescription</key>
   <string>This app needs camera access to detect objects in real-time</string>
   <key>NSMicrophoneUsageDescription</key>
   <string>This app uses microphone for voice feedback features</string>
   <key>NSPhotoLibraryUsageDescription</key>
   <string>This app needs photo library access to analyze existing images</string>
   ```

4. **Run the application**
   ```bash
   flutter run
   ```

## 📋 Dependencies

### Core Dependencies
```yaml
dependencies:
  flutter:
    sdk: flutter
  camera: ^0.10.5+3                    # Camera functionality
  google_mlkit_image_labeling: ^0.13.0 # ML Kit image labeling
  google_mlkit_object_detection: ^0.14.0 # Object detection
  flutter_tts: ^3.8.5                  # Text-to-speech
  image_picker: ^1.0.4                 # Gallery image selection
  permission_handler: ^11.3.1          # Runtime permissions
  path_provider: ^2.1.2                # File system paths
  path: ^1.9.0                         # Path manipulation
```

### Development Dependencies
```yaml
dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^5.0.0                # Code linting
```

## 🏗️ Project Structure

```
lib/
├── main.dart                 # Application entry point and main UI
├── models/                   # Data models (if expanded)
├── services/                 # ML Kit and camera services (if separated)
├── widgets/                  # Reusable UI components (if separated)
└── utils/                    # Helper functions and constants (if added)

android/
├── app/
│   ├── build.gradle         # Android build configuration
│   └── src/main/
│       └── AndroidManifest.xml # Android permissions and configuration

ios/
├── Runner/
│   ├── Info.plist          # iOS permissions and configuration
│   └── Runner.xcodeproj/   # Xcode project configuration
```

## 🎮 Usage Guide

### Detection Modes

1. **Capture Mode** (Default)
   - Point camera at objects
   - Tap "Capture & Detect" button
   - View detailed results with confidence scores

2. **Real-time Mode**
   - Toggle "Real-time" button
   - Automatic detection every 2 seconds
   - Live feedback with voice announcements

3. **Gallery Mode**
   - Tap "Gallery" button
   - Select image from device storage
   - Analyze pre-existing photos

### Features Overview

- **Voice Toggle**: Enable/disable text-to-speech feedback
- **Confidence Threshold**: Currently set to 50% minimum confidence
- **Object Icons**: Context-aware icons for different object types
- **Progress Indicators**: Visual feedback during processing
- **Error Handling**: Graceful error messages and recovery

## 🔧 Configuration

### ML Kit Settings
```dart
// Adjust confidence threshold in main.dart
_imageLabeler = ImageLabeler(
  options: ImageLabelerOptions(
    confidenceThreshold: 0.5  // Change this value (0.0 - 1.0)
  )
);
```

### Text-to-Speech Settings
```dart
// Customize TTS parameters in _initTTS() method
await flutterTts.setLanguage("en-US");    // Language
await flutterTts.setSpeechRate(0.5);      // Speed (0.0 - 1.0)
await flutterTts.setVolume(1.0);          // Volume (0.0 - 1.0)
await flutterTts.setPitch(1.0);           // Pitch (0.0 - 2.0)
```

## 🐛 Troubleshooting

### Common Issues

1. **Camera not working**
   - Ensure camera permissions are granted
   - Check device camera functionality
   - Restart the application

2. **ML Kit detection errors**
   - Verify internet connection for first-time model download
   - Ensure sufficient device storage
   - Check Google Play Services (Android)

3. **TTS not working**
   - Verify microphone permissions
   - Check device TTS settings
   - Ensure language pack is installed

4. **Build errors**
   - Run `flutter clean && flutter pub get`
   - Update Flutter SDK to latest stable version
   - Check platform-specific requirements

### Performance Tips

- **Close other camera apps** before using the application
- **Ensure good lighting** for better detection accuracy
- **Keep objects in clear view** and avoid motion blur
- **Restart app periodically** for optimal performance

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow Flutter/Dart coding standards
- Add comments for complex logic
- Test on both Android and iOS platforms
- Update documentation for new features
- Ensure responsive design principles

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google ML Kit** for powerful on-device machine learning
- **Flutter Team** for the amazing cross-platform framework
- **Camera Plugin Contributors** for camera integration
- **Community Contributors** for ongoing improvements

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/object-detection-app/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/object-detection-app/discussions)
- **Email**: your.email@example.com

## 🗺️ Roadmap

### Upcoming Features
- [ ] Custom model training support
- [ ] Object tracking and counting
- [ ] Multiple language support
- [ ] Cloud-based advanced models
- [ ] Batch image processing
- [ ] Export detection results
- [ ] Social sharing capabilities
- [ ] Accessibility improvements

### Version History
- **v1.0.0**: Initial release with basic object detection
- **v1.1.0**: Added real-time detection mode
- **v1.2.0**: Improved UI/UX and performance optimization

---

**Made with ❤️ using Flutter and Google ML Kit**

*For more information about Flutter development, visit the [official documentation](https://docs.flutter.dev/).*
