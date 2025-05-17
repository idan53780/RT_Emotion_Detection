# 🎭 Real-Time Emotion Detection

A real-time emotion recognition application using webcam, video, or image input. This project is built with **Python**, **OpenCV**, **TensorFlow**, and a **Convolutional Neural Network (CNN)** model.

## 📚 Project Description

This project develops a system for real-time emotion detection from video or static images. It uses OpenCV for face detection and a pre-trained machine learning model to classify emotions displayed on faces.

The goal is to create a tool for applications such as:

- User experience analysis
- Communication aids for individuals with impairments
- Audience reaction analysis

> 🚧 **This project is under active development.** Features and performance will improve in future versions.

## 📌 Features
- Real-time emotion detection via webcam or video file input
- Static image analysis for emotion detection
- Emotion detection using [DeepFace](https://github.com/serengil/deepface)
- Face detection with:
  - Haar Cascade (default)
  - OpenCV DNN (deep learning-based, toggleable in settings)
- Customizable settings (detector, model, confidence threshold, history length)
- Screenshot capture with user-defined save location
- Save analysis results as images
- Project logo overlay in UI
- User-friendly GUI with Tkinter
- Video/image source switching


## 🛠️ Tech Stack
- **Python 3.10+**
- **TensorFlow**
- **OpenCV**
- **NumPy**
- **Tkinter**
- **Pillow**

## 📁 Project Structure

```plaintext
RT_Emotion_Detection/
├── assets/                          # Logo and other static assets
│   └── logo_icon.png
├── Emotion_images/                  # Sample images for testing
├── models/                          # Pre-trained models and classifiers
│   ├── haarcascade_frontalface_default.xml
│   ├── opencv_face_detector.pbtxt
│   └── opencv_face_detector_uint8.pb
├── screenshots/                     # Saved screenshots
├── src/                             # Source code and main app logic
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── processing.py
│   ├── ui.py
│   ├── utils.py
│   └── oldmain.py                  # Legacy main script
├── videos/                          # Sample videos for testing
├── .gitignore
├── LICENSE
├── main.py                          # Entry point
├── README.md
├── requirements.txt
└── venv/                            # Virtual environment (excluded from Git)
```

**⚠️ Note: This project is under active development. Significant changes to code, features, and functionality may occur.**

## 🚀 Getting Started

### Prerequisites
- Python 3.10 - (exactly)
- Git
- Virtual environment (recommended) 

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Idan53780/RT_Emotion_Detection.git
   cd RT_Emotion_Detection
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   - **Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```
   - **Windows (Git Bash)**:
     ```bash
     source venv/Scripts/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Run the main script:
   ```bash
   python main.py
   ```

2. 🪄 Interact with the GUI:
   - Adjust settings (face detector, emotion model, etc.).
   - Select input source (webcam, video, or image).
   - Capture screenshots or save analysis results.
   

## 📝 Project Status
This project is in active development. Current areas of focus include:
- Improving emotion recognition accuracy
- Enhancing facial detection stability
- Optimizing performance for real-time processing
- Refining the user interface
- Adding support for additional input formats

## 🙋‍♂️ Author
Created by [Idan53780](https://github.com/Idan53780)



