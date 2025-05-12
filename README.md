# 🎭 Real Time Emotion Detection

A real-time emotion recognition application using a webcam or video file input. This project is built with **Python**, **OpenCV**, **TensorFlow**, and a **Convolutional Neural Network (CNN)** model.
<br>
## 📚Project Description

This project develops a system for real-time emotion detection from video. 

The system uses OpenCV for facial detection and a pre-trained machine learning model to classify the emotions displayed on faces.

The goal is to create a tool that can assist in applications such as:

* User experience analysis
* Communication aids for individuals with communication impairments
* Audience reaction analysis


> 🚧 **This project is under active development.** Features and performance will improve in future versions.
<br>

## 📌 Features
- Real-time webcam or video file input
- Emotion detection using [DeepFace](https://github.com/serengil/deepface)
- Face detection via:
  - Haar Cascade (default)
  - OpenCV DNN (deep learning-based, toggleable in app)
- Project logo overlay
- GUI 
- Screenshot and video source switching 

<br>

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow**
- **OpenCV**
- **NumPy**
- **TKinter**
- **Pillow**
  


## 📁 Project Structure

```plaintext
RT_Emotion_Detection/

├── models/                          # Pre-trained models and classifiers
│   ├──  haarcascade_frontalface_default.xml
|   ├──  opencv_face_detector.pbtxt
│   └──  opencv_face_detector_uint8.pb 
├── src/                             # Source code and main app logic
│   └── main.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── venvi/                           # Virtual environment (excluded from Git)




**⚠️  Please Note: This project is currently under active development.  ⚠️**

Significant changes to the code, features, and functionality may occur.

```


## 🚀 Getting Started

1.  Clone the repository:

    ```bash
    git clone [https://github.com/your_username/your_repo_name.git](https://github.com/your_username/your_repo_name.git)
    ```

2.  Create a virtual environment:

    ```bash
    python3 -m venv venv
    ```

3.  Activate the virtual environment:

    * Linux/macOS:

        ```bash
        source venv/bin/activate
        ```

    * Windows (Git Bash):

        ```bash
        source venv/Scripts/activate
        ```

4.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```
<br>

## 🗝️ Usage

1.  Run the main script:

    ```bash
    python main.py  
    ```
<br>

## 📝 Project Status

This project is currently in development. The following features are still in progress or subject to change:

* Accuracy of emotion recognition
* Stability of facial detection
* User interface
* Performance optimization


## 🙋‍♂️ Author
Created by Idan53780





