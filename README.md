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

- Detects human emotions (e.g., Happy, Sad, Angry) in real-time.
- Works with webcam or video file input.
- Uses pre-trained deep learning models.
- GUI with: (in development)
  - Start menu
  - Source switch (webcam/video file)
  - Screenshot capture functionality
<br>

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow**
- **OpenCV**
- **NumPy**
- **Deepface (with pre-trained model)**
  
## 🖼️ Logo Integration
 Logo is displayed in the bottom-left corner during runtime.

## 📁 Project Structure

```plaintext
RT_Emotion_Detection/
├── assets/                          
│   ├── start_menu.png
│   └── emotion_detection.png
├── models/                          # Pre-trained models and classifiers
│   ├── haarcascade_frontalface_default.xml
│   └── 
├── src/                             # Source code and main app logic
|   ├── main.py                        
│   └── logo_icon.png
├── .gitignore
├── README.md
├── requirements.txt
└── venvi/                           # Virtual environment (excluded from Git)




**⚠️  Please Note: This project is currently under active development.  ⚠️**

Significant changes to the code, features, and functionality may occur.

```


## 📷 Screenshots

Here are some previews of the app in action:

| Start Menu | Emotion Detection |
|------------|-------------------|
| ![Start Menu](assets/start_menu.png) | ![Emotion Detection](assets/emotion_detection.png) |



<br><br>

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





