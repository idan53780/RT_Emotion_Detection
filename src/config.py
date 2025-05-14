import cv2
import cv2.data

#face detection models

DETECTION_MODELS = {
    "Haar Cascade": cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
    "DNN": None # Intialized in app.py when the user decide to 
}

#Emotion recogntion models

EMOTION_MODELS = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]

#Default set

DEFAULT_DETECTOR = "Haar Cascade"
DEFAULT_EMOTION_MODEL = "VGG-Face"
CONFIDENCE_THRESHOLD = 0.5
HISTORY_LENGTH = 5