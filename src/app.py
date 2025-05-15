import os
import cv2
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import threading
from .config import DETECTION_MODELS, EMOTION_MODELS, DEFAULT_DETECTOR, DEFAULT_EMOTION_MODEL, CONFIDENCE_THRESHOLD, HISTORY_LENGTH
from .ui import create_start_screen, open_settings, create_app_interface,create_image_analysis_interface
from .processing import process_video, _process_image_thread,process_image
from .utils import get_smoothed_emotion, take_screenshot, save_analysis

class EmotionDetectionApp:
    def __init__(self, root):
        print("__init__: EmotionDetectionApp initialized")
        self.root = root
        self.root.title("EmotionLens - Real-Time Emotion Detection")
        self.root.geometry("1200x700")
        self.root.configure(bg="#2c3e50")
        self.root.minsize(1000, 600)
        
        # Load logo
        try:
            self.logo_img = Image.open('assets/logo_icon.png')
            self.logo_img = self.logo_img.resize((60, 60), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(self.logo_img)
        except Exception as e:
            print(f"Could not load logo: {e}")
            self.logo_photo = None
        
        # Variables
        self.cap = None
        self.is_running = False
        self.thread = None
        self.input_source = "webcam"
        self.current_frame = None
        self.frame_with_detection = None
        self.emotion_history = []
        self.detection_models = DETECTION_MODELS
        self.current_detector = DEFAULT_DETECTOR
        self.emotion_models = EMOTION_MODELS
        self.current_emotion_model = DEFAULT_EMOTION_MODEL
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.history_length = HISTORY_LENGTH
        
        # Create start screen
        create_start_screen(self)
    
    def save_settings(self, detector, emotion_model, threshold, history_length, window):
        self.current_detector = detector
        self.current_emotion_model = emotion_model
        self.confidence_threshold = threshold
        self.history_length = history_length
        
        if self.current_detector == "DNN" and self.detection_models["DNN"] is None:
            try:
                model_file = "models/opencv_face_detector_uint8.pb"
                config_file = "models/opencv_face_detector.pbtxt"
                
                if os.path.exists(model_file) and os.path.exists(config_file):
                    self.detection_models["DNN"] = cv2.dnn.readNetFromTensorflow(model_file, config_file)
                else:
                    messagebox.showwarning(
                        "Model Not Found", 
                        "DNN model files not found. Please download them or use Haar Cascade instead."
                    )
                    self.current_detector = "Haar Cascade"
            except Exception as e:
                messagebox.showerror("Error", f"Could not load DNN model: {str(e)}")
                self.current_detector = "Haar Cascade"
        
        window.destroy()
        messagebox.showinfo("Settings Saved", "Your settings have been saved successfully.")
    
    def start_detection(self, source_type):
        print(f"start_detection called with source_type: {source_type}")
        self.input_source = source_type
        
        if source_type == "webcam":
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
        elif source_type == "video":
            video_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
            )
            if not video_path:
                return
            
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video file")
                return
        elif source_type == "image":
            image_path = filedialog.askopenfilename(
                title="Select Image File",
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
            )
            if not image_path:
                return
            
            try:
                self.current_frame = cv2.imread(image_path)
                if self.current_frame is None: 
                    messagebox.showerror("Error", "Could not open image file")
                    return
                
                #process_image(self)
                create_image_analysis_interface(self)
                self.is_running = True
                self.thread = threading.Thread(target=lambda: _process_image_thread(self))
                self.thread.daemon = True
                self.thread.start()
                
                print("start_detection (image): After process_image")
                return
            
            except Exception as e:
                messagebox.showerror("Error", f"Error processing image: {str(e)}")
                return
        
        create_app_interface(self)
        
        self.is_running = True
        self.thread = threading.Thread(target=lambda: process_video(self))
        self.thread.daemon = True
        self.thread.start()
        
    
    def update_display(self):
        if self.frame_with_detection is None or not self.is_running:
            return
            
        try:
            frame_rgb = cv2.cvtColor(self.frame_with_detection, cv2.COLOR_BGR2RGB)
            
            try:
                video_frame_width = self.video_frame.winfo_width()
                video_frame_height = self.video_frame.winfo_height()
            except:
                video_frame_width = 640
                video_frame_height = 480
            
            if video_frame_width <= 1 or video_frame_height <= 1:
                video_frame_width = 640
                video_frame_height = 480
                
            h, w = frame_rgb.shape[:2]
            scale_w = video_frame_width / w
            scale_h = video_frame_height / h
            scale = min(scale_w, scale_h)
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            img = Image.fromarray(frame_resized)
            img_tk = ImageTk.PhotoImage(image=img)
            
            if self.is_running and hasattr(self, 'video_frame') and self.video_frame.winfo_exists():
                self.video_frame.config(image=img_tk)
                self.video_frame.image = img_tk
        except Exception as e:
            print(f"Error updating display: {e}")
    
    def take_screenshot(self):
        take_screenshot(self)
    
    def save_analysis(self):
        save_analysis(self)
    
    def get_smoothed_emotion(self):
        return get_smoothed_emotion(self)
    
    def stop_and_return(self):
        self.is_running = False
        if self.thread is not None:
            self.thread.join(1.0)
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.current_frame = None
        self.frame_with_detection = None
        self.emotion_history = []
        
        create_start_screen(self)
    
    def create_start_screen(self):
        create_start_screen(self)
    
    def open_settings(self):
        open_settings(self)