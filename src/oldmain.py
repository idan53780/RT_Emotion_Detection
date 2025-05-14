"""
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime
import threading
from deepface import DeepFace

class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EmotionLens - Real-Time Emotion Detection")
        self.root.geometry("1200x700")
        self.root.configure(bg="#2c3e50")
        self.root.minsize(1000, 600)
        
        # Load logo
        try:
            self.logo_img = Image.open('assests/logo_icon.png')
            self.logo_img = self.logo_img.resize((60, 60), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(self.logo_img)
        except Exception as e:
            print(f"Could not load logo: {e}")
            self.logo_photo = None
        
        # Variables
        self.cap = None
        self.is_running = False
        self.thread = None
        self.input_source = "webcam"  # Default to webcam
        self.current_frame = None
        self.frame_with_detection = None
        self.emotion_history = []  # For temporal smoothing
        self.history_length = 5  # Number of frames to consider for smoothing
        self.detection_models = {
            "Haar Cascade": cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
            "DNN": None  # Will be initialized on when toggled
        }
        self.current_detector = "Haar Cascade"
        self.emotion_models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
        self.current_emotion_model = "VGG-Face"
        self.confidence_threshold = 0.5  # Minimum confidence to display an emotion
        
        # Create frames
        self.create_start_screen()
        
    def create_start_screen(self):
        # Clear the root window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Main frame
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Title with logo
        title_frame = tk.Frame(main_frame, bg="#2c3e50")
        title_frame.pack(pady=20)
        
        if self.logo_photo:
            logo_label = tk.Label(title_frame, image=self.logo_photo, bg="#2c3e50")
            logo_label.pack(side=tk.LEFT, padx=10)
        
        title_label = tk.Label(
            title_frame, 
            text="EmotionLens", 
            font=("Helvetica", 36, "bold"), 
            fg="#ecf0f1", 
            bg="#2c3e50"
        )
        title_label.pack(side=tk.LEFT)
        
        subtitle = tk.Label(
            main_frame, 
            text="Real-Time Emotion Detection", 
            font=("Helvetica", 18), 
            fg="#bdc3c7", 
            bg="#2c3e50"
        )
        subtitle.pack(pady=(0, 40))
        
        # Buttons frame
        button_frame = tk.Frame(main_frame, bg="#2c3e50")
        button_frame.pack(pady=20)
        
        # Button style
        button_style = {
            "font": ("Helvetica", 14),
            "bg": "#3498db",
            "fg": "white",
            "activebackground": "#2980b9",
            "activeforeground": "white",
            "width": 20,
            "height": 2,
            "borderwidth": 0,
            "cursor": "hand2"
        }
        
        # Start with webcam button
        webcam_btn = tk.Button(
            button_frame, 
            text="Start with Webcam", 
            command=lambda: self.start_detection("webcam"),
            **button_style
        )
        webcam_btn.pack(pady=10)
        
        # Start with video file button
        video_btn = tk.Button(
            button_frame, 
            text="Start with Video File", 
            command=lambda: self.start_detection("video"),
            **button_style
        )
        video_btn.pack(pady=10)
        
        # Load image button
        image_btn = tk.Button(
            button_frame, 
            text="Analyze Image", 
            command=lambda: self.start_detection("image"),
            **button_style
        )
        image_btn.pack(pady=10)
        
        # Settings button
        settings_btn = tk.Button(
            button_frame, 
            text="Settings", 
            command=self.open_settings,
            **button_style
        )
        settings_btn.pack(pady=10)
        
        # Exit button
        exit_btn = tk.Button(
            button_frame, 
            text="Exit", 
            command=self.root.quit,
            **button_style
        )
        exit_btn.pack(pady=10)
        
        #  version and author
        footer = tk.Label(
            main_frame, 
            text="v1.0.0 | Created by Idan53780", 
            font=("Helvetica", 10), 
            fg="#95a5a6", 
            bg="#2c3e50"
        )
        footer.pack(side=tk.BOTTOM, pady=10)
        
    def open_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("EmotionLens Settings")
        settings_window.geometry("600x600")
        settings_window.configure(bg="#2c3e50")
        settings_window.grab_set()  # Modal window
        
        # Settings frame
        settings_frame = tk.Frame(settings_window, bg="#2c3e50", padx=20, pady=20)
        settings_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = tk.Label(
            settings_frame, 
            text="Settings", 
            font=("Helvetica", 24, "bold"), 
            fg="#ecf0f1", 
            bg="#2c3e50"
        )
        title.pack(pady=(0, 20))
        
        # Face Detection Model Selection
        face_detector_frame = tk.LabelFrame(
            settings_frame, 
            text="Face Detection Model", 
            font=("Helvetica", 12), 
            fg="#ecf0f1", 
            bg="#2c3e50",
            padx=10, 
            pady=10
        )
        face_detector_frame.pack(fill=tk.X, pady=10)
        
        detector_var = tk.StringVar(value=self.current_detector)
        
        for i, detector in enumerate(["Haar Cascade", "DNN"]):
            rb = tk.Radiobutton(
                face_detector_frame,
                text=detector,
                variable=detector_var,
                value=detector,
                bg="#2c3e50",
                fg="#ecf0f1",
                selectcolor="#2c3e50",
                activebackground="#2c3e50",
                activeforeground="#ecf0f1",
                font=("Helvetica", 12)
            )
            rb.pack(anchor=tk.W)
        
        # Emotion Model Selection
        emotion_model_frame = tk.LabelFrame(
            settings_frame, 
            text="Emotion Recognition Model", 
            font=("Helvetica", 12), 
            fg="#ecf0f1", 
            bg="#2c3e50",
            padx=10, 
            pady=10
        )
        emotion_model_frame.pack(fill=tk.X, pady=10)
        
        emotion_model_var = tk.StringVar(value=self.current_emotion_model)
        emotion_model_combo = ttk.Combobox(
            emotion_model_frame,
            textvariable=emotion_model_var,
            values=self.emotion_models,
            font=("Helvetica", 12),
            state="readonly"
        )
        emotion_model_combo.pack(fill=tk.X, pady=5)
        
        # Confidence Threshold
        threshold_frame = tk.LabelFrame(
            settings_frame, 
            text="Confidence Threshold", 
            font=("Helvetica", 12), 
            fg="#ecf0f1", 
            bg="#2c3e50",
            padx=10, 
            pady=10
        )
        threshold_frame.pack(fill=tk.X, pady=10)
        
        threshold_var = tk.DoubleVar(value=self.confidence_threshold)
        threshold_scale = tk.Scale(
            threshold_frame,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=threshold_var,
            bg="#2c3e50",
            fg="#ecf0f1",
            highlightthickness=0,
            troughcolor="#34495e",
            activebackground="#3498db",
            font=("Helvetica", 12)
        )
        threshold_scale.pack(fill=tk.X)
        
        # History length for temporal smoothing
        history_frame = tk.LabelFrame(
            settings_frame, 
            text="Temporal Smoothing Frames", 
            font=("Helvetica", 12), 
            fg="#ecf0f1", 
            bg="#2c3e50",
            padx=10, 
            pady=10
        )
        history_frame.pack(fill=tk.X, pady=10)
        
        history_var = tk.IntVar(value=self.history_length)
        history_scale = tk.Scale(
            history_frame,
            from_=1,
            to=10,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=history_var,
            bg="#2c3e50",
            fg="#ecf0f1",
            highlightthickness=0,
            troughcolor="#34495e",
            activebackground="#3498db",
            font=("Helvetica", 12)
        )
        history_scale.pack(fill=tk.X)
        
        # Save button
        save_btn = tk.Button(
            settings_frame,
            text="Save Settings",
            command=lambda: self.save_settings(
                detector_var.get(),
                emotion_model_var.get(),
                threshold_var.get(),
                history_var.get(),
                settings_window
            ),
            font=("Helvetica", 14),
            bg="#2ecc71",
            fg="white",
            activebackground="#27ae60",
            activeforeground="white",
            width=15,
            height=1,
            borderwidth=0,
            cursor="hand2"
        )
        save_btn.pack(pady=20)
    
    def save_settings(self, detector, emotion_model, threshold, history_length, window):
        self.current_detector = detector
        self.current_emotion_model = emotion_model
        self.confidence_threshold = threshold
        self.history_length = history_length
        
        # If DNN is selected, load it if not already loaded
        if self.current_detector == "DNN" and self.detection_models["DNN"] is None:
            try:
                # Load a pre-trained face detection model
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
                return  # User cancelled
            
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
                return  # User cancelled
            
            try:
                self.current_frame = cv2.imread(image_path)
                if self.current_frame is None:
                    messagebox.showerror("Error", "Could not open image file")
                    return
                self.process_image()
                return
            except Exception as e:
                messagebox.showerror("Error", f"Error processing image: {str(e)}")
                return
        
        # Create the main app interface
        self.create_app_interface()
        
        # Start processing
        self.is_running = True
        self.thread = threading.Thread(target=self.process_video)
        self.thread.daemon = True
        self.thread.start()
    
    def create_app_interface(self):
        # Clear the root window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Main frame
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Top control panel
        control_panel = tk.Frame(main_frame, bg="#34495e", height=50)
        control_panel.pack(fill=tk.X)
        
        # If using logo
        if self.logo_photo:
            logo_label = tk.Label(control_panel, image=self.logo_photo, bg="#34495e")
            logo_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        title_label = tk.Label(
            control_panel, 
            text="EmotionLens", 
            font=("Helvetica", 16, "bold"), 
            fg="#ecf0f1", 
            bg="#34495e"
        )
        title_label.pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        btn_frame = tk.Frame(control_panel, bg="#34495e")
        btn_frame.pack(side=tk.RIGHT, padx=10)
        
        # Screenshot button
        screenshot_btn = tk.Button(
            btn_frame,
            text="Screenshot",
            command=self.take_screenshot,
            font=("Helvetica", 12),
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            activeforeground="white",
            padx=10,
            borderwidth=0
        )
        screenshot_btn.pack(side=tk.LEFT, padx=5)
        
        # Return to menu button
        menu_btn = tk.Button(
            btn_frame,
            text="Back to Menu",
            command=self.stop_and_return,
            font=("Helvetica", 12),
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            activeforeground="white",
            padx=10,
            borderwidth=0
        )
        menu_btn.pack(side=tk.LEFT, padx=5)
        
        # Video display frame
        self.video_frame = tk.Label(main_frame, bg="black")
        self.video_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Status bar
        status_frame = tk.Frame(main_frame, bg="#34495e", height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(
            status_frame, 
            text="Ready", 
            font=("Helvetica", 10), 
            fg="#ecf0f1", 
            bg="#34495e",
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Version info
        version_label = tk.Label(
            status_frame, 
            text="v1.0.0", 
            font=("Helvetica", 10), 
            fg="#ecf0f1", 
            bg="#34495e",
            anchor=tk.E
        )
        version_label.pack(side=tk.RIGHT, padx=10, pady=5)
    
    def process_video(self):
        #Process video frames in a separate thread
        try:
            while self.is_running and self.cap is not None:
                ret, frame = self.cap.read()
                if not ret:
                    if self.input_source == "webcam":
                        # Try to reconnect to webcam
                        self.cap.release()
                        self.cap = cv2.VideoCapture(0)
                        if not self.cap.isOpened():
                            print("Lost connection to webcam")
                            self.root.after(0, self.stop_and_return)
                            break
                    else:
                        # End of video file
                        print("End of video file")
                        self.root.after(0, self.stop_and_return)
                        break
                
                self.current_frame = frame.copy()
                self.process_frame()
                
                # Small delay to reduce CPU usage
                cv2.waitKey(10)
                
        except Exception as e:
            print(f"Error in video processing: {e}")
            # Use root.after to safely update UI from this thread
            if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                self.root.after(0, lambda: self.status_label.config(text=f"Error: {str(e)}"))
        finally:
            if self.cap is not None:
                self.cap.release()
    
    def process_frame(self):
        #Process a single frame for emotion detection
        if self.current_frame is None:
            return
        
        frame = self.current_frame.copy()
        self.frame_with_detection = frame.copy()
        
        # Convert to RGB for DeepFace and grayscale for face detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = []
        
        # Detect faces
        if self.current_detector == "Haar Cascade":
            faces_rect = self.detection_models["Haar Cascade"].detectMultiScale(gray, 1.1, 4)
            faces = [(x, y, x+w, y+h) for (x, y, w, h) in faces_rect]
        elif self.current_detector == "DNN" and self.detection_models["DNN"] is not None:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
            self.detection_models["DNN"].setInput(blob)
            detections = self.detection_models["DNN"].forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    faces.append((x1, y1, x2, y2))
        
        # Process each face
        for (x1, y1, x2, y2) in faces:
            try:
                # Ensure coordinates are within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue  # Skip invalid face regions
                
                # Draw rectangle
                cv2.rectangle(self.frame_with_detection, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Extract face region for emotion analysis
                face_roi = rgb_frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue
                
                # Analyze emotion
                result = DeepFace.analyze(
                    face_roi,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='skip',  # Skip detection as we already have face coordinates
                    silent=True
                )[0]
                
                # Get emotion and confidence
                emotion = result['dominant_emotion']
                emotions = result['emotion']
                confidence = emotions[emotion] / 100.0  # Convert percentage to [0,1]
                
                # Add to history for smoothing
                self.emotion_history.append((emotion, confidence))
                if len(self.emotion_history) > self.history_length:
                    self.emotion_history.pop(0)
                
                # Apply temporal smoothing
                smoothed_emotion = self.get_smoothed_emotion()
                
                # Only display if confidence is above threshold
                if smoothed_emotion[1] >= self.confidence_threshold:
                    # Display emotion text
                    emotion_text = f"{smoothed_emotion[0]} ({int(smoothed_emotion[1]*100)}%)"
                    cv2.putText(
                        self.frame_with_detection,
                        emotion_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
                    
                    # Visual confidence meter
                    meter_width = x2 - x1
                    filled_width = int(meter_width * smoothed_emotion[1])
                    cv2.rectangle(
                        self.frame_with_detection,
                        (x1, y2 + 5),
                        (x1 + meter_width, y2 + 15),
                        (0, 0, 255),
                        1
                    )
                    cv2.rectangle(
                        self.frame_with_detection,
                        (x1, y2 + 5),
                        (x1 + filled_width, y2 + 15),
                        (0, 255, 0),
                        cv2.FILLED
                    )
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        # Add a watermark/logo
        if self.logo_photo:
            h_frame, w_frame = self.frame_with_detection.shape[:2]
            logo_np = cv2.cvtColor(np.array(self.logo_img), cv2.COLOR_RGBA2BGRA)
            h_logo, w_logo = logo_np.shape[:2]
            x_offset = 10
            y_offset = h_frame - h_logo - 10
            
            # Add logo to frame
            if logo_np.shape[2] == 4:  # Has alpha channel
                alpha_logo = logo_np[:, :, 3] / 255.0
                alpha_frame = 1.0 - alpha_logo
                
                for c in range(0, 3):
                    self.frame_with_detection[y_offset:y_offset+h_logo, x_offset:x_offset+w_logo, c] = (
                        alpha_logo * logo_np[:, :, c] + 
                        alpha_frame * self.frame_with_detection[y_offset:y_offset+h_logo, x_offset:x_offset+w_logo, c]
                    )
        
        # Update the display
        self.update_display()
    
    def process_image(self):
        #Process a single image for emotion detection
        if self.current_frame is None:
            return
        
        # Create the analysis interface
        self.create_image_analysis_interface()
        
        # Start processing in a separate thread to avoid UI freeze
        threading.Thread(target=self._process_image_thread).start()
    
    def _process_image_thread(self):
        #Image processing in a separate thread
        try:
            # Process the frame
            self.process_frame()
            
            # Update the status
            self.root.after(0, lambda: self.status_label.config(text="Analysis complete"))
        except Exception as e:
            print(f"Error in image processing: {e}")
            self.root.after(0, lambda: self.status_label.config(text=f"Error: {str(e)}"))
    
    def create_image_analysis_interface(self):
        #Create interface for image analysis
        # Clear the root window
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Main frame
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Top control panel
        control_panel = tk.Frame(main_frame, bg="#34495e", height=50)
        control_panel.pack(fill=tk.X)
        
        # If using logo
        if self.logo_photo:
            logo_label = tk.Label(control_panel, image=self.logo_photo, bg="#34495e")
            logo_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        title_label = tk.Label(
            control_panel, 
            text="EmotionLens - Image Analysis", 
            font=("Helvetica", 16, "bold"), 
            fg="#ecf0f1", 
            bg="#34495e"
        )
        title_label.pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        btn_frame = tk.Frame(control_panel, bg="#34495e")
        btn_frame.pack(side=tk.RIGHT, padx=10)
        
        # Save button
        save_btn = tk.Button(
            btn_frame,
            text="Save Result",
            command=self.save_analysis,
            font=("Helvetica", 12),
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            activeforeground="white",
            padx=10,
            borderwidth=0
        )
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # Return to menu button
        menu_btn = tk.Button(
            btn_frame,
            text="Back to Menu",
            command=self.create_start_screen,
            font=("Helvetica", 12),
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            activeforeground="white",
            padx=10,
            borderwidth=0
        )
        menu_btn.pack(side=tk.LEFT, padx=5)
        
        # Image display frame
        self.video_frame = tk.Label(main_frame, bg="black")
        self.video_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Status bar
        status_frame = tk.Frame(main_frame, bg="#34495e", height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(
            status_frame, 
            text="Analyzing image...", 
            font=("Helvetica", 10), 
            fg="#ecf0f1", 
            bg="#34495e",
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
    
    def get_smoothed_emotion(self):
        #Apply temporal smoothing to emotions
        if not self.emotion_history:
            return ("neutral", 0.0)
            
        # Count occurrences of each emotion
        emotion_counts = {}
        for emotion, confidence in self.emotion_history:
            if emotion not in emotion_counts:
                emotion_counts[emotion] = {"count": 0, "total_confidence": 0.0}
            emotion_counts[emotion]["count"] += 1
            emotion_counts[emotion]["total_confidence"] += confidence
        
        # Find the most frequent emotion
        max_count = 0
        smoothed_emotion = "neutral"
        avg_confidence = 0.0
        
        for emotion, data in emotion_counts.items():
            if data["count"] > max_count:
                max_count = data["count"]
                smoothed_emotion = emotion
                avg_confidence = data["total_confidence"] / data["count"]
        
        return (smoothed_emotion, avg_confidence)
    
    def update_display(self):
        #Update the video display with the current frame
        if self.frame_with_detection is None or not self.is_running:
            return
            
        try:
            # Convert OpenCV BGR to RGB for tkinter
            frame_rgb = cv2.cvtColor(self.frame_with_detection, cv2.COLOR_BGR2RGB)
            
            # Get current dimensions of the video frame widget
            try:
                video_frame_width = self.video_frame.winfo_width()
                video_frame_height = self.video_frame.winfo_height()
            except:
                # If widget dimensions can't be accessed, use default values
                video_frame_width = 640
                video_frame_height = 480
            
            if video_frame_width <= 1 or video_frame_height <= 1:
                # The window hasn't been drawn yet, use default dimensions
                video_frame_width = 640
                video_frame_height = 480
                
            # Get original frame dimensions
            h, w = frame_rgb.shape[:2]
                
            # Calculate scaling factor
            scale_w = video_frame_width / w
            scale_h = video_frame_height / h
            scale = min(scale_w, scale_h)
            
            # Calculate new dimensions
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create PIL image
            img = Image.fromarray(frame_resized)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # Update the video frame label if it still exists
            if self.is_running and hasattr(self, 'video_frame') and self.video_frame.winfo_exists():
                self.video_frame.config(image=img_tk)
                self.video_frame.image = img_tk  # Keep a reference to prevent garbage collection
        except Exception as e:
            print(f"Error updating display: {e}")
            # Don't try to update the status label as it might also be gone
            pass
    
    def take_screenshot(self):
        #Take a screenshot of the current frame with detections
        if self.frame_with_detection is None:
            messagebox.showinfo("Screenshot", "No image to capture")
            return
            
        # Create screenshots directory if it doesn't exist
        os.makedirs("screenshots", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        filename = f"screenshots/emotion_detection_{timestamp}.png"
        
        # Save the screenshot
        cv2.imwrite(filename, self.frame_with_detection)
        
        # Update status
        self.status_label.config(text=f"Screenshot saved: {filename}")
        
        # Show confirmation
        messagebox.showinfo("Screenshot", f"Screenshot saved as {filename}")
    
    def save_analysis(self):
        #Save the analyzed image
        if self.frame_with_detection is None:
            messagebox.showinfo("Save", "No image to save")
            return
            
        # Ask user for save location
        filetypes = [
            ("PNG Image", "*.png"),
            ("JPEG Image", "*.jpg"),
            ("All Files", "*.*")
        ]
        filename = filedialog.asksaveasfilename(
            title="Save Analysis Result",
            filetypes=filetypes,
            defaultextension=".png"
        )
        
        if not filename:
            return  # User cancelled
            
        # Save the image
        cv2.imwrite(filename, self.frame_with_detection)
        
        # Show confirmation
        messagebox.showinfo("Save", f"Analysis result saved as {filename}")
    
    def stop_and_return(self):
        #Stop video processing and return to start screen
        self.is_running = False
        if self.thread is not None:
            self.thread.join(1.0)  # Wait for the thread to finish
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Clear variables
        self.current_frame = None
        self.frame_with_detection = None
        self.emotion_history = []
        
        # Return to start screen
        self.create_start_screen()


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()
"""
        