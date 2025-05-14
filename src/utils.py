import os
import cv2
from tkinter import messagebox, filedialog
from datetime import datetime

def get_smoothed_emotion(app):
    if not app.emotion_history:
        return ("netural", 0.0)
    
    emotion_counts = {}
    for emotion, confidence in app.emotion_history:
        if emotion not in emotion_counts:
            emotion_counts[emotion] = {"count": 0, "total_confidence": 0.0}
        emotion_counts[emotion]["count"] += 1
        emotion_counts[emotion]["total_confidence"] += confidence

    max_count = 0       
    smoothed_emotion = "netural"
    avg_confidence = 0.0

    for emotion, data in emotion_counts.items():
        if data["count"] > max_count:
            max_count = data["count"]
            smoothed_emotion = emotion
            avg_confidence = data["total_confidence"] / data["count"]

    return (smoothed_emotion , avg_confidence)        

def take_screenshot(app):
    if app.frame_with_detection is None:
        messagebox.showinfo("Screenshot" , "No image to capture")
        return
    
    os.makedirs("screenshots", exist_ok=True)
    timestamp = datetime.now().strftime("%D_%m_%Y__%H:%M:%S")
    filename = f"screenshots/emotion_detection_{timestamp}.png"

    cv2.imwrite(filename, app.frame_with_detection)

    app.status_label.config(text=f"Screenshot saved: {filename}")
    messagebox.showinfo("Screenshot", f"Screenshot saved as {filename}")

def save_analysis(app):
    if app.frame_with_detection is None:
        messagebox.showinfo("Save", "No image to save")
        return
        
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
        return
            
        
    cv2.imwrite(filename, app.frame_with_detection)
    messagebox.showinfo("Save", f"Analysis result saved as {filename}")
