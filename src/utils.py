import os
import cv2
from tkinter import messagebox, filedialog
from datetime import datetime

filetypes = [
    ("PNG Image", "*.png"),
    ("JPEG Image", "*.jpg"),
    ("All Files", "*.*")
    ]




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
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    default_filename = f"emotion_detection_{timestamp}.png"
    

    filename = filedialog.asksaveasfilename(
        title="Save Screenshot",
        filetypes=filetypes,
        defaultextension=".png",
        initialfile=default_filename,  # Suggest timestamp-based filename
        initialdir=os.path.join(os.getcwd(), "screenshots")  # Default to screenshots folder
    )
    
    if not filename:
        return
    try:
        success = cv2.imwrite(filename, app.frame_with_detection)
        if not success:
            messagebox.showerror("Screenshot Error", f"Failed to save screenshot to {filename}")
            return
        app.status_label.config(text=f"Screenshot saved: {filename}")
        messagebox.showinfo("Screenshot", f"Screenshot saved as {filename}")
    except Exception as e:
        messagebox.showerror("Screenshot Error", f"Error saving screenshot: {str(e)}")

def save_analysis(app):
    if app.frame_with_detection is None:
        messagebox.showinfo("Save", "No image to save")
        return
        
    ''' filetypes = [
    ("PNG Image", "*.png"),
    ("JPEG Image", "*.jpg"),
    ("All Files", "*.*")
    ]
    '''

    filename = filedialog.asksaveasfilename(
    title="Save Analysis Result",
    filetypes=filetypes,
    defaultextension=".png"
    )
    
    if not filename:
        return
            
        
    cv2.imwrite(filename, app.frame_with_detection)
    messagebox.showinfo("Save", f"Analysis result saved as {filename}")
