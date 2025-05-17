import cv2
import numpy as np
from deepface import DeepFace
import threading

def process_video(app):
    try:
        while app.is_running and app.cap is not None:
            ret, frame = app.cap.read()
            if not ret:
                if app.input_source == "webcam":
                    app.cap.release()
                    app.cap = cv2.VideoCapture(0)
                    if not app.cap.isOpened():
                        print("Lost connection to webcam")
                        app.root.after(0, app.stop_and_return)
                        break
                else:
                    print("End of video file")
                    app.root.after(0, app.stop_and_return)
                    break
                
            app.current_frame = frame.copy()
            process_frame(app)
            
            cv2.waitKey(10)
                
    except Exception as e:
        print(f"Error in video processing: {e}")
        if hasattr(app, 'status_label') and app.status_label.winfo_exists():
            app.root.after(0, lambda: app.status_label.config(text=f"Error: {str(e)}"))
    finally:
        if app.cap is not None:
            app.cap.release()

def process_frame(app):
    if app.current_frame is None:
        return
    
    frame = app.current_frame.copy()
    app.frame_with_detection = frame.copy()
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = []
    
    if app.current_detector == "Haar Cascade":
        faces_rect = app.detection_models["Haar Cascade"].detectMultiScale(gray, 1.1, 4)
        faces = [(x, y, x+w, y+h) for (x, y, w, h) in faces_rect]
    elif app.current_detector == "DNN" and app.detection_models["DNN"] is not None:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        app.detection_models["DNN"].setInput(blob)
        detections = app.detection_models["DNN"].forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                faces.append((x1, y1, x2, y2))
    
    for (x1, y1, x2, y2) in faces:
        try:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            cv2.rectangle(app.frame_with_detection, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            face_roi = rgb_frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue
            
            result = DeepFace.analyze(
                face_roi,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='skip',
                silent=True
            )[0]
            
            emotion = result['dominant_emotion']
            emotions = result['emotion']
            confidence = emotions[emotion] / 100.0
            
            app.emotion_history.append((emotion, confidence))
            if len(app.emotion_history) > app.history_length:
                app.emotion_history.pop(0)
            
            smoothed_emotion = app.get_smoothed_emotion()
            
            if smoothed_emotion[1] >= app.confidence_threshold:
                emotion_text = f"{smoothed_emotion[0]} ({int(smoothed_emotion[1]*100)}%)"
                cv2.putText(
                    app.frame_with_detection,
                    emotion_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
                
                meter_width = x2 - x1
                filled_width = int(meter_width * smoothed_emotion[1])
                cv2.rectangle(
                    app.frame_with_detection,
                    (x1, y2 + 5),
                    (x1 + meter_width, y2 + 15),
                    (0, 0, 255),
                    1
                )
                cv2.rectangle(
                    app.frame_with_detection,
                    (x1, y2 + 5),
                    (x1 + filled_width, y2 + 15),
                    (0, 255, 0),
                    cv2.FILLED
                )
        except Exception as e:
            print(f"Error processing face: {e}")
            continue
    
    if app.logo_photo:
        h_frame, w_frame = app.frame_with_detection.shape[:2]
        logo_np = cv2.cvtColor(np.array(app.logo_img), cv2.COLOR_RGBA2BGRA)
        h_logo, w_logo = logo_np.shape[:2]
        x_offset = 10
        y_offset = h_frame - h_logo - 10
        
        if logo_np.shape[2] == 4:
            alpha_logo = logo_np[:, :, 3] / 255.0
            alpha_frame = 1.0 - alpha_logo
            
            for c in range(0, 3):
                app.frame_with_detection[y_offset:y_offset+h_logo, x_offset:x_offset+w_logo, c] = (
                    alpha_logo * logo_np[:, :, c] + 
                    alpha_frame * app.frame_with_detection[y_offset:y_offset+h_logo, x_offset:x_offset+w_logo, c]
                )
    
    app.update_display()

def process_image(app):
    if app.current_frame is None:
        return
    
    from .ui import create_image_analysis_interface
    create_image_analysis_interface(app)
    
    threading.Thread(target=lambda: _process_image_thread(app)).start()

def _process_image_thread(app):
    try:
        process_frame(app)
        app.root.after(0, lambda: app.status_label.config(text="Analysis complete"))
    except Exception as e:
        app.root.after(0, lambda: app.status_label.config(text=f"Error: {str(e)}"))