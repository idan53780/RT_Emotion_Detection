import cv2
import cv2.data
from deepface import DeepFace

cascade_path = r'models\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise IOError(f"Failed to load Haar Cascade from: {cascade_path}")

logo = cv2.imread('src/logo_icon.png',cv2.IMREAD_UNCHANGED)

if logo is None:
    raise IOError("Failed to load logo image. Check the path.")

logo = cv2.resize(logo, (60, 60))

#Initialize webcam
cap = cv2.VideoCapture(0)

#check if the webcam is working
if not cap.isOpened():
    raise IOError("Webcam Error: cannot open webcam")

 #  Text configuration

org = (50,50)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.5
text_color = (255, 183 , 51)
thickness = 2

while True:
    ret,frame = cap.read()
    if not ret:
        print("Frame caputre failed")
        break
    #Convert for Deepface
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    result = DeepFace.analyze( 
                              rgb_frame,
                              actions = ['emotion'],
                              enforce_detection = False
                             )[0]

    #Convert to grayscale for face detection
    gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)

    #Drawing a rectangle around the face

    for( x, y, w, h ) in faces:
        cv2.rectangle( frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
   

    #Use putText() in order to show text over the video
    cv2.putText(
                frame,
                result['dominant_emotion'],
                org,
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA 
                )  
     
    # Overlay logo in bottom-left corner
    h_logo, w_logo = logo.shape[:2]
    h_frame, w_frame = frame.shape[:2]
    x_offset = 10
    y_offset = h_frame - h_logo - 10
    
    if logo.shape[2] == 4:  # Has alpha channel
        b, g, r, a = cv2.split(logo)
        overlay_color = cv2.merge((b, g, r))
        mask = cv2.merge((a, a, a))

        roi = frame[y_offset:y_offset + h_logo, x_offset:x_offset + w_logo]
        img1_bg = cv2.bitwise_and(roi, 255 - mask)
        img2_fg = cv2.bitwise_and(overlay_color, mask)
        dst = cv2.add(img1_bg, img2_fg)
        frame[y_offset:y_offset + h_logo, x_offset:x_offset + w_logo] = dst

    else:
        frame[y_offset:y_offset + h_logo, x_offset:x_offset + w_logo] = logo
    


    #Display the video feed
    cv2.imshow('EmotionLens',frame)
    
    #Press 'q' to quit
    if cv2.waitKey(2) & 0xff == ord('q'):
        break
# End session
cap.release()
cv2.destroyAllWindows()
                