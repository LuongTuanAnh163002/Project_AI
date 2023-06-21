import cv2
import numpy as np
from tensorflow.keras.models import load_model

model_landmark = load_model("best_model_landmark_detect.h5")
video = cv2.VideoCapture(0)
detect_face = cv2.CascadeClassifier(r"cascades_haarcascade_frontalface_default.xml")


while True:
    ret, img = video.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_face.detectMultiScale(gray, 1.25, 6)
    for (x, y, w, h) in faces:
        gray_face = gray[y:y+h, x:x+w]
        color_face = img[y:y+h, x:x+w]
        gray_normalized = gray_face / 255
        original_shape = gray_face.shape
        face_resized_gray = cv2.resize(gray_normalized, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized_gray = face_resized_gray.reshape(1, 96, 96, 1)

        keypoints = model_landmark.predict(face_resized_gray)
        keypoints = keypoints * 48 + 48
        face_resized_color = cv2.resize(color_face, (96, 96), interpolation = cv2.INTER_AREA)
        
        points = []
        for i, co in enumerate(keypoints[0][0::2]):
            points.append((co, keypoints[0][1::2][i]))

        for keypoint in points:
            cv2.circle(face_resized_color, tuple(map(int, keypoint)), 1, (0,255,0), 1)
        
        img[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation = cv2.INTER_CUBIC)

        cv2.imshow("Face LandMark Detection", img)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()