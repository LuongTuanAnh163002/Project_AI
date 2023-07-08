import cv2
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import os
import vlc

os.add_dll_directory(r'C:\Program Files\VideoLAN\VLC')
file_path = 'Video_test.mp4'
vlc_instance = vlc.Instance()
player = vlc_instance.media_player_new()
media = vlc_instance.media_new(file_path)
player.set_media(media)

video = cv2.VideoCapture(0)
detector = HandDetector(detectionCon = 0.8, maxHands = 1)
dic1 = {'0': '0','1': '1','2': '2','3': '3','4': '4','5': '5','6': '6','7': '7','8': '8','9': '9'}

model = load_model("best_model_sign_language_other3.h5")
while True:
    ret, img = video.read()
    img_copy = img.copy()
    hands, img1 = detector.findHands(img, flipType = True)
    hand_img = 0
    hand_img_shape = (0, 0, 0)
    labels = -1
    if hands:
        hand1 = hands[0]
        bbox1 = hand1["bbox"]
        cv2.rectangle(img_copy, (bbox1[0], bbox1[1]), (bbox1[0]+bbox1[2], bbox1[1]+bbox1[3]), (0, 255, 0), 2)
        x1 = bbox1[0]
        x2 = bbox1[0]+bbox1[2]
        y1 = bbox1[1]
        y2 = bbox1[1]+bbox1[3]
        if (x1, x2, y1, y2) != (0, 0, 0, 0):
            hand_img = img_copy[(y1-10):(y2+10), (x1-10):(x2+10)]
            hand_img_shape = hand_img.shape
    if hand_img_shape == (0, 0, 0):
        print("No result return")
    else:
        if hand_img.shape[0] >= 64 and hand_img.shape[1] >= 64:
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            hand_img = hand_img / 255.0
            hand_img = cv2.resize(hand_img, (64, 64))
            hand_img = hand_img.reshape(-1, 64, 64, 1)
            if hand_img.shape == (1, 64, 64, 1):
                preds = model.predict(hand_img)
                answer = np.argmax(preds, axis=1)
                labels = dic1[str(answer[0])]
                cv2.putText(img1, labels, (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
    if labels == "0":
        player.stop()
    elif labels == "1":
        player.play()
    elif labels == "8":
        player.pause()
        print("video is pause")
    elif labels == "9":
        player.set_pause(0)
        print("Continue video")
    cv2.imshow("Face detection", img1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()