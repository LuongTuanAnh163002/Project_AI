import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
IMAGE_DIMS = (96, 96, 3)
def predict_image(image, model, mlb):
    #image = cv2.imread(image)
    (w, h, c) = image.shape
    height_rz = int(h*400/w)
    output = cv2.resize(image, (height_rz, 400))
    image = cv2.resize(image, IMAGE_DIMS[:2])/255.0
    prob = model.predict(np.expand_dims(image, axis=0))[0]
    argmax = np.argsort(prob)[::-1][:2]
    for (i, j) in enumerate(argmax):
        label = "{}: {:.2f}%".format(mlb.classes_[j], prob[j] * 100)
        cv2.putText(output, label, (5, (i * 20) + 15), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 0, 0), 2)
#   plt.figure(figsize=(8, 16))
#   plt.axis('Off')
#   plt.imshow(output)
    return output

with open('mlb.pkl', 'rb') as f:
    sub_model = pickle.load(f)

main_model = tf.keras.models.load_model("model_fashion_multitask_learning.h5")
file_img = st.file_uploader("Choose file")
if file_img is not None:
    file_bytes = np.asarray(bytearray(file_img.read()), dtype=np.uint8)
    imga = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(imga, cv2.COLOR_BGR2RGB)
    img = predict_image(img_rgb, main_model, sub_model)
    st.image(img, caption='Image prediction', use_column_width=True)
else:
    st.write("No file up load")
