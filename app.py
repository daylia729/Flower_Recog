import os
import tensorflow as tf
import keras
import numpy as np
from keras.models import load_model
import streamlit as st


st.header('Flower Classification CNN Model')

flower_names = ['daisy','dandelion','rose','sunflower','tulib']
model = load_model("D:\Flower classification\Flower_Recog_Model.h5")

def classify_images(image_path):
    input_image=tf.keras.utils.load_img(image_path,target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The image belongs to ' + flower_names[np.argmax(result)] + ' with a score of ' + str(100 * np.max(result.numpy()))
    return outcome

uploaded_file = st.file_uploader('Upload an Image')
save_path = None
if uploaded_file is not None:
    if not os.path.exists('upload'):
        os.makedirs('upload')
    save_path = os.path.join('upload', uploaded_file.name)
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file, width = 200)
    out = classify_images(save_path)
    st.markdown(out)
