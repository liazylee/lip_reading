# Import all of the dependencies
import tempfile
import time
import cv2
import streamlit as st
import os
import imageio 
import numpy as np
import tensorflow as tf
# from tensorflow_datasets.core.features.image_feature import cv2

from utils import load_data, num_to_char, load_alignments, load_alignments_text, load_video, cosine_similarity
from modelutil import load_model


video_path = "app/test_video.mp4"
# Temporary file to store the recorded video

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    # st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App') 
# Generating a list of options or videos
# change the path to absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
options = os.listdir(os.path.join('..','data','s1'))

selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2, col3= st.columns(3)

process_button = st.button("Process Video")
record_button = col3.button("Record Video")


if process_button:
    # Rendering the video 
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..','data','s1', selected_video)
        selected_video_name = selected_video.split('.')[0]
        align = os.path.join('..', 'data', 'alignments', 's1', f'{selected_video_name}.align')
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y') # convert to mp4
        st.info('loading the video...')
        time.sleep(1)
        # Rendering inside of the app
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read() 
        st.video(video_bytes)
        if not os.path.exists(align):
            st.error(f'No alignment file found{align}')
        else:
            align =' '.join(load_alignments_text(align))
            st.info(f'alignments: {align}')

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        # wait for the video to be loaded

        video = load_video(file_path)  #  load video features
        # imageio.mimsave('animation.gif', video, fps=10)
        # st.image('animation.gif', width=400)
        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.info(converted_prediction)
        # accuracy use 余弦值相似度计算
        accuracy = cosine_similarity(converted_prediction, align)
        st.info(f'accuracy : {round(accuracy,5)*100} %')




if record_button:
    with col3:
        cap=cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))
        st.info('Recording the video...')
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
               # recognize the mouth and crop the mouth in [46,140] size
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
                frame = cv2.flip(frame, 1)  # flip the frame horizontally
                out.write(frame)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        # wait for the video to be loaded

        video = load_video('output.mp4')

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        padded_video = tf.pad(video, [[0, 75 - tf.shape(video)[0]], [0, 0], [0, 0], [0, 0]])
        yhat = model.predict(tf.expand_dims(padded_video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)
        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)




