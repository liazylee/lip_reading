# Import all of the dependencies
import tempfile
import time

import streamlit as st
import os
import imageio 

import tensorflow as tf
from tensorflow_datasets.core.features.image_feature import cv2

from utils import load_data, num_to_char
from modelutil import load_model


video_path = "app/test_video.mp4"
# Temporary file to store the recorded video
out = None

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
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
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y') # convert to mp4

        time.sleep(5)
        # Rendering inside of the app
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read() 
        st.video(video_bytes)

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        # wait for the video to be loaded

        video, annotations = load_data(tf.convert_to_tensor(file_path))
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
        st.text(converted_prediction)
        

# if record_button:
#     with col3:
#         st.info("Recording... Click 'Record' again to stop.")
#
#         # OpenCV video capture
#         cap = cv2.VideoCapture(0)  # Use 0 for the default camera
#
#         # Define the codec and create a VideoWriter object
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
#
#         while record_button:
#             # Capture frame-by-frame
#             ret, frame = cap.read()
#
#             # Write the frame to the video file
#             out.write(frame)
#
#         # Release the resources
#         cap.release()
#         out.release()
#         st.success("Recording complete!")
#         st.video(video_path)
#
# # Display the recorded video
#
#
#
#         with col2:
#             st.info('This is all the machine learning model sees when making a prediction')
#             # wait for the video to be loaded
#
#             video, annotations = load_data(tf.convert_to_tensor(video_path))
#             # imageio.mimsave('animation.gif', video, fps=10)
#             # st.image('animation.gif', width=400)
#
#             st.info('This is the output of the machine learning model as tokens')
#             model = load_model()
#             yhat = model.predict(tf.expand_dims(video, axis=0))
#             decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
#             st.text(decoder)
#
#             # Convert prediction to text
#             st.info('Decode the raw tokens into words')
#             converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
#             st.text(converted_prediction)




