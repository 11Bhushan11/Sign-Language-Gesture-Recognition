from ultralytics import YOLO
import streamlit as st
import cv2
import settings
import PIL
import tempfile


def load_model(model_path):
    model = YOLO(model_path)
    return model

def _display_detected_frames(model, st_frame, image):

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    res = model.predict(image)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image( res_plotted, caption='Detected Video', channels="BGR", use_column_width=True )


def play_webcam(model):

    source_webcam = settings.WEBCAM_PATH

    if st.sidebar.button('Start Detecting'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(model, st_frame, image)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(model):

    uploaded_video = st.sidebar.file_uploader("Upload a video...", type=["mp4"])
    
    if uploaded_video is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(uploaded_video.read())
                video_path = temp_video.name

                st.video(video_path)

                if st.sidebar.button('Detect Video'):
                    vid_cap = cv2.VideoCapture(video_path)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(model, st_frame, image)
                        else:
                            vid_cap.release()
                            break
        except Exception as e:
            st.sidebar.error("Error Loading Video" + str(e))