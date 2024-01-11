# Python Packages
from pathlib import Path
import PIL

#External Package
import streamlit as st

#lOADING MODULES
import settings
import helper

st.set_page_config(
    page_title="Sign Language Gesture Recognition",
    page_icon="d",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Sign Language Gesture Recognition")

st.sidebar.header("Model Configuration")

# Model Options
model_type = st.sidebar.radio("Type", ['Detection'])

#Selecting Detecting
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)

#Load Pre-Trained Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path : {model_path}")
    st.error(ex)

st.sidebar.header("")
source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image", use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image", use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image', use_column_width=True)
        else:
            if st.sidebar.button('Start Detection'):
                res = model.predict(uploaded_image)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(model)
    
else:
    st.error("Please select a valid source type!")