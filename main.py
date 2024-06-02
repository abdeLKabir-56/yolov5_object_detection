import streamlit as st
from yolo_predictions import Yolo_Pred
from PIL import Image
import numpy as np

# Set page config at the start
st.set_page_config(
    page_title='YOLO V5 OBJECT DETECTION APPLICATION',
    layout='centered',
    page_icon='https://cdn.analyticsvidhya.com/wp-content/uploads/2018/12/yologo_2-850x451.png'
)

def home():
    st.title('YOLO V5 OBJECT DETECTION APPLICATION')
    st.caption("Our project demonstrates ***object detection*** application using the ***YOLO***(You Only Look Once) model version 5 and the ***Streamlit*** framework, combined with computer vision techniques using ***OpenCV***. Object detection plays a crucial role in many computer vision applications, allowing us to identify and locate multiple objects within an image or a video stream. The YOLO model version 5 is a state-of-the-art deep learning algorithm known for its high accuracy and real-time performance. With Streamlit, we create an interactive and user-friendly web application that enables users to upload images or provide a video stream for real-time object detection. The Streamlit framework simplifies the development process by providing intuitive widgets and UI components. To process the uploaded images or video stream, we leverage OpenCV, a powerful library for computer vision tasks. OpenCV allows us to perform various image processing operations, such as resizing, color conversions, and pre-processing required for object detection. We integrate the YOLO model version 5 into our application to detect objects accurately. Our application provides a seamless user experience, allowing users to visualize the detected objects, their bounding boxes, and class labels. Additionally, we can leverage Streamlit's capabilities to incorporate additional functionalities, such as filtering detected objects based on confidence scores or displaying statistical insights about the detected objects. By combining the YOLO model version 5, Streamlit, and OpenCV, our project empowers users to perform efficient and accurate object detection tasks in real-time. This application finds applications in various domains, including surveillance, autonomous vehicles, and object recognition systems.")
    st.markdown("""
    ### This app detects objects from images and videos 
    - Our model detects 20 objects from images and videos:
        - person        
        - car
        - chair
        - bottle
        - pottedplant
        - bird
        - dog
        - sofa
        - bicycle
        - horse
        - boat
        - motorbike
        - cat
        - tvmonitor
        - cow
        - sheep
        - aeroplane
        - train
        - diningtable
        - bus

    - [Click here for image object detection](?page=yolo_image)
    """)

def yolo_image():
    st.title('YOLO OBJECT DETECTION FROM IMAGES')
    st.write('Please upload images to get detections')
    
    with st.spinner('Model loading ....'):
        yolo = Yolo_Pred(onnx_model='./models/best.onnx', data_yaml='./models/data.yaml')
        st.balloons()

    # Upload image
    image_file = st.file_uploader("Choose an image", type=["jpg", "png"])
    if image_file is not None:
        prediction = False
        # Read and display the image
        st.info('The image uploaded successfully')
        image = Image.open(image_file)
        st.image(image)
        button = st.button('GET PREDICTIONS')
        if button:
            image_array = np.array(image)
            img_pred = yolo.predictions(image_array)
            pred_img_obj = Image.fromarray(img_pred)
            prediction = True
        if prediction:
            st.image(pred_img_obj)

def main():
    query_params = st.experimental_get_query_params()
    page = query_params.get("page", ["home"])[0]

    if page == "home":
        home()
    elif page == "yolo_image":
        yolo_image()

if __name__ == "__main__":
    main()
