import streamlit as st
import cv2
from yolo_predictions import Yolo_Pred
from PIL import Image
import numpy as np

st.set_page_config(page_title='object detection from images',layout='centered',page_icon='https://cdn.analyticsvidhya.com/wp-content/uploads/2018/12/yologo_2-850x451.png')
st.title('YOLO OBJECT DETECTION FROM IMAGES')
st.write('   please Upload images to get detections   ')
with st.spinner('model loading ....') :
                yolo=Yolo_Pred(onnx_model='./models/best.onnx',data_yaml='./models/data.yaml')
                st.balloons()

#upload image
image_file = st.file_uploader("Choose an image", type=["jpg", "png"])
if image_file is not None:
    prediction=False;
    # Read and display the image
    
    st.info('the image uploaded successfully')
    image = Image.open(image_file)
    st.image(image)
    button=st.button('GET PREDICTIONS')
    if button:
            image_array =np.array(image)
            img_pred=yolo.predictions(image_array)
            pred_img_obj=Image.fromarray(img_pred)
            prediction=True;
    if prediction:
        st.image(pred_img_obj)        
 
       
