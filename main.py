import streamlit as st

st.set_page_config(page_title="YOLO Object Detection Application", layout="centered")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Image Object Detection"])

if page == "Home":
    import home
    home.show()
elif page == "Image Object Detection":
    import pages.1_yolo_image as yolo_image
    yolo_image.show()
