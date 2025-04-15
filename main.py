import cv2
import streamlit as st
import numpy as np
from PIL import Image
from simple_facerec import SimpleFacerec

# Load known faces
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

st.title("🔍 Real-time Face Recognition")
st.write("Upload an image or use your webcam to detect known faces.")

# Webcam or image upload option
option = st.radio("Choose input method:", ["📸 Webcam", "🖼️ Upload Image"])

def process_image(img):
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    face_locations, face_names = sfr.detect_known_faces(frame)

    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

if option == "📸 Webcam":
    picture = st.camera_input("Take a photo")
    if picture:
        image = Image.open(picture)
        image_np = np.array(image)
        result = process_image(image_np)
        st.image(result, caption="Detected Faces", use_container_width=True)

elif option == "🖼️ Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        result = process_image(image_np)
        st.image(result, caption="Detected Faces", use_container_width=True)
