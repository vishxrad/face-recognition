import cv2
import streamlit as st
import numpy as np
from PIL import Image
from simple_facerec import SimpleFacerec
from io import BytesIO
import os

# Set page config
st.set_page_config(
    page_title="Face Recognition App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #424242;
        margin-bottom: 2rem;
        text-align: center;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .success-msg {
        padding: 1rem;
        border-radius: 5px;
        background-color: #81C784;
        color: white;
        margin-bottom: 1rem;
    }
    .error-msg {
        padding: 1rem;
        border-radius: 5px;
        background-color: #e57373;
        color: white;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1.5rem;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .result-container {
        padding: 1.5rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        font-size: 0.8rem;
        color: #9e9e9e;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for app info and instructions
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/face-id.png", width=80)
    st.markdown("## About")
    st.info("This app uses face recognition to identify people from a database of known faces.")
    
    st.markdown("## How to use")
    st.markdown("1. Choose input method (webcam or upload)")
    st.markdown("2. Take a photo or upload an image")
    st.markdown("3. View detection results")
    
    st.markdown("## Settings")
    confidence_threshold = st.slider("Recognition confidence", 0.0, 1.0, 0.5, 0.05)
    
    st.markdown("---")
    st.markdown('<div class="footer">Facial Recognition App v1.0</div>', unsafe_allow_html=True)

# Main content
st.markdown('<h1 class="main-header">Facial Recognition System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Identify faces in real-time using your webcam or uploaded images</p>', unsafe_allow_html=True)

# Check if images directory exists
images_path = "images/"
if not os.path.exists(images_path):
    st.markdown('<div class="error-msg">⚠️ Images directory not found! Please create a directory named "images" and add reference face images.</div>', unsafe_allow_html=True)
    st.stop()

# Use session state to prevent reloading encodings on every interaction
try:
    if 'face_recognizer' not in st.session_state:
        st.session_state['face_recognizer'] = SimpleFacerec()
        with st.spinner('🔄 Loading face database...'):
            st.session_state['face_recognizer'].load_encoding_images(images_path)
        st.markdown('<div class="success-msg">✅ Face database loaded successfully!</div>', unsafe_allow_html=True)

    # Use the stored face recognizer
    sfr = st.session_state['face_recognizer']
except Exception as e:
    st.markdown(f'<div class="error-msg">❌ Error loading face database: {str(e)}</div>', unsafe_allow_html=True)
    st.stop()

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    
    # Webcam or image upload option
    option = st.radio("Choose input method:", ["📸 Webcam", "🖼️ Upload Image"])
    st.markdown('</div>', unsafe_allow_html=True)

def process_image(img):
    try:
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        face_locations, face_names = sfr.detect_known_faces(frame)
        
        # Count faces for summary
        recognized_faces = len([name for name in face_names if name != "Unknown"])
        unknown_faces = len([name for name in face_names if name == "Unknown"])
        
        # Draw rectangles and labels
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            
            # Different colors for known vs unknown
            if name == "Unknown":
                color = (200, 0, 0)  # Red for unknown
            else:
                color = (0, 128, 0)  # Green for known
                
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), recognized_faces, unknown_faces
    except Exception as e:
        st.markdown(f'<div class="error-msg">❌ Error processing image: {str(e)}</div>', unsafe_allow_html=True)
        return None, 0, 0

with col1:
    if option == "📸 Webcam":
        picture = st.camera_input("Take a photo", key="camera")
        if picture:
            image = Image.open(picture)
            image_np = np.array(image)
            with st.spinner("Processing image..."):
                result, recognized, unknown = process_image(image_np)
    
    elif option == "🖼️ Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            with st.spinner("Processing image..."):
                result, recognized, unknown = process_image(image_np)

# Display results
with col2:
    if 'result' in locals() and result is not None:
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.image(result, caption="Processed Image", use_container_width =True)
        
        # Detection stats
        st.markdown("### Detection Results")
        st.markdown(f"✅ **Recognized faces:** {recognized}")
        st.markdown(f"❓ **Unknown faces:** {unknown}")
        st.markdown(f"🔍 **Total faces detected:** {recognized + unknown}")
        
        # Download button for the processed image
        if result is not None:
            result_pil = Image.fromarray(result)
            buf = BytesIO()
            result_pil.save(buf, format="JPEG")
            st.download_button(
                label="Download Processed Image",
                data=buf.getvalue(),
                file_name="face_recognition_result.jpg",
                mime="image/jpeg"
            )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("### Results will appear here")
        st.markdown("Take a photo or upload an image to see face recognition results.")
        st.markdown('</div>', unsafe_allow_html=True)
