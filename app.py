import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import io
import requests  # for loading sample URLs

# DeepFace uses its own models; specify backend for face detection if needed
# 'opencv' is fast and no TF dependency for detection; emotion still uses lightweight DL

SAMPLE_IMAGES = {
    "Happy": "sample/happy.jpg",
    "Sad": "sample/sad.jpg",
    "Angry": "sample/angry.jpg",
    "Neutral": "sample/neutral.jpg",
    "Surprise": "sample/surprise.jpg",
}

def process_image(image):
    """Use DeepFace to analyze emotion → return annotated image + results"""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    try:
        # Analyze: returns list of dicts for each face
        results = DeepFace.analyze(img_cv, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        
        for res in results:
            region = res['region']  # x, y, w, h
            dominant = res['dominant_emotion']
            score = res['emotion'][dominant]
            
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            # Draw box
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Label
            label = f"{dominant.upper()}: {score:.1f}%"
            cv2.putText(img_cv, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        img_annotated = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        return img_annotated, results
    
    except Exception as e:
        return image, []  # fallback to original if error

# ───────────────────────────────────────────────
# Streamlit UI (same as before)
# ───────────────────────────────────────────────
st.title("Facial Emotion Recognition — Images Only (DeepFace)")
st.markdown("""
Upload a photo or pick a sample → see detected emotions with boxes.
Powered by DeepFace (no forced TensorFlow dependency in basic mode).
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    option = st.radio("Source", ["Upload your image", "Try sample images"])
    
    input_image = None
    
    if option == "Upload your image":
        uploaded = st.file_uploader("Upload photo (jpg/png)", type=["jpg", "jpeg", "png"])
        if uploaded:
            input_image = Image.open(uploaded)
            st.image(input_image, caption="Uploaded", use_column_width=True)
    
    else:
        sample_choice = st.selectbox("Sample", list(SAMPLE_IMAGES.keys()))
        if sample_choice:
            file_path = SAMPLE_IMAGES[sample_choice]
            try:
                input_image = Image.open(file_path)
                st.image(input_image, caption=f"Sample: {sample_choice}", use_column_width=True)
            except FileNotFoundError:
                st.error(f"Sample image not found: {file_path}")
with col2:
    st.subheader("Analyzed Result")
    
    if input_image:
        with st.spinner("Analyzing with DeepFace..."):
            annotated, detections = process_image(input_image)
            
            st.image(annotated, caption="With emotions", use_column_width=True)
            
            if detections:
                st.success(f"{len(detections)} face(s) detected!")
                for i, res in enumerate(detections, 1):
                    dominant = res['dominant_emotion']
                    score = res['emotion'][dominant]
                    st.write(f"**Face {i}** → **{dominant.capitalize()}** ({score:.1f}%)")
                    st.json(res['emotion'])  # full scores
            else:
                st.warning("No faces found or analysis skipped.")
    else:
        st.info("Choose or upload an image.")