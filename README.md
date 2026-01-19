# Facial Emotion Recognition Web App

A simple and interactive web application that detects facial emotions from uploaded images or sample photos using **DeepFace** and **OpenCV**. Built with **Streamlit** for an easy-to-use interface.
## Demo

Live app: (https://emotion-recogn.streamlit.app/)


### App Screenshot
!(demo-screenshot.png)

## Features
- Upload your own photo (JPG/PNG) or select from pre-loaded sample images
- Real-time facial emotion detection (happy, sad, angry, neutral, surprise, etc.)
- Draws bounding boxes around detected faces with dominant emotion and confidence score
- Displays full emotion probability distribution for transparency
- Uses **DeepFace** with **OpenCV** backend for fast and lightweight inference
- Clean two-column layout for side-by-side original vs. annotated images

## Demo
Live app: [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)  
*(Update this link after deploying on Streamlit Community Cloud)*

## Tech Stack
- **Frontend/UI**: Streamlit
- **Core Model**: DeepFace (emotion analysis)
- **Face Detection**: OpenCV backend
- **Image Processing**: OpenCV + Pillow
- **Deployment**: Streamlit Community Cloud

## Installation & Local Run

1. Clone the repository:
   ```bash
   git clone https://github.com/apurvapillai24/fer.git
   cd fer
