📌 Overview

This project implements a real-time facial emotion recognition system. It detects faces from live video or a webcam feed and classifies emotions into eight categories:

Surprise, Happiness, Anger, Disgust, Contempt, Sadness, Fear, and Neutral

Face Detection: OpenCV Haar Cascade (fast + lightweight)

Emotion Recognition: FER library (CNN trained on FER2013 dataset)

Contempt: Custom rule-based heuristic (mouth asymmetry)

Neutral: Fallback when no other emotion is dominant

Both video file/YouTube input and live webcam mode are supported. Results are logged with timestamps and exported to CSV.
