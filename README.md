# AI-Mental-Health-Analyzer
Emotion Detection + Stress Prediction using CNN &amp; ML



 Overview
AI Mental Health Analyzer

Developed an end-to-end AI-based application that analyzes human emotions from facial images and predicts stress levels using lifestyle data. The system integrates computer vision and machine learning to provide real-time insights into a user’s mental well-being.

The project uses a Convolutional Neural Network (CNN) trained on facial emotion datasets to classify emotions such as angry, happy, sad, and neutral. It further combines these predictions with user-provided lifestyle inputs (sleep hours, mood score, work hours, screen time, and social interaction) to estimate stress levels using a machine learning model.

An interactive web interface was built using Streamlit, allowing users to upload or capture images via webcam, input daily habits, and instantly receive emotion detection results along with stress predictions and personalized recommendations.

This project demonstrates practical implementation of deep learning, data preprocessing, feature engineering, and real-time deployment of AI models into a user-friendly application.

 Features
- Emotion Detection (CNN)
- Real-time Image Capture (Webcam)
- Stress Prediction (Machine Learning)
- Interactive UI (Streamlit)

Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- Scikit-learn

Run Locally
pip install -r requirements.txt
streamlit run app.py

Output
Emotion detected from face
Stress level (High / Low)
Lifestyle-based recommendations


 

