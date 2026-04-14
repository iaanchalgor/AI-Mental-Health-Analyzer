import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import joblib

# =========================
# LOAD MODELS
# =========================
cnn_model = load_model("emotion_classifier_final.keras")   # your 

# If you saved stress model, uncomment below:
# stress_model = joblib.load("stress_model.pkl")
# le = joblib.load("label_encoder.pkl")

# Emotion labels (CHANGE if yours are different)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 48, 48, 1)
    return reshaped

# =========================
# FACE DETECTION
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# =========================
# UI
# =========================
st.title("🧠 AI Mental Health Analyzer")
st.caption("Emotion Detection + Stress Prediction System")

st.markdown("---")

# =========================
# IMAGE INPUT OPTIONS
# =========================
option = st.radio("Choose Input Method:", ["📷 Camera", "📁 Upload Image"])

img = None

# 📷 Camera
if option == "📷 Camera":
    img_file = st.camera_input("Take a picture")
    if img_file is not None:
        image = Image.open(img_file)
        img = np.array(image)

# 📁 Upload
else:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = np.array(image)

# =========================
# PROCESS IMAGE
# =========================
if img is not None:

    st.image(img, caption="Input Image", use_container_width=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("⚠️ No face detected. Try again.")
    else:
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]

            processed = preprocess_image(face)

            pred = cnn_model.predict(processed)
            emotion_index = np.argmax(pred)
            emotion = emotion_labels[emotion_index]
            confidence = np.max(pred) * 100

            # Draw rectangle
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            st.image(img, caption="Detected Face", use_container_width=True)

            st.success(f"😊 Emotion: {emotion}")
            st.write(f"Confidence: {confidence:.2f}%")

# =========================
# USER INPUTS FOR STRESS
# =========================
st.markdown("---")
st.subheader("📊 Enter Your Daily Details")

sleep = st.slider("Sleep Hours", 0, 12, 6)
mood = st.slider("Mood (1-10)", 1, 10, 5)
work = st.slider("Work Hours", 0, 12, 6)
screen = st.slider("Screen Time", 0, 12, 5)
social = st.slider("Social Interaction (1-10)", 1, 10, 5)

# =========================
# STRESS PREDICTION
# =========================
if st.button("🔍 Predict Stress"):

    st.markdown("## 🔥 FINAL RESULT")

    # Dummy logic if model not saved
    if sleep < 5 or screen > 7 or mood < 4:
        stress = 1
    else:
        stress = 0

    # If using trained model, use this:
    # input_data = np.array([[sleep, mood, work, screen, social]])
    # stress = stress_model.predict(input_data)[0]

    if stress == 1:
        st.error("⚠️ HIGH STRESS")
        st.write("💡 Advice: Take rest, reduce screen time.")
    else:
        st.success("✅ LOW STRESS")
        st.write("💡 Advice: Keep maintaining your routine.")