import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime
import streamlit as st
from PIL import Image

# Load the fine-tuned model
@st.cache_resource
def load_smile_detection_model():
    return load_model('smile_detection_mobilenetv2.h5')

model = load_smile_detection_model()

# Directory to save the images
save_dir = r"E:\face\pics"
os.makedirs(save_dir, exist_ok=True)

# Global variables
cap = None
is_running = False
smile_detected = False

def detect_smile(frame):
    global smile_detected
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_rgb = rgb[y:y+h, x:x+w]
        roi_rgb_resized = cv2.resize(roi_rgb, (224, 224))
        roi_rgb_resized = np.expand_dims(roi_rgb_resized, axis=0) / 255.0
        prediction = model.predict(roi_rgb_resized)

        if prediction[0][0] > 0.5 and not smile_detected:
            label = "Smiling"
            color = (0, 255, 0)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
            image_name = f'smile_detected_{timestamp}.png'
            image_path = os.path.join(save_dir, image_name)
            cv2.imwrite(image_path, frame)
            st.success(f"Smile detected and picture saved as {image_name}")
            smile_detected = True
            st.write(f"Prediction: {prediction[0][0]:.4f}")
        else:
            label = "Not Smiling"
            color = (0, 0, 255)
            st.info("Waiting for a smile...")
            st.write(f"Prediction: {prediction[0][0]:.4f}")

        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame, smile_detected

def main():
    global cap, is_running, smile_detected

    st.title("Smile Detection")

    start_button = st.button("Start Smile Detection")
    stop_button = st.button("Stop Smile Detection")

    if start_button:
        is_running = True
        smile_detected = False
        cap = cv2.VideoCapture(0)

    if stop_button:
        is_running = False
        if cap is not None:
            cap.release()
        st.warning("Smile detection stopped.")

    frame_placeholder = st.empty()

    while is_running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break

        frame, smile_detected = detect_smile(frame)
        frame_placeholder.image(frame, channels="BGR")

        if smile_detected:
            is_running = False
            if cap is not None:
                cap.release()
            st.success("Smile detected! Click 'Start Smile Detection' to continue.")

    if cap is not None:
        cap.release()

if __name__ == "__main__":
    main()