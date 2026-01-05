import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import time

st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---------- MAIN CONTAINER ---------- */
.block-container {
    max-width: 1150px !important;
    padding-top: 2rem !important;
}

/* ---------- CARDS ---------- */
.card {
    background: linear-gradient(145deg, #0b1220, #020617) !important;
    border: 1px solid #1e293b !important;
    border-radius: 18px !important;
    padding: 1.6rem !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.45);
    margin-bottom: 1.5rem;
}

/* ---------- TITLES ---------- */
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #f9fafb;
    margin-bottom: 2rem;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e5e7eb;
    margin-bottom: 1rem;
}

/* ---------- FILE UPLOADER ---------- */
div[data-testid="stFileUploader"] {
    border-radius: 14px;
    border: 1px dashed #334155;
    padding: 1rem;
}

/* ---------- BADGE ---------- */
.badge {
    background: linear-gradient(90deg, #16a34a, #22c55e);
    color: #022c22;
    font-weight: 700;
    padding: 0.8rem 1.2rem;
    border-radius: 14px;
    font-size: 1.1rem;
}

/* ---------- PROGRESS ---------- */
div[data-testid="stProgress"] > div > div {
    background-image: linear-gradient(90deg, #3b82f6, #22c55e);
}

/* ---------- DATAFRAME ---------- */
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    border: 1px solid #1e293b;
}

/* ---------- METRICS ---------- */
.metric-label {
    color: #9ca3af;
    font-size: 0.85rem;
    margin-top: 1rem;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
}

/* ---------- RISK ---------- */
.risk-low { color: #22c55e; font-weight: 700; }
.risk-medium { color: #facc15; font-weight: 700; }
.risk-high { color: #ef4444; font-weight: 700; }

</style>
""", unsafe_allow_html=True)


# ----------------------------------
# CLASSIFIER
# ----------------------------------
class SkinLesionClassifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()
        self.label_map = {
            0: "Actinic Keratosis",
            1: "Basal Cell Carcinoma",
            2: "Dermatofibroma",
            3: "Melanoma",
            4: "Nevus",
            5: "Pigmented Benign Keratosis",
            6: "Seborrheic Keratosis",
            7: "Squamous Cell Carcinoma",
            8: "Vascular Lesion"
        }
        self.risk_levels = {
            "Melanoma": "High",
            "Basal Cell Carcinoma": "Medium",
            "Squamous Cell Carcinoma": "Medium",
            "Actinic Keratosis": "Medium",
            "Pigmented Benign Keratosis": "Low",
            "Seborrheic Keratosis": "Low",
            "Nevus": "Low",
            "Dermatofibroma": "Low",
            "Vascular Lesion": "Low"
        }

    def load_model(self):
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def detect_skin(self, image, threshold=0.15):
        try:
            img_array = np.array(image)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            ratio = np.count_nonzero(mask) / mask.size
            return ratio >= threshold, ratio
        except Exception:
            return True, 1.0

    def preprocess(self, image):
        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new("RGB", image.size, (255, 255, 255))
            image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1])
            image = background

        image = image.resize((100, 75))
        img_array = np.array(image) / 255.0

        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        return img_array.astype(np.float32)

    def predict(self, image):
        processed = self.preprocess(image)
        input_data = np.expand_dims(processed, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output[0]

# ----------------------------------
# APP
# ----------------------------------
def app():
    st.markdown('<div class="main-title">Single Image Skin Lesion Analysis</div>', unsafe_allow_html=True)

    if "classifier" not in st.session_state:
        st.session_state.classifier = SkinLesionClassifier("dermac_ai_model.tflite")

    clf = st.session_state.classifier

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Select a clear lesion image (JPG / PNG)",
            type=["jpg", "jpeg", "png", "bmp"]
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, use_container_width=True)
            st.caption(f"{uploaded.name} • {image.size[0]}×{image.size[1]} px")

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Classification Results</div>', unsafe_allow_html=True)

        if uploaded:
            with st.spinner("Analyzing image…"):
                is_skin, _ = clf.detect_skin(image)
                time.sleep(0.6)

            if not is_skin:
                st.error("No skin region detected.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            prediction = clf.predict(image)
            idx = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            pred_class = clf.label_map[idx]
            risk = clf.risk_levels.get(pred_class, "Unknown")

            st.markdown(
                f'<div class="badge">Predicted Class: {pred_class}</div>',
                unsafe_allow_html=True
            )

            st.markdown('<div class="metric-label">Confidence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{confidence:.2%}</div>', unsafe_allow_html=True)
            st.progress(confidence)

            risk_class = (
                "risk-high" if risk == "High"
                else "risk-medium" if risk == "Medium"
                else "risk-low"
            )

            st.markdown('<div class="metric-label">Risk Level</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="{risk_class}">{risk}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded and is_skin:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Class Probabilities</div>', unsafe_allow_html=True)

        prob_df = pd.DataFrame({
            "Class": [clf.label_map[i] for i in range(len(prediction))],
            "Probability": prediction
        }).sort_values("Probability", ascending=False)

        st.dataframe(
            prob_df.style.format({"Probability": "{:.2%}"}),
            use_container_width=True
        )

        st.download_button(
            "Download Report",
            data=f"Prediction: {pred_class}\nConfidence: {confidence:.2%}\nRisk: {risk}",
            file_name=f"dermac_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        )

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    app()
