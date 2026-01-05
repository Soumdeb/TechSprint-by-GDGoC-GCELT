import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from single_img_analyzer import SkinLesionClassifier
from progress_tracking import ProgressTracker
import class_accuracy
import about

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="DERMAC AI",
    page_icon="🔬",
    layout="wide"
)

# ----------------------------------
# CUSTOM CSS (CLIENT-FRIENDLY UI)
# ----------------------------------
st.markdown("""
<style>
/* ---------- GLOBAL DARK THEME ---------- */
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", Arial, sans-serif;
    background-color: #020617;
    color: #e5e7eb;
}

/* Main container */
.block-container {
    padding: 2rem 3rem;
    max-width: 1200px;
}

/* Headings */
h1, h2, h3 {
    color: #f9fafb;
    font-weight: 700;
}

h1 {
    font-size: 2.4rem;
}

/* Sub headers */
.sub-header {
    font-size: 1.4rem;
    font-weight: 600;
    margin: 1.5rem 0 1rem 0;
    color: #e5e7eb;
}

/* Cards */
.card {
    background: linear-gradient(145deg, #0f172a, #020617);
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1.5rem;
}

/* Metrics */
div[data-testid="metric-container"] {
    background: linear-gradient(145deg, #020617, #0f172a);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem;
}

div[data-testid="metric-container"] label {
    color: #9ca3af;
}

/* File uploader */
section[data-testid="stFileUploader"] {
    background-color: #020617;
    border: 1px dashed #334155;
    border-radius: 12px;
}

/* DataFrames */
.stDataFrame {
    background-color: #020617;
    border-radius: 12px;
    border: 1px solid #1e293b;
}

/* Buttons */
button[kind="primary"] {
    background-color: #2563eb;
    border-radius: 10px;
    font-weight: 600;
}

button[kind="primary"]:hover {
    background-color: #1d4ed8;
}

/* Progress bar */
div[data-testid="stProgress"] > div > div {
    background-image: linear-gradient(90deg, #22c55e, #16a34a);
}

/* Images */
img {
    border-radius: 12px;
}

/* Popover Menu */
div[data-testid="stPopover"] {
    border-radius: 12px;
    background-color: #020617;
    border: 1px solid #1e293b;
}

/* ---------- RISK ---------- */
.risk-low { color: #22c55e; font-weight: 700; font-size: 1.3rem; }
.risk-medium { color: #facc15; font-weight: 700; font-size: 1.3rem; }
.risk-high { color: #ef4444; font-weight: 700; font-size: 1.3rem; }

            
/* Hide Streamlit branding */
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


st.set_option("client.showErrorDetails", True)

# ----------------------------------
# LOAD MODELS (ONCE)
# ----------------------------------
@st.cache_resource
def load_classifier():
    return SkinLesionClassifier("dermac_ai_model.tflite")

@st.cache_resource
def load_tracker():
    return ProgressTracker()

clf = load_classifier()
tracker = load_tracker()

# ----------------------------------
# SESSION STATE
# ----------------------------------
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# ----------------------------------
# NAVIGATION
# ----------------------------------
def navigation():
    with st.popover("☰ Menu"):
        selected = option_menu(
            None,
            ["Home", "Single Image Analysis", "Progress Tracking", "Class Accuracy", "About"],
            icons=[
                "house-fill",
                "bar-chart-line-fill",
                "clipboard-check-fill",
                "bullseye",
                "info-circle-fill"
            ],
            default_index=0
        )
        st.session_state.current_page = selected

# ----------------------------------
# PAGES
# ----------------------------------
def home():
    st.title("DERMAC AI")
    st.info("AI-powered skin lesion classification and progression tracking.")

# -------- SINGLE IMAGE ANALYSIS --------
def single_image_analysis_page():
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Upload Image")
        uploaded = st.file_uploader(
            "Select a clear lesion image (JPG/PNG)",
            type=["jpg", "jpeg", "png", "bmp"],
            key="single_upload"
        )

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.caption(f"File: {uploaded.name} | Size: {image.size}")

    with col2:
        st.markdown("### Classification Results")
        if uploaded:
            with st.spinner("Analyzing image..."):
                is_skin, _ = clf.detect_skin(image)
                time.sleep(0.6)

            if not is_skin:
                st.error("No skin region detected.")
                return

            prediction = clf.predict(image)
            idx = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            pred_class = clf.label_map[idx]
            risk = clf.risk_levels.get(pred_class, "Unknown")

            st.success(f"Predicted Class: {pred_class}")
            st.metric("Confidence", f"{confidence:.2%}")

            risk_class = (
                "risk-high" if risk == "High"
                else "risk-medium" if risk == "Medium"
                else "risk-low"
            )

            st.markdown("**Risk Level**")
            st.markdown(
                f'<div class="{risk_class}">{risk}</div>',
                unsafe_allow_html=True
            )

            st.progress(confidence)

            prob_df = pd.DataFrame({
                "Class": [clf.label_map[i] for i in range(len(prediction))],
                "Probability": prediction
            }).sort_values("Probability", ascending=False)

            st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))

# -------- PROGRESS TRACKING --------
def progress_tracking_page():
    st.markdown('<div class="sub-header">Lesion Progress Tracking</div>', unsafe_allow_html=True)
    st.write("Upload two images of the same lesion taken at different times to track changes.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Baseline Image**")
        baseline_img = st.file_uploader(
            "Upload baseline image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="baseline"
        )
        if baseline_img:
            baseline_image = Image.open(baseline_img).convert("RGB")
            st.image(baseline_image, caption="Baseline Image", use_container_width=True)

    with col2:
        st.markdown("**Follow-up Image**")
        followup_img = st.file_uploader(
            "Upload follow-up image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="followup"
        )
        if followup_img:
            followup_image = Image.open(followup_img).convert("RGB")
            st.image(followup_image, caption="Follow-up Image", use_container_width=True)

    if baseline_img and followup_img:
        with st.spinner("Analyzing lesion progression..."):

            baseline_resized = baseline_image.resize((300, 300))
            followup_resized = followup_image.resize((300, 300))

            baseline_np = np.array(baseline_resized)
            followup_np = np.array(followup_resized)


            #Correct tracker usage
            metrics = tracker.calculate_metrics(baseline_np, followup_np)

            #heatmap call (matches tracker exactly)
            heatmap = tracker.generate_heatmap_delta(
                baseline_np,
                metrics["registered_img2"],
                metrics["mask1"],
                metrics["mask2"]
            )


        st.markdown('<div class="sub-header">Progress Analysis Results</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Area Change", f"{metrics['area_change']:.1f}%")
        col2.metric("Redness Change", f"{metrics['redness_change']:.1f}%")
        col3.metric("Pigmentation Change", f"{metrics['pigmentation_change']:.1f}%")

        st.markdown("**Visual Analysis**")
        fig_col1, fig_col2, fig_col3 = st.columns(3)

        with fig_col1:
            st.image(baseline_np, caption="Baseline Image", use_container_width=True)
            st.image(metrics["mask1"], caption="Baseline Mask", use_container_width=True)

        with fig_col2:
            st.image(metrics["registered_img2"], caption="Follow-up (Registered)", use_container_width=True)
            st.image(metrics["mask2"], caption="Follow-up Mask", use_container_width=True)


        with fig_col3:
            st.image(heatmap, caption="Change Heatmap", use_container_width=True)

        st.markdown("**Interpretation**")
        interpretation_text = ""

        if abs(metrics["area_change"]) > 20:
            interpretation_text += "• **Significant size change detected**\n"
        elif abs(metrics["area_change"]) > 10:
            interpretation_text += "• **Moderate size change detected**\n"

        if abs(metrics["redness_change"]) > 25:
            interpretation_text += "• **Significant redness change detected**\n"
        elif abs(metrics["redness_change"]) > 15:
            interpretation_text += "• **Moderate redness change detected**\n"

        if abs(metrics["pigmentation_change"]) > 25:
            interpretation_text += "• **Significant pigmentation change detected**\n"
        elif abs(metrics["pigmentation_change"]) > 15:
            interpretation_text += "• **Moderate pigmentation change detected**\n"

        if interpretation_text:
            st.warning(interpretation_text)
        else:
            st.success("No significant changes detected. Lesion appears stable.")


# -------- CLASS ACCURACY --------
def class_accuracy_page():
    st.header("Class-wise Model Accuracy")
    class_accuracy.app()

# ----------------------------------
# ROUTER
# ----------------------------------
def router():
    page = st.session_state.current_page
    if page == "Home":
        home()
    elif page == "Single Image Analysis":
        single_image_analysis_page()
    elif page == "Progress Tracking":
        progress_tracking_page()
    elif page == "Class Accuracy":
        class_accuracy_page()
    elif page == "About":
        about.app()

# ----------------------------------
# RUN APP
# ----------------------------------
navigation()
router()
