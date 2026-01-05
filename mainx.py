import streamlit as st
import requests
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


FIREBASE_API_KEY = "AIzaSyBPko0wgXlCiUr2Yb2rwg8q_SV6N2_VySE"

def signup(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    r = requests.post(url, json=payload)
    return r.json()

def login(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    r = requests.post(url, json=payload)
    return r.json()


if "page" not in st.session_state:
    st.session_state.page = "landing"



import time

def animated_text(text, speed=0.01):
    placeholder = st.empty()
    typed = ""

    for char in text:
        typed += char
        placeholder.markdown(
            f"<p style='font-size:22px; opacity:0.8; text-align:center;'>{typed}</p>",
            unsafe_allow_html=True
        )
        time.sleep(speed)
    

def landing_page():
    st.markdown(
        """
        <style>
        .hero {
            text-align: center;
            padding: 120px 20px;
        }
        .hero h1 {
            font-size: 64px;
            font-weight: 700;
            margin-bottom: -150px;
        }
        .hero p {
            font-size: 22px;
            opacity: 0.8;
            margin-top: -300px;
            margin-bottom: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero">
            <h1>DERMAC AI</h1>
            
    
            
        </div>
        """,
        unsafe_allow_html=True
    )

    animated_text("Where Artificial Intelligence Meets Skin Science")

    col1, col2, col3 = st.columns([3, 2, 3])
    with col2:
        if st.button("Get Started Now →", use_container_width=True):
            st.session_state.page = "auth"
            st.rerun()



def auth_page():
    st.title("DERMAC AI – Authentication")

    # Track auth mode
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "signup"

    if st.session_state.auth_mode == "signup":
        st.subheader("Create an Account")

        full_name = st.text_input("Full Name", key="signup_fullname")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")

        if st.button("Create Account"):
            res = signup(email, password)
            if "idToken" in res:
             st.success("Account created successfully. Please log in.")
             st.session_state.auth_mode = "login"
             st.rerun()

            else:
                st.error(res.get("error", {}).get("message", "Signup failed"))

        st.markdown("<small>Already have an account?</small>", unsafe_allow_html=True)
        if st.button("Login here"):
            st.session_state.auth_mode = "login"
            st.rerun()   
         

    else:  # LOGIN MODE
        st.subheader("Login")

        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            res = login(email, password)
            if "idToken" in res:
                st.session_state["user"] = res
                st.session_state.page = "app"
                st.rerun()
            else:
                st.error(res.get("error", {}).get("message", "Login failed"))

        st.markdown("<small>Don’t have an account?</small>", unsafe_allow_html=True)
        if st.button("Create one"):
            st.session_state.auth_mode = "signup"
            st.rerun()

st.set_page_config(page_title="DERMAC AI", layout="centered")

if st.session_state.page == "landing":
    landing_page()
    st.stop()

if st.session_state.page == "auth":
    auth_page()
    st.stop()



# ===== PROTECTED APP =====

if "user" not in st.session_state:
    st.session_state.page = "auth"
    st.rerun()

def logout():
    st.session_state.pop("user", None)
    st.session_state.page = "landing"
    st.rerun()

    with st.sidebar:
     st.markdown("### Account")
     st.write(st.session_state["user"]["email"])

    if st.button("Logout"):
        logout()


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

/* ---------- GLOBAL DARK THEME (KEEP) ---------- */
html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", Arial, sans-serif;
    background-color: #020617;
    color: #e5e7eb;
}

/* Remove Streamlit padding so we can center card */
.block-container {
    padding: 0;
    max-width: 100%;
}

/* Hide Streamlit branding */
footer { visibility: hidden; }
header { visibility: hidden; }

/* ---------- AUTH PAGE ONLY ---------- */
.auth-page {
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    background: radial-gradient(circle at top, #1a1f2b, #020617);
}

/* Main login card */
.auth-card {
    display: flex;
    width: 900px;
    height: 520px;
    background: linear-gradient(145deg, #0f172a, #020617);
    border: 1px solid #1e293b;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 40px 80px rgba(0,0,0,0.6);
}

/* Left side (form) */
.auth-left {
    flex: 1;
    padding: 60px;
    color: #f9fafb;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

/* Title */
.auth-title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 32px;
}

/* ---------------- STREAMLIT INPUT FIX ---------------- */
/* Streamlit text inputs */
.auth-left div[data-testid="stTextInput"] input {
    background-color: #020617 !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 14px !important;
    color: #e5e7eb !important;
}

/* Remove Streamlit input label spacing */
.auth-left label {
    color: #9ca3af;
    font-size: 0.9rem;
}

/* Spacing between inputs */
.auth-left div[data-testid="stTextInput"] {
    margin-bottom: 18px;
}

/* ---------------- BUTTON FIX ---------------- */
.auth-btn .stButton > button {
    width: 100%;
    height: 48px;
    background: linear-gradient(90deg, #6d28d9, #9333ea);
    border-radius: 999px;
    border: none;
    font-weight: 600;
    color: white;
}

.auth-btn .stButton > button:hover {
    background: linear-gradient(90deg, #7c3aed, #a855f7);
}

/* Right side (image panel) */
.auth-right {
    flex: 1;
    background: #f5f3ff;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 40px;
    text-align: center;
}

/* Right panel text */
.auth-right h2 {
    color: #7c3aed;
    font-weight: 600;
    margin-top: 20px;
    font-size: 1.3rem;
}

/* Image */
.auth-right img {
    max-width: 320px;
    border-radius: 12px;
}



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

    st.markdown("## Class Information")
    st.markdown("Overview of supported skin lesion classes and associated risk levels.")

    cols = st.columns(3)

    for i, (idx, name) in enumerate(clf.label_map.items()):
        with cols[i % 3]:
            with st.expander(f"{name} ({clf.risk_levels.get(name, 'Unknown')} Risk)"):
                
                if name == "Melanoma":
                    st.write("Highly aggressive skin cancer. Immediate medical evaluation is required.")
                elif name in ["Basal Cell Carcinoma", "Squamous Cell Carcinoma"]:
                    st.write("Common skin cancers. Early detection leads to high treatment success.")
                elif name == "Actinic Keratosis":
                    st.write("Precancerous lesion caused by sun exposure. Monitoring recommended.")
                else:
                    st.write("Generally benign lesion with low malignancy risk.")

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
                time.sleep(0.3)

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