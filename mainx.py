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

# ---------------------
# FIREBASE AUTH SETUP
# ---------------------
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

# ----------------------------------
# LANDING PAGE 
# ----------------------------------

if "page" not in st.session_state:
    st.session_state.page = "landing"


def animated_text(text, speed=0.02, key="animated_text"):
    if key in st.session_state:
        st.markdown(
            f"<p style='font-size:22px; opacity:0.8; text-align:center;'>{text}</p>",
            unsafe_allow_html=True
        )
        return
    
    placeholder = st.empty()
    typed = []

    for char in text:
        typed.append(char)
        placeholder.markdown(
            f"<p style='font-size:22px; opacity:0.8; text-align:center;'>"
            f"{''.join(typed)}</p>",
            unsafe_allow_html=True
        )
        time.sleep(speed)

    st.session_state[key] = True

    
def landing_page():
    st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
#MainMenu, footer, header { visibility: hidden; }

html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(
        1200px 600px at 50% -200px,
        #0b1220 0%,
        #020617 60%
    );
    color: #e5e7eb;
    font-family: 'Inter', sans-serif;
}

/* ---------- HERO SECTION ---------- */
.hero {
    text-align: center;
    padding-top: 140px;
    padding-bottom: 48px;
}

.hero h1 {
    font-size: 56px;
    font-weight: 300;
    margin-bottom: 14px;
    color: #f9fafb;
    letter-spacing: -0.02em;
}

.hero p {
    font-size: 18px;
    color: #9ca3af;
    margin-bottom: 36px;
    max-width: 640px;
    margin-left: auto;
    margin-right: auto;
}

/* ---------- PRIMARY BUTTON (DARK RED) ---------- */
.stButton button {
    background: #7f1d1d;
    color: #f9fafb;
    border-radius: 10px;
    padding: 13px 18px;
    font-size: 14px;
    font-weight: 600;
    border: 1px solid #991b1b;
    width: 100%;
}

.stButton button:hover {
    background: #991b1b;
}

/* ---------- PREVENT LAYOUT JUMP ---------- */
.block-container {
    padding-top: 0 !important;
}

</style>
""", unsafe_allow_html=True)


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



# ----------------------------------
# AUTHENTICATION
# ----------------------------------

def auth_page():

    st.markdown(
        """
    <style>
    /* Hide Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #020617, #020617);
        color: #e5e7eb;
    }

    /* Auth Card */
    .auth-container {
        max-width: 440px;
        margin: 90px auto;
        padding: 42px 40px;
        background: #020617;
        border-radius: 14px;
        border: 1px solid #1f2933;
    }

    /* Title */
    .auth-title {
        text-align: center;
        font-size: 28px;
        font-weight: 600;
        margin-bottom: 32px;
        color: #f9fafb;
    }

    /* Labels */
    label {
        color: #9ca3af !important;
        font-size: 13px;
        margin-bottom: 6px;
    }

    /* Inputs */
    input {
        background: #020617 !important;
        border: 1px solid #1f2933 !important;
        border-radius: 10px !important;
        padding: 13px !important;
        color: #f9fafb !important;
    }

    input:focus {
        border-color: #b91c1c !important;
        outline: none;
    }

    /* Primary button (RED) */
    .stButton button {
        background: #b91c1c;
        color: white;
        border-radius: 10px;
        padding: 13px;
        font-size: 14px;
        font-weight: 600;
        border: none;
    }

    .stButton button:hover {
        background: #991b1b;
    }

    /* Secondary link button */
    .stButton:nth-of-type(2) button {
        background: transparent;
        color: #9ca3af;
        border: 1px solid #1f2933;
        margin-top: 6px;
    }

    /* Footer text */
    .auth-footer {
        text-align: center;
        font-size: 13px;
        color: #6b7280;
        margin-top: 20px;
    }

    </style>
        """,
        unsafe_allow_html=True
    )

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

        st.markdown("<medium>Already have an account?</medium>", unsafe_allow_html=True)
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

        st.markdown("<medium>Don’t have an account?</medium>", unsafe_allow_html=True)
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


# ----------------------------------
# PROTED APP
# ----------------------------------

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

st.markdown(
    """
    <style>
    /* Hide Streamlit chrome */
        #MainMenu, footer, header {visibility: hidden;}

    /* Remove layout influence */
    div[data-testid="stPopover"] {
        all: unset;
    }

    /* Small fixed menu button (top-right) */
    button[data-testid="stPopoverButton"] {
        position: fixed !important;
        top: 14px;
        right: 16px;        /* moved to top-right */
        left: auto !important;

        z-index: 9999;
        background: #020617;
        color: #e5e7eb;
        border: 1px solid #1f2933;
        border-radius: 8px;

        padding: 6px 8px;
        font-size: 13px;
        line-height: 1;
        min-width: unset;
        width: auto;
    }

    button[data-testid="stPopoverButton"]:hover {
        border-color: #7f1d1d;   /* muted red */
        background: #020617;
    }

    /* Popover panel */
    div[data-testid="stPopover"] {
        position: fixed !important;
        top: 48px !important;
        right: 16px !important;
        left: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


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
# NAVIGATION
# ----------------------------------
def navigation():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

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
            default_index=[
                "Home",
                "Single Image Analysis",
                "Progress Tracking",
                "Class Accuracy",
                "About"
            ].index(st.session_state.current_page),
            styles={
                "container": {"padding": "0"},
                "icon": {"font-size": "14px"},
                "nav-link": {
                    "font-size": "14px",
                    "padding": "10px 12px",
                    "text-align": "left",
                },
                "nav-link-selected": {
                    "background-color": "#7f1d1d",
                    "color": "#f9fafb",
                },
            }
        )

        if selected != st.session_state.current_page:
            st.session_state.current_page = selected
            st.rerun()


# ----------------------------------
# PAGES
# ----------------------------------
def home():
    st.markdown(
        """
    <style>
    /* Hide Streamlit chrome */
        #MainMenu, footer, header {visibility: hidden;}

    /* Page spacing */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* Title refinement */
    h1 {
        font-size: 40px;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }

    /* Info banner */
    div[data-testid="stAlert"] {
        max-width: 1000px;
        margin: 0 0 36px 0;
        padding: 14px 18px;
        border-radius: 10px;
        background: #020617;
        border: 1px solid #7f1d1d;
        font-size: 14px;
    }

    /* Section header spacing */
    h2 {
        margin-top: 10px;
        margin-bottom: 6px;
    }

    /* Expander cards – cleaner */
    div[data-testid="stExpander"] {
        background: #020617;
        border: 1px solid #1f2933;
        border-radius: 12px;
        transition: border-color 0.15s ease;
    }

    div[data-testid="stExpander"]:hover {
        border-color: #374151;
    }

    /* Expander title text */
    div[data-testid="stExpander"] summary {
        font-size: 14px;
        font-weight: 500;
        color: #f9fafb;
    }

    /* Expander content text */
    div[data-testid="stExpander"] p {
        font-size: 13px;
        color: #9ca3af;
        line-height: 1.5;
    }
    </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.title("DERMAC AI")
    st.info("AI-powered skin lesion classification and progression tracking.")

    # Section
    st.markdown("## Class Information")
    st.markdown(
        "<p>Overview of supported skin lesion classes and associated risk levels.</p>",
        unsafe_allow_html=True
    )

    cols = st.columns(3)

    for i, (idx, name) in enumerate(clf.label_map.items()):
        with cols[i % 3]:
            with st.expander(
                f"{name} ({clf.risk_levels.get(name, 'Unknown')} Risk)"
            ):
                if name == "Melanoma":
                    st.write(
                        "Highly aggressive skin cancer. Immediate medical evaluation is required."
                    )
                elif name in ["Basal Cell Carcinoma", "Squamous Cell Carcinoma"]:
                    st.write(
                        "Common skin cancers. Early detection leads to high treatment success."
                    )
                elif name == "Actinic Keratosis":
                    st.write(
                        "Precancerous lesion caused by sun exposure. Monitoring recommended."
                    )
                else:
                    st.write(
                        "Generally benign lesion with low malignancy risk."
                    )

# -------- SINGLE IMAGE ANALYSIS --------
def single_image_analysis_page():
    st.markdown('<div class="main-title">Single Image Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <style>

    /* ---------- GLOBAL ---------- */
    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(
            1200px 600px at 50% -200px,
            #0b1220 0%,
            #020617 60%
        );
        color: #e5e7eb;
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* Remove only the FIRST empty spacer Streamlit injects */
    .block-container > div:first-child:empty {
        display: none !important;
    }


    /* ---------- MAIN CONTAINER ---------- */
    .block-container {
        max-width: 1000px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 1.5rem !important;
    }

    /* ---------- TITLES ---------- */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f9fafb;
        margin-bottom: 1.4rem;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 0.6rem;
    }

    /* ---------- FILE UPLOADER ---------- */
    div[data-testid="stFileUploader"] {
        background: #020617;
        border: 1px dashed #1f2933;
        border-radius: 12px;
        padding: 0.9rem;
    }


    /* ---------- DATAFRAME ---------- */
    div[data-testid="stDataFrame"] {
        background: #020617;
        border-radius: 12px;
        border: 1px solid #1f2933;
    }

    /* ---------- METRICS ---------- */
    [data-testid="stMetricValue"] {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f9fafb;
    }

    /* ---------- RISK ---------- */
    .risk-low    { color: #22c55e; font-weight: 600; }
    .risk-medium { color: #facc15; font-weight: 600; }
    .risk-high   { color: #dc2626; font-weight: 700; }

    </style>
    """, unsafe_allow_html=True)


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

st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(
        1200px 600px at 50% -200px,
        #0b1220 0%,
        #020617 60%
    );
    color: #e5e7eb;
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ---------- MAIN CONTAINER ---------- */
.block-container {
    max-width: 1000px !important;
    padding-top: 1.8rem !important;
    padding-bottom: 1.5rem !important;
}

/* ---------- HEADERS ---------- */
.sub-header {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f9fafb;
    margin-bottom: 0.6rem;
}

.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e5e7eb;
    margin-bottom: 0.5rem;
}

/* ---------- FILE UPLOADER ---------- */
div[data-testid="stFileUploader"] {
    background: #020617;
    border: 1px dashed #1f2933;
    border-radius: 12px;
    padding: 1rem;
}

/* ---------- METRICS ---------- */
[data-testid="stMetric"] {
    background: #020617;
    border: 1px solid #1f2933;
    border-radius: 12px;
    padding: 0.6rem;
}

[data-testid="stMetricValue"] {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f9fafb;
}

/* ---------- BUTTONS ---------- */
.stButton button {
    background: #7f1d1d;
    color: #f9fafb;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid #991b1b;
}

.stButton button:hover {
    background: #991b1b;
}

/* ---------- ALERTS ---------- */
.stWarning {
    background: rgba(127, 29, 29, 0.15);
    border-left: 4px solid #dc2626;
}

.stSuccess {
    background: rgba(34, 197, 94, 0.15);
    border-left: 4px solid #22c55e;
}

</style>
""", unsafe_allow_html=True)


def progress_tracking_page():

    st.markdown('<div class="sub-header">Lesion Progress Tracking</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#9ca3af; font-size:0.95rem; margin-bottom:1.2rem;">'
        'Upload two images of the same lesion taken at different times to track changes.'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Baseline Image</div>', unsafe_allow_html=True)
        baseline_img = st.file_uploader(
            "Upload baseline image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="baseline"
        )
        if baseline_img:
            baseline_image = Image.open(baseline_img).convert("RGB")
            st.image(baseline_image, caption="Baseline Image", use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Follow-up Image</div>', unsafe_allow_html=True)
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

    st.markdown("""
<style>

/* ---------- PLOTLY CHART ---------- */
div[data-testid="stPlotlyChart"] {
    background: #020617 !important;
    border-radius: 14px;
    border: 1px solid #1f2933;
    padding: 0.8rem;
}

/* ---------- DATAFRAME CONTAINER ---------- */
div[data-testid="stDataFrame"] {
    background: #020617 !important;
    border-radius: 14px;
    border: 1px solid #1f2933;
    padding: 0.4rem;
}

/* ---------- DATAFRAME INNER TABLE ---------- */
div[data-testid="stDataFrame"] table {
    background: #020617 !important;
    color: #e5e7eb;
}

/* ---------- DATAFRAME HEADER ---------- */
div[data-testid="stDataFrame"] thead tr th {
    background: #020617 !important;
    color: #9ca3af !important;
    border-bottom: 1px solid #1f2933 !important;
}

/* ---------- DATAFRAME ROWS ---------- */
div[data-testid="stDataFrame"] tbody tr {
    background: #020617 !important;
}

div[data-testid="stDataFrame"] tbody tr:hover {
    background: rgba(127, 29, 29, 0.12) !important;
}

/* ---------- EXPANDER ---------- */
div[data-testid="stExpander"] {
    background: rgba(2, 6, 23, 0.92);
    border: 1px solid #1f2933;
    border-radius: 14px;
}

/* Expander header */
div[data-testid="stExpander"] summary {
    color: #f9fafb;
    font-weight: 600;
}

/* ---------- REMOVE EXTRA SPACING ---------- */
div[data-testid="stExpander"] > div {
    padding-top: 0.6rem;
}

</style>
""", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="sub-header">Class-wise Model Accuracy</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="color:#9ca3af; font-size:0.95rem; margin-bottom:1.2rem;">'
        'Performance metrics for each lesion class based on validation data.'
        '</div>',
        unsafe_allow_html=True
    )

    # Embed the existing accuracy app
    class_accuracy.app()

    st.markdown('</div>', unsafe_allow_html=True)


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