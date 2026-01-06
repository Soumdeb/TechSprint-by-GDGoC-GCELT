import streamlit as st

def app():
    # ---------- Global CSS ----------
    st.markdown("""
    <style>
    /* Page width */
    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* Main title */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #e5e7eb;
        margin-bottom: 1rem;
    }

    /* Section titles */
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f9fafb;
        margin-top: 3rem;
        margin-bottom: 1rem;
    }

    /* Card style */
    .card {
        background: linear-gradient(145deg, #0f172a, #020617);
        padding: 1.4rem;
        border-radius: 14px;
        border: 1px solid #1e293b;
        margin-bottom: 1.5rem;
    }

    /* Muted text */
    .muted {
        color: #9ca3af;
    }

    /* Team styles */
    .team-name {
        text-align: center;
        font-weight: 700;
        margin-top: 0.6rem;
        color: #f3f4f6;
    }

    .team-role {
        text-align: center;
        font-size: 0.85rem;
        color: #9ca3af;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- Title ----------
    st.markdown('<div class="main-title">About DERMAC AI</div>', unsafe_allow_html=True)

    # ---------- Intro ----------
    st.markdown("""
    **DERMAC AI** is an **online, web-based dermatological decision-support platform**
    developed exclusively for **educational and research purposes**.

    The project was built by **Soumyadeb (Machine Learning)** and
    **Adhiraj (Cloud & Firebase)**, with contributions from
    **Soumyasundar Sai (Research)** and **Swastika Khan (CSS & Planning)**.

    The platform leverages artificial intelligence to assist clinicians and researchers
    in **skin lesion analysis**, supporting both **single-image diagnosis** and
    **web-based longitudinal progress tracking**, while maintaining strict adherence
    to **fairness, security, and privacy principles**.
    """)

    # ---------- Key Highlights ----------
    st.markdown('<div class="section-title">Key Highlights</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Web-Based Skin Lesion Analysis Platform</h3>
        <p class="muted">
        DERMAC AI is delivered as a <b>browser-based web application</b>, not a mobile app.
        It enables clinicians and researchers to upload and analyze individual skin lesion
        images through a secure and authenticated interface.
        </p>
    </div>

    <div class="card">
        <h3>Single Image Analyzer & Progress Tracking</h3>
        <ul class="muted">
            <li>Single-image skin lesion classification for immediate analysis</li>
            <li>Web-based progress tracking to compare historical and current images</li>
            <li>Visualization of lesion evolution using metrics and heatmap-based insights</li>
        </ul>
    </div>

    <div class="card">
        <h3>Model & Dataset</h3>
        <p class="muted">
        A custom <b>Convolutional Neural Network (CNN)</b> with approximately
        <b>1.44 million parameters</b> was designed and trained from scratch.
        Training and evaluation were performed using the
        <b>ISIC (International Skin Imaging Collaboration)</b> skin cancer dataset.
        </p>
    </div>

    <div class="card">
        <h3>Security & Privacy by Design</h3>
        <p class="muted">
        Patient-related data is handled with a security-first approach.
        As no database or persistent storage is currently connected,
        no personal data is stored or transmitted, eliminating the risk
        of privacy leakage. Controlled access workflows and encryption
        mechanisms are applied at the system level to ensure confidentiality
        and prevent unauthorized data exposure.
        </p>
    </div>

    <div class="card">
        <h3>Beyond Baseline Solutions</h3>
        <p class="muted">
        Beyond standard lesion classification, DERMAC AI integrates
        research-oriented evaluation, progress visualization, and
        fairness-aware model development, establishing a clinician-centric
        AI research platform.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Team ----------
    st.markdown('<div class="section-title">Team</div>', unsafe_allow_html=True)


    

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image("assets/Soumyadeb.jpeg", use_container_width=True)
        st.markdown('<div class="team-name">Soumyadeb Nandy</div>', unsafe_allow_html=True)
        st.markdown('<div class="team-role">Machine Learning</div>', unsafe_allow_html=True)

    with col2:
        st.image("assets/adhiraj.jpg", use_container_width=True)
        st.markdown('<div class="team-name">Adhiraj Ghosh</div>', unsafe_allow_html=True)
        st.markdown('<div class="team-role">Cloud & Firebase</div>', unsafe_allow_html=True)

    with col3:
        st.image("assets/soumosundar.jpeg", use_container_width=True)
        st.markdown('<div class="team-name">Soumyasundar Sai</div>', unsafe_allow_html=True)
        st.markdown('<div class="team-role">Research</div>', unsafe_allow_html=True)

    with col4:
        st.image("assets/swastika.jpg", use_container_width=True)
        st.markdown('<div class="team-name">Swastika Khan</div>', unsafe_allow_html=True)
        st.markdown('<div class="team-role">CSS & Planning</div>', unsafe_allow_html=True)

        # ---------- Footer ----------
    st.markdown(
        """
        <hr style="border:0; border-top:1px solid #1f2933; margin-top:40px;"/>
        <div style="text-align:center; color:#6b7280; font-size:12px; padding:10px 0;">
            © Copyright 2026, All Rights Reserved by <strong>DERMAC AI</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
