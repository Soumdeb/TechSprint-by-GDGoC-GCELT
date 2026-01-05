import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def app():
    # ---------- DATA (UNCHANGED LOGIC) ----------
    class_names = [
        "Actinic Keratosis",
        "Basal Cell Carcinoma",
        "Dermatofibroma",
        "Melanoma",
        "Nevus",
        "Pigmented Benign Keratosis",
        "Seborrheic Keratosis",
        "Squamous Cell Carcinoma",
        "Vascular Lesion"
    ]

    accuracies = [87.5, 91.2, 84.0, 79.3, 93.1, 90.7, 88.4, 85.9, 94.5]

    acc_df = pd.DataFrame({
        "Class": class_names,
        "Accuracy (%)": accuracies
    }).sort_values("Accuracy (%)", ascending=False)

    # ---------- UI CARD ----------
    st.markdown(
        """
        <div class="result-card">
            <h3>Model Performance Overview</h3>
            <p style="color:#8B949E;">
                Higher accuracy indicates better classification reliability per class.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------- CHART ----------
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        acc_df["Class"],
        acc_df["Accuracy (%)"],
        color="#58A6FF",
        alpha=0.85
    )

    ax.set_xlabel("Skin Disease Class", fontsize=10, color="#E6EDF3")
    ax.set_ylabel("Accuracy (%)", fontsize=10, color="#E6EDF3")
    ax.set_title(
        "Model Accuracy per Class",
        fontsize=13,
        color="#79C0FF",
        weight="bold"
    )

    ax.set_facecolor("#0D1117")
    fig.patch.set_facecolor("#0D1117")

    plt.xticks(rotation=45, ha="right", fontsize=9, color="#E6EDF3")
    plt.yticks(color="#E6EDF3")

    # ---------- VALUE LABELS ----------
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#FFB86C"
        )

    plt.tight_layout()
    st.pyplot(fig)

    # ---------- TABLE ----------
    with st.expander("View Detailed Accuracy Table"):
        st.dataframe(
            acc_df.reset_index(drop=True),
            use_container_width=True
        )
