import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================================================
# PAGE CONFIG (Mobile-friendly)
# =========================================================
st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐱🐶",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS (Mobile Responsive)
# =========================================================
st.markdown(
    """
    <style>
    /* Make images responsive */
    img {
        max-width: 100%;
        height: auto;
        border-radius: 12px;
    }

    /* Center titles on mobile */
    h1, h2, h3 {
        text-align: center;
    }

    /* Improve button & uploader touch area */
    button, input {
        font-size: 16px !important;
    }

    /* Card style containers */
    .card {
        padding: 1.2rem;
        border-radius: 14px;
        background-color: rgba(255,255,255,0.03);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# CONSTANTS
# =========================================================
MODEL_PATH = "cats_vs_dogs_mobilenetv2.keras"
CLASS_NAMES = ["Cat", "Dog"]
IMAGE_SIZE = (224, 224)

# =========================================================
# LOAD MODEL (Cached)
# =========================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =========================================================
# SIDEBAR (Optional Info)
# =========================================================
with st.sidebar:
    st.title("🐾 About")
    st.markdown(
        """
        **Cats vs Dogs Image Classifier**

        - 🧠 Model: MobileNetV2  
        - 🎯 Accuracy: **96%**  
        - ⚙️ Framework: TensorFlow  
        - 🚀 Deployed on Streamlit Cloud  

        Upload an image and get instant AI predictions.
        """
    )

    st.markdown("### 📸 Tips")
    st.markdown(
        """
        - Use clear images  
        - One animal only  
        - Face visible  
        """
    )

# =========================================================
# MAIN UI
# =========================================================
st.title("🐱🐶 Cats vs Dogs Image Classifier")
st.caption(
    "Upload an image and the AI will predict whether it’s a **Cat** or a **Dog**."
)

uploaded_file = st.file_uploader(
    "📤 Upload an image (JPG / JPEG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# PREPROCESSING
# =========================================================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================================================
# PREDICTION FLOW
# =========================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📷 Uploaded Image")
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("🧠 Analyzing image..."):
        x = preprocess_image(image)
        prob_dog = float(model.predict(x, verbose=0)[0][0])
        prob_cat = 1 - prob_dog

    pred_label = "Dog" if prob_dog >= 0.5 else "Cat"
    confidence = max(prob_dog, prob_cat)

    # =====================================================
    # RESULTS
    # =====================================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔮 Prediction")

    if pred_label == "Dog":
        st.success("🐶 **Dog**")
    else:
        st.success("🐱 **Cat**")

    col1, col2 = st.columns(2)
    col1.metric("🐶 Dog Probability", f"{prob_dog:.2%}")
    col2.metric("🐱 Cat Probability", f"{prob_cat:.2%}")

    st.progress(confidence)

    if confidence < 0.70:
        st.warning(
            "⚠️ The model is **not very confident**. "
            "Try a clearer image for better results."
        )
    else:
        st.info("✅ The model is confident about this prediction.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <hr>
    <div style="text-align:center; font-size:0.85rem; opacity:0.8;">
        ⚙️ Built with TensorFlow & Streamlit · 🚀 Streamlit Community Cloud<br>
        © 2026 <strong>Ometh</strong>. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)

