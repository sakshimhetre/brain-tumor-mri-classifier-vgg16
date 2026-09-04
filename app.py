"""
Brain Tumor MRI Classifier — Professional Streamlit UI

Required files in the same folder:
    app.py
    brain_tumor_model.keras
    class_names.json

Run:
    python -m streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "brain_tumor_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

IMG_SIZE = (224, 224)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL DARK THEME
# ============================================================
#
# IMPORTANT:
# This CSS is only styling.
# No HTML components are used to build the application UI.
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background-color: #0b0f17;
        color: #f8fafc;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    header[data-testid="stHeader"] {
        background-color: transparent;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #111620;
        border-right: 1px solid #273040;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    /* ========================================================
       MAIN HEADINGS
       ======================================================== */

    h1,
    h2,
    h3 {
        color: #f8fafc !important;
    }

    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        background-color: #151b26;
        border: 1px dashed #4b5870;
        border-radius: 16px;
        padding: 12px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #121824;
        border-radius: 12px;
    }

    /* ========================================================
       METRIC CARDS
       ======================================================== */

    [data-testid="stMetric"] {
        background-color: #151b26;
        border: 1px solid #2a3445;
        border-radius: 14px;
        padding: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #8e9db2 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* ========================================================
       ALERTS
       ======================================================== */

    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #3a465b;
        background-color: #181f2c;
        color: #ffffff;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #6b7a94;
        color: #ffffff;
    }

    /* ========================================================
       IMAGE
       ======================================================== */

    [data-testid="stImage"] {
        border-radius: 14px;
    }

    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border-color: #252e3d;
    }

    /* ========================================================
       EXPANDER
       ======================================================== */

    [data-testid="stExpander"] {
        background-color: #121824;
        border: 1px solid #2a3445;
        border-radius: 14px;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer-text {
        text-align: center;
        color: #68768b;
        font-size: 13px;
        padding-top: 20px;
        padding-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained Keras model and class names once."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found:\n{CLASS_NAMES_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        class_names = json.load(file)

    # Support a list:
    # ["glioma", "meningioma", "notumor", "pituitary"]

    if isinstance(class_names, list):
        final_class_names = class_names

    # Support a dictionary if produced in another format
    elif isinstance(class_names, dict):
        try:
            final_class_names = [
                class_names[str(i)]
                for i in range(len(class_names))
            ]
        except Exception:
            final_class_names = list(
                class_names.values()
            )

    else:
        raise ValueError(
            "class_names.json must contain a list or dictionary."
        )

    return model, final_class_names


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess(image: Image.Image) -> np.ndarray:
    """
    Keep this exactly aligned with model training.

    RGB
    -> resize to 224x224
    -> VGG16 preprocess_input
    -> batch dimension
    """

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE,
        Image.Resampling.LANCZOS,
    )

    arr = np.array(
        image,
        dtype="float32",
    )

    arr = tf.keras.applications.vgg16.preprocess_input(
        arr
    )

    arr = np.expand_dims(
        arr,
        axis=0,
    )

    return arr


# ============================================================
# CLASS NAME FORMATTER
# ============================================================

def format_class_name(name: str) -> str:
    """Convert model class names to clean UI labels."""

    value = str(name).strip().lower()

    if value in {
        "notumor",
        "no_tumor",
        "no tumor",
        "not tumor",
    }:
        return "No Tumor"

    return value.replace("_", " ").title()


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    with st.sidebar:

        st.markdown(
            "# 🧠 BrainTumor AI"
        )

        st.caption(
            "MRI Classification System"
        )

        st.divider()

        st.markdown(
            "### About"
        )

        st.write(
            "This application uses a deep learning "
            "model to classify brain MRI images into "
            "four categories."
        )

        st.divider()

        st.markdown(
            "### Classes"
        )

        st.markdown(
            """
            🔵 **Glioma**

            🟣 **Meningioma**

            🟠 **Pituitary**

            🟢 **No Tumor**
            """
        )

        st.divider()

        st.markdown(
            "### Model"
        )

        st.write(
            "**Architecture:** VGG16 Transfer Learning"
        )

        st.write(
            "**Framework:** TensorFlow / Keras"
        )

        st.write(
            "**Input:** 224 × 224 RGB"
        )

        st.divider()

        st.caption(
            "For educational and research purposes only."
        )


# ============================================================
# HERO / HEADER
# ============================================================

def render_header():

    st.title(
        "🧠 Brain Tumor MRI Classifier"
    )

    st.caption(
        "AI-powered MRI image classification"
    )


# ============================================================
# WARNING
# ============================================================

def render_warning():

    st.warning(
        "⚠️ Educational/research use only. "
        "This model is not validated for clinical diagnosis. "
        "Always consult a qualified medical professional "
        "for an actual diagnosis."
    )


# ============================================================
# UPLOAD SECTION
# ============================================================

def render_upload():

    st.subheader(
        "📤 Upload MRI Scan"
    )

    st.caption(
        "Upload a brain MRI image in JPG, JPEG, or PNG format."
    )

    return st.file_uploader(
        "Drag and drop an MRI image here",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        help="Supported formats: JPG, JPEG, PNG",
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_image(model, image):
    """
    Run prediction while keeping the original ML pipeline.
    """

    input_array = preprocess(
        image
    )

    predictions = model.predict(
        input_array,
        verbose=0,
    )[0]

    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )

    return predictions


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    render_sidebar()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    render_header()

    # --------------------------------------------------------
    # WARNING
    # --------------------------------------------------------

    render_warning()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Loading trained VGG16 model..."
        ):
            model, class_names = load_model()

    except Exception as error:

        st.error(
            "Could not load the trained model."
        )

        st.write(
            "Make sure these files exist next to app.py:"
        )

        st.write(
            "- brain_tumor_model.keras"
        )

        st.write(
            "- class_names.json"
        )

        st.exception(error)

        return

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    uploaded_file = render_upload()

    # --------------------------------------------------------
    # EMPTY STATE
    # --------------------------------------------------------

    if uploaded_file is None:

        st.info(
            "👆 Upload an MRI image above to begin."
        )

        st.divider()

        st.subheader(
            "How it works"
        )

        step1, step2, step3 = st.columns(3)

        with step1:

            st.metric(
                "Step 1",
                "Upload",
                "MRI image",
            )

        with step2:

            st.metric(
                "Step 2",
                "Analyze",
                "VGG16 model",
            )

        with step3:

            st.metric(
                "Step 3",
                "Predict",
                "4 classes",
            )

        st.markdown(
            """
            <div class="footer-text">
                Brain Tumor MRI Classifier ·
                VGG16 Transfer Learning ·
                TensorFlow / Keras · Streamlit
            </div>
            """,
            unsafe_allow_html=True,
        )

        return

    # --------------------------------------------------------
    # OPEN IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        )

        image.load()

        image = image.convert("RGB")

    except Exception as error:

        st.error(
            "Unable to open the uploaded image."
        )

        st.exception(error)

        return

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    with st.spinner(
        "🧠 Analyzing MRI image..."
    ):

        try:

            predictions = predict_image(
                model,
                image,
            )

        except Exception as error:

            st.error(
                "Prediction failed."
            )

            st.exception(error)

            return

    # --------------------------------------------------------
    # VALIDATE OUTPUT
    # --------------------------------------------------------

    if predictions.ndim != 1:

        st.error(
            f"Unexpected model output shape: "
            f"{predictions.shape}"
        )

        return

    if len(predictions) != len(class_names):

        st.error(
            "Model outputs and class_names.json "
            "do not contain the same number of classes."
        )

        st.write(
            f"Model outputs: {len(predictions)}"
        )

        st.write(
            f"Class names: {len(class_names)}"
        )

        return

    # --------------------------------------------------------
    # PROBABILITY NORMALIZATION
    # --------------------------------------------------------

    prediction_sum = float(
        np.sum(predictions)
    )

    if (
        np.any(predictions < 0)
        or np.any(predictions > 1)
        or not np.isclose(
            prediction_sum,
            1.0,
            atol=0.01,
        )
    ):

        predictions = tf.nn.softmax(
            predictions
        ).numpy()

    # --------------------------------------------------------
    # TOP RESULT
    # --------------------------------------------------------

    top_index = int(
        np.argmax(predictions)
    )

    raw_label = class_names[
        top_index
    ]

    display_label = format_class_name(
        raw_label
    )

    confidence = float(
        predictions[top_index]
    ) * 100

    # --------------------------------------------------------
    # ANALYSIS RESULT
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🔍 Analysis Result"
    )

    image_col, result_col = st.columns(
        [1.05, 1],
        gap="large",
    )

    # ========================================================
    # LEFT: MRI IMAGE
    # ========================================================

    with image_col:

        st.markdown(
            "### 🖼️ Uploaded MRI"
        )

        st.image(
            image,
            caption=uploaded_file.name,
            width="stretch",
        )

        st.caption(
            f"Original image size: "
            f"{image.width} × {image.height}px"
        )

    # ========================================================
    # RIGHT: PREDICTION
    # ========================================================

    with result_col:

        st.markdown(
            "### 🎯 Prediction"
        )

        if display_label == "No Tumor":

            st.success(
                f"**No Tumor Detected**\n\n"
                f"Confidence: **{confidence:.2f}%**"
            )

        else:

            st.error(
                f"**{display_label}**\n\n"
                f"Confidence: **{confidence:.2f}%**"
            )

        st.progress(
            min(
                max(
                    confidence / 100,
                    0.0,
                ),
                1.0,
            )
        )

        st.caption(
            f"Model confidence: {confidence:.2f}%"
        )

        st.markdown(
            "### 📊 Class Probabilities"
        )

        probability_dict = {}

        for index, name in enumerate(
            class_names
        ):

            clean_name = format_class_name(
                name
            )

            probability_dict[
                clean_name
            ] = float(
                predictions[index]
            )

        st.bar_chart(
            probability_dict,
            height=250,
        )

    # --------------------------------------------------------
    # DETAILED PROBABILITY TABLE
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📈 Detailed Probabilities"
    )

    probability_cols = st.columns(
        len(class_names)
    )

    for index, name in enumerate(
        class_names
    ):

        probability = (
            float(predictions[index])
            * 100
        )

        with probability_cols[index]:

            st.metric(
                format_class_name(name),
                f"{probability:.2f}%",
            )

    # --------------------------------------------------------
    # MODEL DETAILS
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "🤖 Model Information"
    )

    info1, info2, info3 = st.columns(3)

    with info1:

        st.metric(
            "Architecture",
            "VGG16",
            "Transfer Learning",
        )

    with info2:

        st.metric(
            "Input",
            "224 × 224",
            "RGB",
        )

    with info3:

        st.metric(
            "Classes",
            str(len(class_names)),
            "Multiclass",
        )

    # --------------------------------------------------------
    # HOW IT WORKS
    # --------------------------------------------------------

    with st.expander(
        "🔬 How does the prediction work?"
    ):

        st.write(
            """
            1. The uploaded MRI image is converted to RGB.
            
            2. The image is resized to 224 × 224 pixels.
            
            3. VGG16 preprocessing is applied, matching
               the preprocessing used during model training.
            
            4. The trained VGG16 transfer-learning model
               generates class probabilities.
            
            5. The class with the highest probability is
               displayed as the prediction.
            """
        )

    # --------------------------------------------------------
    # MEDICAL DISCLAIMER
    # --------------------------------------------------------

    st.warning(
        "⚠️ Important: This application is an educational "
        "and research demonstration. Its predictions are "
        "not a medical diagnosis and should not replace "
        "professional medical evaluation."
    )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="footer-text">
            Brain Tumor MRI Classifier ·
            VGG16 Transfer Learning ·
            TensorFlow / Keras · Streamlit
            <br>
            Educational and research demonstration only.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()