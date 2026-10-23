import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tempfile, os, base64
from fpdf import FPDF

# --------------------------------------
# PAGE CONFIG
# --------------------------------------
st.set_page_config(page_title="AI FNAC Breast Cancer Detection", layout="wide")

# --------------------------------------
# PREMIUM PURPLE-BLUE LIGHT GLASS UI
# --------------------------------------
glass_ui_c = """
<style>
.stApp {
  background: linear-gradient(160deg, #f3f6ff 0%, #eef2ff 50%, #f7f9ff 100%);
  min-height: 100vh;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial;
}

/* Header */
.header h1{
  color:#4f46e5; font-size:42px; text-align:center; margin-top:18px; font-weight:800;
}
.header p{ text-align:center; color:#000000 !important; font-size:18px; }

/* Glass Card */
.glass-card{ background:rgba(255,255,255,.88); border-radius:16px; padding:26px; margin:18px;
  box-shadow:0 10px 30px rgba(15,23,42,0.06); backdrop-filter:blur(8px); color:#000000 !important;
}

/* Upload box */
.upload-box{background:rgba(255,255,255,.96); padding:14px; border-radius:10px; color !important;}
div[data-testid="stFileUploader"] { color:#000 !important; }

/* Prediction */
.prediction-box{ background:rgba(255,255,255,.9); padding:20px; margin-top:16px; border-radius:12px;
 text-align:center; font-size:20px; font-weight:700; border:1px solid rgba(79,70,229,.15); color:#4f46e5;}

/* Buttons */
button[data-testid="baseButton-secondary"]{
  background:linear-gradient(90deg,#4f46e5,#7c3aed) !important; color:black !important; font-weight:600;
}

/* Dashboard Text Black */
.dashboard-text, .dashboard-text p, .dashboard-text h3, .dashboard-text strong, .dashboard-text span {
  color: #000000 !important;
}
/* Image caption text color black */
.css-1i9za0i, .e1ewe7hr3, figcaption {
    color: #000000 !important;
    font-weight: 600 !important;
}
/* Change st.metric label color */
div[data-testid="metric-container"] label {
    color: #1e3a8a !important;   /* Dark blue */
    font-weight: 700 !important;
}

/* Change st.metric value color */
div[data-testid="metric-container"] div {
    color: #7c3aed !important;   /* Purple */
    font-size: 22px !important;
    font-weight: 800 !important;
}
/* ===== FIXED st.metric COLORS (Latest Streamlit) ===== */

/* Metric label (Model Type, Status, Explainability) */
div[data-testid="stMetricLabel"] {
    color: #1e3a8a !important;   /* Dark Blue */
    font-weight: 700 !important;
    font-size: 16px !important;
}

/* Metric value (CNN-FNAC, Active, Grad-CAM Enabled) */
div[data-testid="stMetricValue"] {
    color: #7c3aed !important;   /* Purple */
    font-weight: 800 !important;
    font-size: 22px !important;
}
/* ===== FORCE COLOR FOR st.metric LABELS ===== */
div[data-testid="stMetric"] > label,
div[data-testid="stMetricLabel"],
div[data-testid="stMetric"] label {
    color: #dc2626 !important;   /* 🔴 Red (change as you like) */
    font-weight: 800 !important;
    font-size: 17px !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ===== FORCE COLOR FOR st.metric VALUES ===== */
div[data-testid="stMetricValue"] {
    color: #16a34a !important;   /* 🟢 Green */
    font-weight: 900 !important;
    font-size: 23px !important;
}


</style>
"""
st.markdown(glass_ui_c, unsafe_allow_html=True)

# --------------------------------------
# SIDEBAR MENU
# --------------------------------------
st.sidebar.title("📍 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Dashboard", "🧠 AI Model"])

# --------------------------------------
# MODEL LOADING
# --------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("breast_cancer_fnac_model.h5")

try:
    model = load_model()
except:
    st.error("⚠ Model not found. Place 'breast_cancer_fnac_model.h5' near app.py.")
    st.stop()

# --------------------------------------
# GRAD-CAM FUNCTION
# --------------------------------------
def make_gradcam_heatmap(img_array, model, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = np.maximum(tf.squeeze(conv_outputs[0] @ pooled_grads[..., tf.newaxis]), 0)
    if heatmap.max() != 0: heatmap /= heatmap.max()
    return heatmap

# --------------------------------------
# DASHBOARD PAGE
# --------------------------------------
if page == "🏠 Dashboard":
    st.markdown("<div class='header'><h1>📊 FNAC AI Dashboard</h1>"
                "<p>Welcome to the Smart Diagnostics Dashboard</p></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-card dashboard-text'>
        <h3>🌟 Dashboard Overview</h3>
        <p>This AI platform detects breast cancer from FNAC microscopic images and highlights malignant regions using <strong>Explainable AI Grad-CAM</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type", "CNN - FNAC")
    col2.metric("Status", "Active")
    col3.metric("Explainability", "Grad-CAM Enabled")

    st.info("➡ Switch to **AI Model** tab from the left to upload and test images.")

# --------------------------------------
# MODEL PAGE
# --------------------------------------
if page == "🧠 AI Model":

    st.markdown("<div class='header'><h1>🧠 FNAC Breast Cancer Prediction</h1></div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'><div class='upload-box'><strong>📤 Upload FNAC Image</strong></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp.write(uploaded_file.read()); temp.flush()

        img = tf.keras.preprocessing.image.load_img(temp.name, target_size=(224, 224))
        x = np.expand_dims(tf.keras.preprocessing.image.img_to_array(img), 0) / 255.0

        preds = model.predict(x)
        malignant_prob = float(preds[0][0]) if preds.shape[-1] == 1 else float(preds[0][1])
        benign_prob = 1.0 - malignant_prob
        result = "Malignant" if malignant_prob > 0.5 else "Benign"
        conf = max(malignant_prob, benign_prob) * 100

        st.markdown(f"<div class='prediction-box'>{result}<br><span style='font-size:14px'>{conf:.2f}% Confidence</span></div>", unsafe_allow_html=True)

        # GRAD CAM
        conv_layers = [l.name for l in model.layers if 'conv' in l.name.lower()]
        heatmap = make_gradcam_heatmap(x, model, conv_layers[-1])

        orig = cv2.cvtColor(cv2.imread(temp.name), cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, (224, 224))
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        final = cv2.addWeighted(orig, 0.65, heatmap, 0.45, 0)

        col1, col2 = st.columns(2)

        with col1:
            st.image(orig, use_container_width=True)
            st.markdown("<p style='text-align:center; color:#000; font-size:18px; font-weight:700;'>Original FNAC Image</p>", unsafe_allow_html=True)

        with col2:
            st.image(final, use_container_width=True)
            st.markdown("<p style='text-align:center; color:#000; font-size:18px; font-weight:700;'>🔥 Grad-CAM Heatmap</p>", unsafe_allow_html=True)



        # REPORT GENERATION
        if st.button("📄 Generate Diagnosis Report"):
            pdf = FPDF()
            pdf.add_page(); pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt="AI FNAC Breast Cancer Diagnostic Report", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"Diagnosis Result: {result}", ln=True)
            pdf.cell(200, 10, txt=f"Confidence: {conf:.2f}%", ln=True)
            pdf.ln(10)
            final_img_path = "gradcam_result.png"
            cv2.imwrite(final_img_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
            pdf.image(final_img_path, x=10, y=60, w=180)
            report_path = "FNAC_Report.pdf"
            pdf.output(report_path)

            with open(report_path, "rb") as f:
                st.download_button("⬇ Download Report PDF", f, file_name="FNAC_Report.pdf")

        os.remove(temp.name)
