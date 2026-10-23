import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from fpdf import FPDF
from PIL import Image

st.set_page_config(page_title="FNAC AI Screening", page_icon="🔬", layout="wide")

# ---- Edit these with the real numbers from your notebook ----
MODEL_STATS = {"Accuracy": "~91%", "Input size": "224 x 224", "Classes": "Benign / Malignant"}

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 1100px;}
.hero {background: linear-gradient(135deg,#4f46e5,#7c3aed); border-radius: 18px;
       padding: 34px 30px; color: #fff; margin-bottom: 20px;}
.hero h1 {margin: 0; font-size: 34px; color: #fff;}
.hero p {margin: 8px 0 0; font-size: 16px; opacity: .92; color: #fff;}
.card {background: #fff; border-radius: 14px; padding: 20px;
       box-shadow: 0 6px 22px rgba(15,23,42,.07); border: 1px solid #e8ebf7;}
.result {border-radius: 14px; padding: 22px; text-align: center; border: 2px solid;}
.result h2 {margin: 0; font-size: 30px;}
.benign {background: #ecfdf5; border-color: #16a34a; color: #166534;}
.malignant {background: #fef2f2; border-color: #dc2626; color: #991b1b;}
.disclaimer {background: #fffbeb; border-left: 4px solid #f59e0b; padding: 12px 16px;
             border-radius: 8px; font-size: 14px; color: #78350f;}
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg,#4f46e5,#7c3aed); color: #fff; border: 0;
    border-radius: 10px; padding: 10px 22px; font-weight: 600;}
.stButton>button:hover, .stDownloadButton>button:hover {color: #fff; opacity: .92;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("breast_cancer_fnac_model.h5")


try:
    model = load_model()
except Exception as e:
    st.error("Model file 'breast_cancer_fnac_model.h5' could not be loaded.")
    st.caption(str(e))
    st.stop()


def preprocess(pil_img):
    # NEAREST matches Keras load_img's default, so predictions stay identical to training-time loading
    img = pil_img.convert("RGB").resize((224, 224), Image.NEAREST)
    arr = np.array(img, dtype="float32")
    return img, np.expand_dims(arr, 0) / 255.0


def make_gradcam_heatmap(img_array, model, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = np.maximum(tf.squeeze(conv_outputs[0] @ pooled[..., tf.newaxis]), 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()
    return heatmap


def overlay_gradcam(orig_rgb, x):
    conv_layers = [l.name for l in model.layers if "conv" in l.name.lower()]
    heatmap = make_gradcam_heatmap(x, model, conv_layers[-1])
    heatmap = cv2.resize(np.array(heatmap), (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(orig_rgb, 0.65, heatmap, 0.45, 0)


def build_pdf(result, conf, final_rgb):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=15)
    pdf.cell(190, 10, txt="AI FNAC Screening Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(6)
    pdf.cell(190, 9, txt=f"AI prediction: {result}", ln=True)
    pdf.cell(190, 9, txt=f"Model confidence: {conf:.2f}%", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(190, 6, txt="Research prototype. Not a medical diagnosis. "
                               "Consult a qualified pathologist.")
    with tempfile.TemporaryDirectory() as d:
        img_path, pdf_path = os.path.join(d, "g.png"), os.path.join(d, "r.pdf")
        cv2.imwrite(img_path, cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))
        pdf.image(img_path, x=30, y=75, w=150)
        pdf.output(pdf_path)
        with open(pdf_path, "rb") as f:
            return f.read()


st.markdown("""
<div class="hero">
  <h1>🔬 FNAC AI Screening</h1>
  <p>Upload a cytology image. A CNN predicts benign or malignant, and Grad-CAM shows
  which regions influenced the decision.</p>
</div>
""", unsafe_allow_html=True)

tab_analyze, tab_about = st.tabs(["🧠 Analyze image", "ℹ️ About the model"])

with tab_analyze:
    st.markdown(
        "<div class='disclaimer'><b>Research prototype.</b> Not a medical device and not a "
        "diagnosis. Results must be reviewed by a qualified pathologist.</div>",
        unsafe_allow_html=True)
    st.write("")

    uploaded = st.file_uploader("Upload an FNAC image (JPG or PNG)", type=["jpg", "jpeg", "png"])

    if uploaded:
        try:
            pil_img = Image.open(uploaded)
            orig_img, x = preprocess(pil_img)
        except Exception:
            st.error("That file could not be read as an image. Please upload a valid JPG or PNG.")
            st.stop()

        with st.spinner("Analyzing image..."):
            preds = model.predict(x, verbose=0)
            malignant = float(preds[0][0]) if preds.shape[-1] == 1 else float(preds[0][1])
            benign = 1.0 - malignant
            result = "Malignant" if malignant > 0.5 else "Benign"
            conf = max(malignant, benign) * 100
            orig_rgb = np.array(orig_img)
            try:
                final = overlay_gradcam(orig_rgb, x)
            except Exception:
                final = None

        left, right = st.columns([1, 1.3], gap="large")
        with left:
            css = "malignant" if result == "Malignant" else "benign"
            st.markdown(
                f"<div class='result {css}'><div style='font-size:13px;letter-spacing:1px'>"
                f"AI PREDICTION</div><h2>{result}</h2>"
                f"<div style='font-size:15px'>{conf:.1f}% confidence</div></div>",
                unsafe_allow_html=True)
            st.write("")
            st.caption("Class probabilities")
            st.progress(benign, text=f"Benign: {benign * 100:.1f}%")
            st.progress(malignant, text=f"Malignant: {malignant * 100:.1f}%")

        with right:
            c1, c2 = st.columns(2)
            c1.image(orig_rgb, caption="Original image", use_container_width=True)
            if final is not None:
                c2.image(final, caption="Grad-CAM heatmap", use_container_width=True)
            else:
                c2.info("Heatmap unavailable for this model.")

        if final is not None:
            st.write("")
            st.download_button("⬇ Download screening report (PDF)",
                               data=build_pdf(result, conf, final),
                               file_name="FNAC_Report.pdf", mime="application/pdf")
    else:
        st.info("Upload an image above to see the prediction and heatmap.")

with tab_about:
    cols = st.columns(len(MODEL_STATS))
    for col, (k, v) in zip(cols, MODEL_STATS.items()):
        col.metric(k, v)
    st.markdown("""
**How it works**
1. The image is resized to 224 x 224 and normalized.
2. A convolutional neural network (CNN) outputs the probability of malignancy.
3. Grad-CAM highlights the regions that most influenced the prediction.

**Limitations**
- Trained on a limited dataset; performance on images from other labs or stains may differ.
- Intended for learning and research, not clinical use.
""")
