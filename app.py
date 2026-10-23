import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from fpdf import FPDF
from PIL import Image

st.set_page_config(page_title="FNAC AI Screening", page_icon="🔬", layout="wide")

# ---- Replace with the real numbers from your notebook ----
STATS = [("🎯", "Accuracy", "~91%"), ("🖼️", "Input size", "224×224"),
         ("🧬", "Classes", "2"), ("🔥", "Explainability", "Grad-CAM")]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {font-family: 'Inter', sans-serif;}
#MainMenu, footer, header {visibility: hidden;}
.stApp {background: radial-gradient(circle at 10% 0%, #eef2ff 0%, #f8faff 45%, #f5f3ff 100%);}
.block-container {padding-top: 1.2rem; max-width: 1150px;}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {background: linear-gradient(180deg,#1e1b4b 0%,#312e81 55%,#4c1d95 100%);}
[data-testid="stSidebar"] * {color: #e0e7ff !important;}
[data-testid="stSidebar"] [role="radiogroup"] label {
  background: rgba(255,255,255,.07); border-radius: 12px; padding: 10px 14px; margin-bottom: 8px;
  transition: all .25s ease; border: 1px solid rgba(255,255,255,.08);}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: rgba(255,255,255,.18); transform: translateX(4px);}
.brand {font-size: 22px; font-weight: 800; color: #fff !important; margin-bottom: 2px;}
.brand-sub {font-size: 12px; opacity: .75; margin-bottom: 18px; letter-spacing: .5px;}

/* ---------- Animations ---------- */
@keyframes fadeUp {from {opacity:0; transform: translateY(22px);} to {opacity:1; transform: none;}}
@keyframes gradientMove {0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;}}
@keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(34,197,94,.7);} 70% {box-shadow: 0 0 0 10px rgba(34,197,94,0);} 100% {box-shadow: 0 0 0 0 rgba(34,197,94,0);}}
@keyframes float {0%,100% {transform: translateY(0);} 50% {transform: translateY(-8px);}}
@keyframes fillBar {from {width: 0;}}
@keyframes popIn {0% {opacity:0; transform: scale(.85);} 70% {transform: scale(1.04);} 100% {opacity:1; transform: scale(1);}}

/* ---------- Hero ---------- */
.hero {position: relative; overflow: hidden; border-radius: 22px; padding: 44px 38px; color: #fff;
  background: linear-gradient(120deg,#4f46e5,#7c3aed,#2563eb,#7c3aed); background-size: 300% 300%;
  animation: gradientMove 12s ease infinite, fadeUp .7s ease both;
  box-shadow: 0 20px 45px rgba(79,70,229,.30); margin-bottom: 26px;}
.hero::after {content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px;
  background: radial-gradient(circle, rgba(255,255,255,.28), transparent 70%); border-radius:50%;
  animation: float 6s ease-in-out infinite;}
.hero h1 {margin: 10px 0 8px; font-size: 40px; font-weight: 800; color:#fff; letter-spacing: -.5px;}
.hero p {margin: 0; font-size: 17px; max-width: 640px; opacity: .93; color:#fff; line-height: 1.6;}
.pill {display:inline-flex; align-items:center; gap:8px; background: rgba(255,255,255,.18);
  border: 1px solid rgba(255,255,255,.3); padding: 6px 14px; border-radius: 999px; font-size: 13px; font-weight: 600;}
.dot {width: 10px; height: 10px; border-radius: 50%; background: #22c55e; animation: pulse 1.8s infinite;}

/* ---------- Tags ---------- */
.tags {margin-top: 18px;}
.tag {display:inline-block; padding: 5px 13px; margin: 0 8px 8px 0; border-radius: 999px; font-size: 12.5px;
  font-weight: 600; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28); color:#fff;
  transition: all .25s ease;}
.tag:hover {background: rgba(255,255,255,.32); transform: translateY(-2px);}
.tag.dark {background:#eef2ff; border-color:#c7d2fe; color:#4338ca;}
.tag.dark:hover {background:#e0e7ff;}

/* ---------- Cards ---------- */
.section-title {font-size: 22px; font-weight: 800; color:#1e1b4b; margin: 30px 0 14px;}
.stat-grid {display:grid; grid-template-columns: repeat(4, 1fr); gap: 16px;}
.stat {background:#fff; border-radius:18px; padding: 22px 18px; border:1px solid #e5e7fb; text-align:center;
  box-shadow: 0 8px 24px rgba(15,23,42,.06); transition: all .3s ease; animation: fadeUp .7s ease both;}
.stat:nth-child(2) {animation-delay:.1s;} .stat:nth-child(3) {animation-delay:.2s;} .stat:nth-child(4) {animation-delay:.3s;}
.stat:hover {transform: translateY(-8px); box-shadow: 0 18px 38px rgba(79,70,229,.22); border-color:#a5b4fc;}
.stat .ic {font-size: 30px;}
.stat .val {font-size: 26px; font-weight: 800; margin: 6px 0 2px;
  background: linear-gradient(90deg,#4f46e5,#7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.stat .lbl {font-size: 13px; color:#64748b; font-weight: 600; text-transform: uppercase; letter-spacing: .6px;}

.step-grid {display:grid; grid-template-columns: repeat(4, 1fr); gap: 16px;}
.step {position:relative; background:#fff; border-radius:18px; padding: 24px 20px; border:1px solid #e5e7fb;
  box-shadow: 0 8px 24px rgba(15,23,42,.05); transition: all .3s ease; animation: fadeUp .8s ease both;}
.step:hover {transform: translateY(-6px); box-shadow: 0 16px 34px rgba(124,58,237,.2);}
.step .num {width: 38px; height: 38px; border-radius: 12px; display:flex; align-items:center; justify-content:center;
  color:#fff; font-weight: 800; background: linear-gradient(135deg,#4f46e5,#7c3aed); margin-bottom: 12px;}
.step h4 {margin: 0 0 6px; color:#1e1b4b; font-size: 16px;}
.step p {margin: 0; color:#475569; font-size: 14px; line-height: 1.55;}

.card {background:#fff; border-radius:18px; padding: 22px; border:1px solid #e5e7fb;
  box-shadow: 0 8px 24px rgba(15,23,42,.06); animation: fadeUp .6s ease both;}
.disclaimer {background:#fffbeb; border-left: 5px solid #f59e0b; padding: 14px 18px; border-radius: 12px;
  font-size: 14px; color:#78350f; margin-bottom: 16px;}

/* ---------- Result ---------- */
.result {border-radius: 20px; padding: 28px 22px; text-align:center; color:#fff; animation: popIn .6s ease both;}
.result.benign {background: linear-gradient(135deg,#16a34a,#22c55e); box-shadow: 0 16px 36px rgba(22,163,74,.35);}
.result.malignant {background: linear-gradient(135deg,#dc2626,#f43f5e); box-shadow: 0 16px 36px rgba(220,38,38,.35);}
.result .small {font-size: 12px; letter-spacing: 2px; opacity: .9; font-weight: 700;}
.result h2 {margin: 6px 0; font-size: 38px; font-weight: 800; color:#fff;}
.result .conf {font-size: 16px; font-weight: 600;}

.bar-wrap {margin-top: 18px;}
.bar-label {display:flex; justify-content:space-between; font-size: 13.5px; font-weight: 600; color:#334155; margin-bottom: 6px;}
.bar {height: 12px; background:#e2e8f0; border-radius: 999px; overflow:hidden; margin-bottom: 14px;}
.bar > span {display:block; height:100%; border-radius: 999px; animation: fillBar 1.2s ease both;}
.bar .g {background: linear-gradient(90deg,#16a34a,#4ade80);}
.bar .r {background: linear-gradient(90deg,#dc2626,#fb7185);}

/* ---------- Streamlit widgets ---------- */
[data-testid="stFileUploader"] section {background: #fff; border: 2px dashed #a5b4fc; border-radius: 18px;
  padding: 26px; transition: all .3s ease;}
[data-testid="stFileUploader"] section:hover {border-color:#7c3aed; background:#f5f3ff; transform: scale(1.01);}
[data-testid="stImage"] img {border-radius: 16px; box-shadow: 0 10px 26px rgba(15,23,42,.15); transition: transform .35s ease;}
[data-testid="stImage"] img:hover {transform: scale(1.03);}
[data-testid="stImageCaption"] {font-weight: 600; color:#334155;}
.stButton>button, .stDownloadButton>button {background: linear-gradient(90deg,#4f46e5,#7c3aed); color:#fff; border:0;
  border-radius: 12px; padding: 12px 26px; font-weight: 700; transition: all .25s ease;
  box-shadow: 0 8px 20px rgba(79,70,229,.35);}
.stButton>button:hover, .stDownloadButton>button:hover {transform: translateY(-3px); color:#fff;
  box-shadow: 0 14px 28px rgba(79,70,229,.45);}
.footer {text-align:center; color:#94a3b8; font-size: 13px; margin: 40px 0 10px;}

@media (max-width: 900px) {.stat-grid, .step-grid {grid-template-columns: repeat(2, 1fr);} .hero h1 {font-size: 30px;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(s):
    """Render HTML; lines are stripped so Markdown doesn't treat indentation as code."""
    st.markdown("".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


# ================= MODEL (same as your original code) =================
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
    # Nearest-neighbour resize matches Keras load_img's default, so predictions stay identical
    img = pil_img.convert("RGB").resize((224, 224), Image.NEAREST)
    return img, np.expand_dims(np.array(img, dtype="float32"), 0) / 255.0


# ================= GRAD-CAM (your original working logic) =================
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


# ================= PDF =================
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
    pdf.multi_cell(190, 6, txt="Research prototype. Not a medical diagnosis. Consult a qualified pathologist.")
    with tempfile.TemporaryDirectory() as d:
        ip, pp = os.path.join(d, "g.png"), os.path.join(d, "r.pdf")
        cv2.imwrite(ip, cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR))
        pdf.image(ip, x=30, y=75, w=150)
        pdf.output(pp)
        with open(pp, "rb") as f:
            return f.read()


# ================= SIDEBAR =================
with st.sidebar:
    html("<div class='brand'>🔬 FNAC AI</div><div class='brand-sub'>SMART CYTOLOGY SCREENING</div>")
    page = st.radio("Navigate", ["🏠  Dashboard", "🧠  AI Model"], label_visibility="collapsed")
    st.write("")
    st.caption("Research prototype. Not for clinical use.")

# ================= DASHBOARD =================
if page.startswith("🏠"):
    html("""
    <div class='hero'>
      <span class='pill'><span class='dot'></span>System online</span>
      <h1>AI-Powered FNAC Breast Cancer Screening</h1>
      <p>A deep-learning model analyses cytology images, predicts benign or malignant, and uses
      Explainable AI to show the regions behind every prediction.</p>
      <div class='tags'><span class='tag'>CNN</span><span class='tag'>TensorFlow</span>
      <span class='tag'>Keras</span><span class='tag'>OpenCV</span><span class='tag'>Grad-CAM</span>
      <span class='tag'>Streamlit</span></div>
    </div>""")

    cards = "".join(f"<div class='stat'><div class='ic'>{i}</div><div class='val'>{v}</div>"
                    f"<div class='lbl'>{l}</div></div>" for i, l, v in STATS)
    html(f"<div class='stat-grid'>{cards}</div>")

    html("<div class='section-title'>How it works</div>")
    steps = [("Upload", "Add an FNAC microscopy image in JPG or PNG format."),
             ("Preprocess", "The image is resized to 224×224 and normalised."),
             ("Predict", "The CNN outputs the probability of malignancy."),
             ("Explain", "Grad-CAM highlights the regions that drove the result.")]
    s = "".join(f"<div class='step'><div class='num'>{n}</div><h4>{t}</h4><p>{d}</p></div>"
                for n, (t, d) in enumerate(steps, 1))
    html(f"<div class='step-grid'>{s}</div>")

    html("<div class='section-title'>Tech stack</div>")
    html("<div>" + "".join(f"<span class='tag dark'>{t}</span>" for t in
         ["Python", "TensorFlow", "Keras", "OpenCV", "NumPy", "Pillow", "Streamlit", "FPDF"]) + "</div>")
    html("<div class='disclaimer' style='margin-top:26px'><b>Disclaimer:</b> built for learning and research. "
         "It is not a medical device and must not be used for diagnosis.</div>")

# ================= AI MODEL =================
else:
    html("""
    <div class='hero' style='padding:30px 34px'>
      <span class='pill'><span class='dot'></span>Model ready</span>
      <h1 style='font-size:32px'>Analyze an FNAC Image</h1>
      <p>Upload a cytology image to get a prediction and an explainable heatmap.</p>
    </div>""")
    html("<div class='disclaimer'><b>Research prototype.</b> Results are not a diagnosis and must be "
         "reviewed by a qualified pathologist.</div>")

    uploaded = st.file_uploader("Drag and drop an FNAC image here (JPG or PNG)", type=["jpg", "jpeg", "png"])

    if uploaded:
        try:
            orig_img, x = preprocess(Image.open(uploaded))
        except Exception:
            st.error("That file could not be read as an image. Please upload a valid JPG or PNG.")
            st.stop()

        gc_error = None
        with st.spinner("Analyzing image..."):
            preds = model.predict(x, verbose=0)
            mal = float(preds[0][0]) if preds.shape[-1] == 1 else float(preds[0][1])
            ben = 1.0 - mal
            result = "Malignant" if mal > 0.5 else "Benign"
            conf = max(mal, ben) * 100
            orig_rgb = np.array(orig_img)
            try:
                final = overlay_gradcam(orig_rgb, x)
            except Exception as e:
                final = None
                gc_error = repr(e)

        left, right = st.columns([1, 1.35], gap="large")
        with left:
            css = "malignant" if result == "Malignant" else "benign"
            icon = "⚠️" if result == "Malignant" else "✅"
            html(f"""
            <div class='result {css}'><div class='small'>AI PREDICTION</div>
            <h2>{icon} {result}</h2><div class='conf'>{conf:.1f}% confidence</div></div>
            <div class='card bar-wrap' style='margin-top:18px'>
              <div class='bar-label'><span>Benign</span><span>{ben*100:.1f}%</span></div>
              <div class='bar'><span class='g' style='width:{ben*100:.1f}%'></span></div>
              <div class='bar-label'><span>Malignant</span><span>{mal*100:.1f}%</span></div>
              <div class='bar'><span class='r' style='width:{mal*100:.1f}%'></span></div>
              <span class='tag dark'>224×224 input</span><span class='tag dark'>CNN</span>
              <span class='tag dark'>Grad-CAM</span>
            </div>""")
        with right:
            c1, c2 = st.columns(2)
            c1.image(orig_rgb, caption="Original image", use_container_width=True)
            if final is not None:
                c2.image(final, caption="🔥 Grad-CAM heatmap", use_container_width=True)
            else:
                c2.warning("Heatmap could not be generated.")
                with c2.expander("Technical details"):
                    st.code(gc_error)

        if final is not None:
            st.write("")
            st.download_button("⬇ Download screening report (PDF)", data=build_pdf(result, conf, final),
                               file_name="FNAC_Report.pdf", mime="application/pdf")
    else:
        st.info("Upload an image above to see the prediction and heatmap.")

html("<div class='footer'>Built by Laxmi Sonnad · Research prototype</div>")
