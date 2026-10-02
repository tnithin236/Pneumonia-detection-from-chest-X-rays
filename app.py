"""Pneumonia Detector: YOLO classifier + Grad-CAM, dark dashboard UI (Streamlit)."""
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from ultralytics import YOLO

MODEL_PATH = Path("C:\\Users\\Nithin T\\Desktop\\CV_Projects\\pneumonia-detector\\best.pt")
IMG_SIZE = 224          # model input size
DISP = 512              # display size
USER_NAME = "Nithin T"  # shown in the header chip
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁", layout="wide")

CSS = """
<style>
#MainMenu, footer {visibility:hidden;}
header[data-testid="stHeader"] {background:transparent;}
.stApp {background:radial-gradient(1200px 600px at 20% -10%, #0c1c45 0%, #060d1f 55%);}
.block-container {padding-top:1rem; max-width:1500px;}
section[data-testid="stSidebar"] {background:#08122b; border-right:1px solid #14284f;}
section[data-testid="stSidebar"] > div {padding-top:5.2rem;}

/* top bar */
.topbar {display:flex; align-items:center; justify-content:space-between;
  padding:6px 4px 14px; border-bottom:1px solid #14284f; margin-bottom:14px;}
.brand {display:flex; align-items:center; gap:14px;}
.brand .logo {font-size:2.4rem; line-height:1; filter:drop-shadow(0 0 10px #2f80ff88);}
.brand h1 {margin:0; font-size:1.7rem; font-weight:700; color:#fff; padding:0;}
.brand h1 span {color:#3b8cff;}
.brand p {margin:0; color:#8fa3c8; font-size:.85rem;}
.userchip {display:flex; align-items:center; gap:12px; color:#8fa3c8;}
.avatar {width:46px; height:46px; border-radius:50%; background:#1b5fd1;
  display:flex; align-items:center; justify-content:center; font-size:1.4rem;}
.userchip b {color:#fff; display:block; font-size:.95rem;}
.userchip small::before {content:"●"; color:#22d36b; margin-right:5px; font-size:.7rem;}

/* sidebar nav (radio styled as menu) */
section[data-testid="stSidebar"] [role="radiogroup"] {gap:6px;}
section[data-testid="stSidebar"] [role="radiogroup"] label {
  padding:12px 16px; border-radius:10px; width:100%; cursor:pointer; color:#b8c6e4;}
section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {display:none;}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#0f2147;}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background:linear-gradient(90deg,#123a8f,#1b5fd1); color:#fff; font-weight:600;}
.sidecard {margin-top:2.2rem; padding:16px; border:1px solid #1a3a73;
  border-radius:12px; background:#0a1735; color:#8fa3c8; font-size:.85rem;}
.sidecard b {color:#fff; font-size:.95rem;}

/* cards */
[class*="st-key-card_"] {background:#0a1735; border:1px solid #14284f !important;
  border-radius:16px; padding:6px 8px;}
.ch {display:flex; align-items:center; gap:12px; margin-bottom:10px;}
.ci {width:38px; height:38px; border-radius:10px; background:#17408f;
  display:flex; align-items:center; justify-content:center; font-size:1.1rem;}
.ct {font-weight:700; color:#fff; font-size:1.05rem; flex:1;}
.badge {background:#0f2a5c; color:#4fb0ff; font-size:.72rem; font-weight:600;
  padding:5px 12px; border-radius:20px;}
.chip {display:flex; justify-content:space-between; align-items:center; margin-top:10px;
  padding:10px 14px; border:1px solid #14284f; border-radius:10px; color:#cfdcf5; font-size:.85rem;}
.chip span {background:#0f2a5c; color:#4fb0ff; padding:4px 12px; border-radius:20px; font-weight:600;}
.cbar {height:10px; border-radius:6px; margin-top:12px;
  background:linear-gradient(90deg,#1c1cff,#00b4ff,#00e08a,#ffe600,#ff7a00,#ff1a1a);}
.clab {display:flex; justify-content:space-between; color:#8fa3c8; font-size:.78rem; margin-top:4px;}
.empty {height:380px; display:flex; align-items:center; justify-content:center;
  color:#5f7399; border:1px dashed #1c3566; border-radius:12px; text-align:center; padding:20px;}
.upl-title {color:#fff; font-weight:700; font-size:1.1rem;}
.upl-sub {color:#8fa3c8; font-size:.88rem;}

/* result card */
.result {border-radius:16px; padding:18px 20px; border:2px solid; margin-bottom:14px;}
.result.pos {border-color:#e0364f; background:linear-gradient(160deg,#2a0f20,#1a1030);}
.result.neg {border-color:#1fbf6b; background:linear-gradient(160deg,#0c2a22,#0a1735);}
.rhead {display:flex; justify-content:space-between; align-items:center; color:#fff; font-weight:700;}
.tag {font-size:.72rem; padding:5px 12px; border-radius:20px; font-weight:600;}
.pos .tag {background:#4a1626; color:#ff4d6a;} .neg .tag {background:#0e3a2c; color:#2de28b;}
.rmain {display:flex; align-items:center; gap:18px; margin:16px 0 10px;}
.rcircle {width:76px; height:76px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-size:2.2rem;}
.pos .rcircle {background:#e0364f;} .neg .rcircle {background:#1fbf6b;}
.rlabel {font-size:1.9rem; font-weight:800; letter-spacing:.5px;}
.pos .rlabel {color:#ff4d6a;} .neg .rlabel {color:#2de28b;}
.rconf {color:#cfdcf5;} .rconf b {color:#fff; font-size:1.5rem; margin-left:6px;}
.rdesc {color:#9fb0d3; font-size:.88rem;}

/* probabilities */
.prow {display:flex; justify-content:space-between; color:#fff; margin-top:10px; font-size:.95rem;}
.track {height:10px; background:#1b2a4d; border-radius:6px; margin-top:6px; overflow:hidden;}
.fill {height:100%; border-radius:6px;}
.note {display:flex; gap:12px; margin-top:16px; padding:14px; border:1px solid #1a3a73;
  border-radius:10px; background:#0b1c40; color:#9fb0d3; font-size:.82rem;}

/* uploader */
[data-testid="stFileUploader"] section {background:#0a1735; border:1.5px dashed #1f4a8a;
  border-radius:12px;}
[data-testid="stFileUploader"] button {background:#17408f; color:#fff; border:none;}
.stButton > button {background:#17408f; color:#fff; border:none; border-radius:8px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------- model ----------------
class Wrap(nn.Module):
    """YOLO's eval head returns (probs, logits); always return logits only."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        o = self.m(x)
        return o[1] if isinstance(o, (tuple, list)) else o


@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    net = YOLO(path).model.to(DEVICE).float().eval()
    for p in net.parameters():
        p.requires_grad = True
    wnet = Wrap(net).to(DEVICE).eval()
    return wnet, GradCAM(model=wnet, target_layers=[net.model[-2]])


@st.cache_data(show_spinner="Analysing X-ray...")
def analyze(file_bytes: bytes, alpha: float, _wnet, _cam):
    import io
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    small = np.asarray(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    x = torch.from_numpy(small).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = float(torch.softmax(_wnet(x), dim=1)[0, 1].cpu())
    with torch.enable_grad():
        heat = _cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
    disp = np.asarray(img.resize((DISP, DISP))).astype(np.float32) / 255.0
    heat = cv2.resize(heat, (DISP, DISP))
    overlay = show_cam_on_image(disp, heat, use_rgb=True, image_weight=1 - alpha)
    return disp, overlay, p


# ---------------- ui helpers ----------------
def header(icon, title, badge):
    st.markdown(
        f'<div class="ch"><div class="ci">{icon}</div><div class="ct">{title}</div>'
        f'<div class="badge">{badge}</div></div>', unsafe_allow_html=True)


def chip(text, size):
    st.markdown(f'<div class="chip">{text}<span>{size}</span></div>', unsafe_allow_html=True)


def result_card(p, thr):
    pos = p >= thr
    conf = p if pos else 1 - p
    desc = ("AI model detected signs of pneumonia in the chest X-ray image. "
            "Follow-up with a doctor is advised." if pos else
            "No signs of pneumonia were detected by the model. If symptoms persist, "
            "consult a doctor.")
    if pos and p < 0.5:
        desc += " (Flagged by the high-recall threshold.)"
    st.markdown(f"""
    <div class="result {'pos' if pos else 'neg'}">
      <div class="rhead"><span>🧠&nbsp; Classification Result</span><span class="tag">AI Prediction</span></div>
      <div class="rmain"><div class="rcircle">🫁</div>
        <div><div class="rlabel">{'PNEUMONIA' if pos else 'NORMAL'}</div>
        <div class="rconf">Confidence: <b>{conf:.1%}</b></div></div></div>
      <div class="rdesc">{desc}</div></div>""", unsafe_allow_html=True)


def prob_card(p):
    st.markdown(f"""
    <div class="ch"><div class="ci">📊</div><div class="ct">Class Probabilities</div></div>
    <div class="prow"><span>Pneumonia</span><span style="color:#ff4d6a;font-weight:700">{p:.1%}</span></div>
    <div class="track"><div class="fill" style="width:{p*100:.1f}%;background:#ff4d6a"></div></div>
    <div class="prow"><span>Normal</span><span style="color:#2de28b;font-weight:700">{1-p:.1%}</span></div>
    <div class="track"><div class="fill" style="width:{(1-p)*100:.1f}%;background:#2de28b"></div></div>
    <div class="note"><span>ℹ️</span><span>This prediction is based on an AI model and should
    not be used as a substitute for professional medical advice.</span></div>""",
                unsafe_allow_html=True)


# ---------------- state ----------------
st.session_state.setdefault("thr", 0.50)
st.session_state.setdefault("alpha", 0.50)
st.session_state.setdefault("history", [])
st.session_state.setdefault("seen", None)

# ---------------- top bar + sidebar ----------------
st.markdown(f"""
<div class="topbar">
  <div class="brand"><div class="logo">🫁</div>
    <div><h1>Pneumonia <span>Detector</span></h1><p>AI Powered Chest X-Ray Analysis</p></div></div>
  <div class="userchip"><span style="font-size:1.3rem">🌙</span>
    <div class="avatar">👤</div><div><b>{USER_NAME}</b><small>Online</small></div></div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    page = st.radio("nav", ["🏠  Home", "☁️  Upload Image", "🕒  History", "ℹ️  About", "⚙️  Settings"],
                    label_visibility="collapsed")
    st.markdown("""<div class="sidecard"><b>🛡️ AI for Better Healthcare</b><br><br>
    Early detection saves lives. This tool helps in identifying pneumonia from chest
    X-ray images using deep learning.</div>""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error(f"Model not found at `{MODEL_PATH}`. Copy your trained `best.pt` into `models/`.")
    st.stop()
wnet, cam = load_model(str(MODEL_PATH))

# ---------------- pages ----------------
if page.endswith(("Home", "Upload Image")):
    with st.container(border=True, key="card_upload"):
        left, right = st.columns([1, 1.8], vertical_alignment="center")
        left.markdown('<div class="upl-title">🖼️&nbsp; Upload Chest X-Ray Image</div>'
                      '<div class="upl-sub">Select an image file (JPG, PNG) or drag and drop it here</div>',
                      unsafe_allow_html=True)
        f = right.file_uploader("Upload", type=["jpg", "jpeg", "png"],
                                label_visibility="collapsed")

    thr, alpha = st.session_state.thr, st.session_state.alpha
    c1, c2, c3 = st.columns([1.15, 1.15, 1])

    if f is None:
        for col, key, icon, title, badge in [(c1, "card_o", "🖼️", "Original X-Ray Image", "Input"),
                                             (c2, "card_h", "🔥", "Grad-CAM Heatmap", "Model Attention")]:
            with col, st.container(border=True, key=key):
                header(icon, title, badge)
                st.markdown('<div class="empty">Upload a chest X-ray to begin</div>', unsafe_allow_html=True)
        with c3, st.container(border=True, key="card_r"):
            st.markdown('<div class="empty">Results will appear here</div>', unsafe_allow_html=True)
    else:
        data = f.getvalue()
        disp, overlay, p = analyze(data, alpha, wnet, cam)

        sig = (f.name, len(data), round(thr, 2))
        if st.session_state.seen != sig:
            st.session_state.seen = sig
            st.session_state.history.insert(0, {
                "time": datetime.now().strftime("%d %b %Y, %H:%M"), "name": f.name, "p": p,
                "label": "PNEUMONIA" if p >= thr else "NORMAL",
                "thumb": cv2.resize(overlay, (140, 140))})

        with c1, st.container(border=True, key="card_o"):
            header("🖼️", "Original X-Ray Image", "Input")
            st.image(disp, use_container_width=True)
            chip("Chest X-Ray (Original)", f"{DISP} × {DISP}")
        with c2, st.container(border=True, key="card_h"):
            header("🔥", "Grad-CAM Heatmap", "Model Attention")
            st.image(overlay, use_container_width=True)
            st.markdown('<div class="cbar"></div><div class="clab"><span>Low</span>'
                        '<span>Medium</span><span>High</span></div>', unsafe_allow_html=True)
            chip("AI Heatmap (PNEUMONIA)", f"{DISP} × {DISP}")
        with c3:
            result_card(p, thr)
            with st.container(border=True, key="card_p"):
                prob_card(p)

elif page.endswith("History"):
    with st.container(border=True, key="card_hist"):
        header("🕒", "Analysis History", f"{len(st.session_state.history)} scans")
        if not st.session_state.history:
            st.markdown('<div class="empty">No scans analysed yet</div>', unsafe_allow_html=True)
        for h in st.session_state.history:
            a, b, c, d = st.columns([1, 3, 2, 2], vertical_alignment="center")
            a.image(h["thumb"], width=70)
            b.markdown(f"**{h['name']}**  \n<span style='color:#8fa3c8'>{h['time']}</span>",
                       unsafe_allow_html=True)
            colour = "#ff4d6a" if h["label"] == "PNEUMONIA" else "#2de28b"
            c.markdown(f"<b style='color:{colour}'>{h['label']}</b>", unsafe_allow_html=True)
            d.markdown(f"P(pneumonia): **{h['p']:.1%}**")
        if st.session_state.history and st.button("Clear history"):
            st.session_state.history = []
            st.session_state.seen = None
            st.rerun()

elif page.endswith("About"):
    with st.container(border=True, key="card_about"):
        header("ℹ️", "About", "v1.0")
        st.markdown("""
**Pneumonia Detector** classifies chest X-rays as *Normal* or *Pneumonia* with a fine-tuned
YOLO classification network (transfer learning), tuned for **high recall** so sick patients are
not missed. **Grad-CAM** heatmaps show which regions of the image influenced the prediction.

- Dataset: Kaggle *Chest X-Ray Pneumonia* (pediatric scans)
- Stack: PyTorch, Ultralytics YOLO, pytorch-grad-cam, Streamlit
- Limitations: trained on one source; results may not generalise to other hospitals or adults.

> Educational project. Not a medical device and not a substitute for a radiologist.""")

else:  # Settings
    with st.container(border=True, key="card_set"):
        header("⚙️", "Settings", "Model")
        thr = st.slider("Pneumonia threshold", 0.05, 0.95, float(st.session_state.thr), 0.01,
                        help="Lower = higher recall (fewer missed cases, more false alarms). "
                             "Use the threshold chosen on the validation set in the notebook.")
        alpha = st.slider("Heatmap strength", 0.2, 0.9, float(st.session_state.alpha), 0.05)
        st.session_state.thr, st.session_state.alpha = thr, alpha
        st.caption(f"Device: {DEVICE}  |  Model: {MODEL_PATH}")
