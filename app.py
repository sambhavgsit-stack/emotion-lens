import os

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import io

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmoSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load Model (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    import keras
    from huggingface_hub import hf_hub_download
    with st.spinner("🔄 Loading EmoSense model from HuggingFace..."):
        model_path = hf_hub_download(
            repo_id="sambhavjain13/emotion_model_full.h5",
            filename="emotion_model_full.h5"
        )
        model = keras.models.load_model(model_path, compile=False)
    return model
# ─── Emotion Config ─────────────────────────────────────────────────────────
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

EMOTION_COLORS = {
    'Angry':    '#FF4B4B',
    'Disgust':  '#A8FF3E',
    'Fear':     '#9B59B6',
    'Happy':    '#FFD700',
    'Sad':      '#4B9FFF',
    'Surprise': '#FF8C00',
    'Neutral':  '#B0C4DE',
}

EMOTION_EMOJIS = {
    'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨',
    'Happy': '😄', 'Sad': '😢', 'Surprise': '😲', 'Neutral': '😐'
}

# ─── Face Detector ──────────────────────────────────────────────────────────
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face_img):
    face = cv2.resize(face_img, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face

def predict_emotion(model, face_img):
    processed = preprocess_face(face_img)
    preds = model.predict(processed, verbose=0)[0]
    idx = np.argmax(preds)
    return EMOTIONS[idx], float(preds[idx]), preds

def draw_emotion_overlay(frame, x, y, w, h, emotion, confidence, color_hex):
    color_bgr = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 2)
    # Draw label background
    label = f"{EMOTION_EMOJIS.get(emotion,'')} {emotion} {confidence*100:.1f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y-th-12), (x+tw+10, y), color_bgr, -1)
    cv2.putText(frame, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame

def process_frame(frame, model, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    results = []
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        try:
            emotion, confidence, all_preds = predict_emotion(model, face_roi)
            color = EMOTION_COLORS.get(emotion, '#FFFFFF')
            frame = draw_emotion_overlay(frame, x, y, w, h, emotion, confidence, color)
            results.append({'emotion': emotion, 'confidence': confidence, 'all_preds': all_preds})
        except Exception:
            pass
    return frame, results

# ─── Custom CSS ─────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    :root {
        --bg: #0A0A0F;
        --surface: #111118;
        --surface2: #1A1A24;
        --accent: #7C3AED;
        --accent2: #06B6D4;
        --text: #F0F0FF;
        --muted: #6B7280;
        --border: rgba(124,58,237,0.3);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 1.5rem 2rem 2rem 2rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] > div { padding: 1.5rem 1rem; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #0A0A0F 0%, #1a0533 50%, #0A0F1A 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(ellipse at 30% 40%, rgba(124,58,237,0.15) 0%, transparent 60%),
                    radial-gradient(ellipse at 70% 60%, rgba(6,182,212,0.10) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #A78BFA, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.1;
    }
    .hero-sub {
        color: var(--muted);
        font-size: 1.05rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    .badge {
        display: inline-block;
        background: rgba(124,58,237,0.2);
        border: 1px solid rgba(124,58,237,0.5);
        border-radius: 50px;
        padding: 0.2rem 0.8rem;
        font-size: 0.75rem;
        color: #A78BFA;
        margin-right: 0.5rem;
        margin-top: 0.8rem;
    }

    /* Emotion card */
    .emotion-card {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s;
    }
    .emotion-card:hover { border-color: var(--accent); }
    .emotion-name {
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
    }

    /* Metric pill */
    .metric-pill {
        background: rgba(124,58,237,0.12);
        border: 1px solid rgba(124,58,237,0.3);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #A78BFA;
    }
    .metric-label {
        font-size: 0.75rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Mode toggle buttons */
    .stButton > button {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        border-color: var(--accent) !important;
        background: rgba(124,58,237,0.15) !important;
        color: #A78BFA !important;
    }

    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
        border-radius: 10px !important;
    }

    /* Divider */
    hr { border-color: var(--border) !important; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: var(--surface2) !important;
        border: 2px dashed var(--border) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
    }

    /* Section header */
    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #A78BFA;
        margin-bottom: 0.5rem;
    }

    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 8px #22c55e;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    </style>
    """, unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; margin-bottom:1.5rem;">
            <div style="font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800;
                        background:linear-gradient(90deg,#A78BFA,#06B6D4);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                🧠 EmoSense
            </div>
            <div style="color:#6B7280; font-size:0.8rem;">Real-time Emotion Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">⚙️ Settings</div>', unsafe_allow_html=True)

        detection_mode = st.radio(
            "Detection Mode",
            ["📷 Live Camera", "🎬 Upload Video"],
            index=0,
            help="Choose how to feed video into the model"
        )

        st.markdown("---")
        st.markdown('<div class="section-title">🎛️ Parameters</div>', unsafe_allow_html=True)

        confidence_threshold = st.slider(
            "Min Confidence %", min_value=10, max_value=90, value=40, step=5
        ) / 100

        frame_skip = st.slider(
            "Frame Processing Rate", min_value=1, max_value=5, value=2, step=1,
            help="Process every Nth frame (higher = faster)"
        )

        show_all_emotions = st.toggle("Show Full Emotion Breakdown", value=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem; color:#6B7280; line-height:1.6;">
            <b style="color:#A78BFA;">Model:</b> FER-2013 Architecture<br>
            <b style="color:#A78BFA;">Classes:</b> 7 Emotions<br>
            <b style="color:#A78BFA;">Input:</b> 48×48 Grayscale<br>
            <b style="color:#A78BFA;">Source:</b> HuggingFace Hub
        </div>
        """, unsafe_allow_html=True)

    return detection_mode, confidence_threshold, frame_skip, show_all_emotions

# ─── Emotion Bar Chart ────────────────────────────────────────────────────────
def render_emotion_bars(all_preds):
    st.markdown('<div class="section-title">📊 Emotion Breakdown</div>', unsafe_allow_html=True)
    sorted_pairs = sorted(zip(EMOTIONS, all_preds), key=lambda x: x[1], reverse=True)
    for emo, prob in sorted_pairs:
        col1, col2 = st.columns([3, 1])
        with col1:
            color = EMOTION_COLORS.get(emo, '#A78BFA')
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                <span style="font-size:1.1rem;">{EMOTION_EMOJIS.get(emo,'')}</span>
                <span style="font-size:0.85rem; font-weight:500; color:#D1D5DB; width:70px;">{emo}</span>
                <div style="flex:1; background:rgba(255,255,255,0.05); border-radius:6px; height:10px; overflow:hidden;">
                    <div style="width:{prob*100:.1f}%; background:{color}; height:100%; border-radius:6px;
                                transition:width 0.5s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div style="text-align:right; font-size:0.8rem; color:#9CA3AF; padding-top:2px;">{prob*100:.1f}%</div>', unsafe_allow_html=True)

# ─── Live Camera Mode ─────────────────────────────────────────────────────────
def live_camera_mode(model, face_cascade, confidence_threshold, show_all_emotions):
    st.markdown('<div class="section-title">📷 Live Camera Detection</div>', unsafe_allow_html=True)

    col_cam, col_stats = st.columns([3, 2])

    with col_cam:
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

    with col_stats:
        st.markdown("""
        <div class="emotion-card" style="border-color:rgba(6,182,212,0.4);">
            <div style="font-size:0.75rem; color:#6B7280; text-transform:uppercase; letter-spacing:1px;">Current Emotion</div>
            <div id="emotion-display" style="font-family:'Syne',sans-serif; font-size:2.5rem; font-weight:800; color:#A78BFA;">
                — Waiting —
            </div>
        </div>
        """, unsafe_allow_html=True)
        dominant_placeholder = st.empty()
        breakdown_placeholder = st.empty()

    start, stop_col = st.columns(2)
    with start:
        run = st.toggle("🟢 Start Camera", value=False, key="cam_toggle")

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("⚠️ Could not access camera. Please check permissions.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0
        last_results = []

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Frame capture failed.")
                break

            frame_count += 1
            processed_frame = frame.copy()
            processed_frame, results = process_frame(processed_frame, model, face_cascade)

            if results:
                last_results = [r for r in results if r['confidence'] >= confidence_threshold]

            rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

            if last_results:
                top = max(last_results, key=lambda x: x['confidence'])
                emo = top['emotion']
                conf = top['confidence']
                color = EMOTION_COLORS.get(emo, '#A78BFA')

                dominant_placeholder.markdown(f"""
                <div class="emotion-card" style="border-color:{color}50;">
                    <div style="font-size:3rem; text-align:center;">{EMOTION_EMOJIS.get(emo,'')}</div>
                    <div class="emotion-name" style="color:{color}; text-align:center; font-size:1.8rem;">{emo}</div>
                    <div style="text-align:center; color:#9CA3AF; font-size:0.9rem;">{conf*100:.1f}% confident</div>
                </div>
                """, unsafe_allow_html=True)

                if show_all_emotions:
                    with breakdown_placeholder.container():
                        render_emotion_bars(top['all_preds'])

            status_placeholder.markdown(
                f'<span class="status-dot"></span><span style="color:#6B7280; font-size:0.8rem;">'
                f'Frame {frame_count} | {len(last_results)} face(s) detected</span>',
                unsafe_allow_html=True
            )

            # Respect toggle
            run = st.session_state.get("cam_toggle", False)

        cap.release()
        frame_placeholder.empty()
        st.info("Camera stopped.")

# ─── Video Upload Mode ────────────────────────────────────────────────────────
def video_upload_mode(model, face_cascade, confidence_threshold, frame_skip, show_all_emotions):
    st.markdown('<div class="section-title">🎬 Video Upload Detection</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your video here", type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supports MP4, AVI, MOV, MKV"
    )

    if uploaded is None:
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#4B5563;">
            <div style="font-size:3rem;">🎬</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.2rem; margin-top:0.5rem;">
                Upload a video to begin analysis
            </div>
            <div style="font-size:0.85rem; margin-top:0.3rem;">
                Supported: MP4, AVI, MOV, MKV
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded.read())
    tfile.close()

    col_v, col_s = st.columns([3, 2])

    with col_v:
        video_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

    with col_s:
        stats_placeholder = st.empty()
        breakdown_placeholder = st.empty()

    if st.button("▶️ Analyze Video", type="primary"):
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        emotion_counts = {e: 0 for e in EMOTIONS}
        frame_idx = 0
        last_results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            progress = frame_idx / max(total_frames, 1)
            progress_bar.progress(min(progress, 1.0))

            if frame_idx % frame_skip != 0:
                continue

            processed, results = process_frame(frame.copy(), model, face_cascade)

            if results:
                last_results = [r for r in results if r['confidence'] >= confidence_threshold]
                for r in last_results:
                    emotion_counts[r['emotion']] += 1

            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels="RGB", use_container_width=True)

            elapsed_sec = frame_idx / fps
            status_text.markdown(
                f'<span class="status-dot"></span><span style="color:#6B7280; font-size:0.8rem;">'
                f'Frame {frame_idx}/{total_frames} | {elapsed_sec:.1f}s | '
                f'{len(last_results)} face(s)</span>',
                unsafe_allow_html=True
            )

            # Live stats
            total_detections = sum(emotion_counts.values()) or 1
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            color = EMOTION_COLORS.get(dominant_emotion, '#A78BFA')

            stats_placeholder.markdown(f"""
            <div class="emotion-card" style="border-color:{color}50;">
                <div style="font-size:2.5rem; text-align:center;">{EMOTION_EMOJIS.get(dominant_emotion,'')}</div>
                <div class="emotion-name" style="color:{color}; text-align:center; font-size:1.5rem;">{dominant_emotion}</div>
                <div style="text-align:center; color:#9CA3AF; font-size:0.85rem; margin-top:4px;">
                    Dominant emotion so far
                </div>
            </div>
            <div style="margin-top:0.8rem;">
            """ + "".join([
                f'<div style="display:flex; justify-content:space-between; align-items:center; '
                f'margin-bottom:3px; font-size:0.82rem;">'
                f'<span>{EMOTION_EMOJIS.get(e,"")} {e}</span>'
                f'<span style="color:{EMOTION_COLORS.get(e,"#A78BFA")}; font-weight:600;">'
                f'{emotion_counts[e]/total_detections*100:.1f}%</span></div>'
                for e in sorted(emotion_counts, key=emotion_counts.get, reverse=True)
            ]) + "</div>", unsafe_allow_html=True)

        cap.release()
        os.unlink(tfile.name)
        progress_bar.progress(1.0)

        # Final summary
        st.markdown("---")
        st.markdown('<div class="section-title">📈 Analysis Summary</div>', unsafe_allow_html=True)
        total_d = sum(emotion_counts.values()) or 1
        cols = st.columns(4)
        top4 = sorted(emotion_counts, key=emotion_counts.get, reverse=True)[:4]
        for i, emo in enumerate(top4):
            with cols[i]:
                pct = emotion_counts[emo] / total_d * 100
                color = EMOTION_COLORS.get(emo, '#A78BFA')
                st.markdown(f"""
                <div class="metric-pill" style="border-color:{color}40;">
                    <div style="font-size:1.8rem;">{EMOTION_EMOJIS.get(emo,'')}</div>
                    <div class="metric-value" style="color:{color}; font-size:1.5rem;">{pct:.1f}%</div>
                    <div class="metric-label">{emo}</div>
                </div>
                """, unsafe_allow_html=True)

        st.success(f"✅ Analysis complete! Processed {frame_idx} frames — dominant emotion: **{dominant_emotion}**")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    inject_css()

    # Hero
    st.markdown("""
    <div class="hero">
        <div>
            <span class="badge">🤖 AI-Powered</span>
            <span class="badge">⚡ Real-time</span>
            <span class="badge">🎯 7 Emotions</span>
        </div>
        <h1 class="hero-title">EmoSense AI</h1>
        <p class="hero-sub">Real-time facial emotion detection powered by deep learning.<br>
        Live camera & video analysis in one unified interface.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar always renders first
    detection_mode, confidence_threshold, frame_skip, show_all_emotions = render_sidebar()

    # Load model
    model = None
    face_cascade = None
    with st.spinner("🔄 Loading model from HuggingFace (first load ~30s)..."):
        try:
            model = load_model()
            face_cascade = load_face_cascade()
            st.success("✅ Model loaded — sambhavjain13/emotion_model_full.h5")
        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")
            st.code(str(e), language="text")
            st.warning("⚠️ Check that tensorflow-cpu==2.13.0 and keras==2.13.1 are in requirements.txt")
            return

    st.markdown("---")

    # Route to mode
    if "Live Camera" in detection_mode:
        live_camera_mode(model, face_cascade, confidence_threshold, show_all_emotions)
    else:
        video_upload_mode(model, face_cascade, confidence_threshold, frame_skip, show_all_emotions)

if __name__ == "__main__":
    main()