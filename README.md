# 🧠 EmoSense AI — Emotion Detection App

Real-time facial emotion detection with live camera & video upload.

---

## 📁 Project Structure

```
emotion_app/
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── packages.txt            ← System-level dependencies (Streamlit Cloud)
├── .streamlit/
│   └── config.toml         ← Streamlit theme & server config
└── README.md
```

---

## 🚀 Deployment Guide — Streamlit Cloud (Free)

### Step 1 — Push to GitHub

```bash
# Initialize a new repo
git init
git add .
git commit -m "Initial EmoSense AI app"

# Push to GitHub (create repo first at github.com)
git remote add origin https://github.com/YOUR_USERNAME/emosense-ai.git
git branch -M main
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with your GitHub account
3. Click **"New App"**
4. Fill in:
   - **Repository**: `YOUR_USERNAME/emosense-ai`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy!"**

⏱️ First deploy takes ~3–5 minutes to install TensorFlow.

### Step 3 — Share Your App

Your app will be live at:
```
https://YOUR_USERNAME-emosense-ai-app-XXXX.streamlit.app
```

---

## 💻 Run Locally

```bash
# 1. Clone your repo
git clone https://github.com/YOUR_USERNAME/emosense-ai.git
cd emosense-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open browser at: **http://localhost:8501**

---

## 🔧 Troubleshooting

| Issue | Fix |
|---|---|
| Camera not working on Streamlit Cloud | Use video upload (cloud browsers restrict webcam in iframes) |
| Model download slow | First load caches it — subsequent loads are instant |
| `libGL.so` error | `packages.txt` includes system libs — ensure it's committed |
| TF version mismatch | Pin `tensorflow==2.13.0` in requirements.txt |

---

## 🎭 Detected Emotions

| Emotion | Emoji |
|---|---|
| Angry | 😠 |
| Disgust | 🤢 |
| Fear | 😨 |
| Happy | 😄 |
| Sad | 😢 |
| Surprise | 😲 |
| Neutral | 😐 |

---

## ⚠️ Camera Note for Cloud Deployment

Streamlit Cloud runs in a browser sandbox. Live camera access works best when:
- App is accessed via **HTTPS** (Streamlit Cloud provides this)
- Browser grants camera permissions
- User is on **Chrome or Edge** (best WebRTC support)

For video upload mode, any MP4/AVI/MOV/MKV file works perfectly.

---

## Model Info

- **Architecture**: Custom CNN trained on FER-2013
- **HuggingFace**: `sambhavjain13/emotion_model_full.h5`
- **Input size**: 48×48 grayscale
- **Output**: 7-class softmax
