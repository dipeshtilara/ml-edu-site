import streamlit as st

st.set_page_config(page_title="AI Learning Hub — Demos", page_icon="🧠", layout="wide")

st.title("🧠 AI Learning Hub — Streamlit Demos")
st.write(
    "This is the Streamlit hub for interactive ML demos from the **ml-edu-site** project. "
    "Use the left sidebar to open each project."
)

st.markdown("""
### Included (so far)
- 📧 Email Spam Detector (Logistic Regression) — *demo data, works offline*
- 💬 Sentiment Analysis — *template page for your next project*
- 🖼️ Image Classifier — *template placeholder*

Add more by creating new Python files in `streamlit_app/pages/NN_YourProject.py`.
""")

st.info("Tip: Keep model files small, or host big ones on Hugging Face and download at runtime with caching.")
