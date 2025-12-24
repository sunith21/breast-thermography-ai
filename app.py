import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

from src.model import get_model

# ------------------
# CONFIG
# ------------------
MODEL_PATH = "best_model.pth"
CLASS_NAMES = ["Normal", "Sick"]
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------
# LOAD MODEL (cached)
# ------------------
@st.cache_resource
def load_model():
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# ------------------
# TRANSFORM (same as training)
# ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ------------------
# UI
# ------------------
st.set_page_config(page_title="Breast Thermography AI", layout="centered")

st.title("🩺 Breast Thermography AI Screening")
st.write("Upload a **thermal image** for AI-assisted screening.")
st.warning("⚠️ This is NOT a medical diagnosis. For research use only.")

uploaded_file = st.file_uploader(
    "Drag & drop an image here",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=500)


    if st.button("🔍 Predict"):
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        label = CLASS_NAMES[predicted.item()]
        confidence = confidence.item() * 100

        st.subheader("Prediction Result")
        st.write(f"**Class:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        if label == "Sick":
            st.error("⚠️ Abnormal thermal pattern detected")
        else:
            st.success("✅ Thermal pattern appears normal")
