import torch
from torchvision import transforms
from PIL import Image
import sys

from model import get_model

# ------------------
# Config
# ------------------
MODEL_PATH = "best_model.pth"
CLASS_NAMES = ["Normal", "Sick"]

device = "cuda" if torch.cuda.is_available() else "cpu"


def predict(image_path):
    # Load model
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Image preprocessing (SAME as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = CLASS_NAMES[predicted.item()]
    confidence = confidence.item() * 100

    print("\nPrediction Result")
    print("------------------")
    print(f"Class      : {label}")
    print(f"Confidence : {confidence:.2f}%")

    if label == "Sick":
        print("\n⚠️ This indicates abnormal thermal patterns.")
        print("⚠️ NOT a medical diagnosis.")
    else:
        print("\n✅ Thermal pattern appears normal.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    predict(sys.argv[1])
