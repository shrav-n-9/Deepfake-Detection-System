import torch
from timm import create_model
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# -------------------------------
# Setup
# -------------------------------
app = Flask(__name__)
CORS(app)  # allow cross-origin (frontend can call API)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Xception model (change num_classes if different in your checkpoint)
model = create_model("xception", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(
    "work_pytorch/xception_best.pth", map_location=device
))
model.eval().to(device)

# Preprocessing pipeline (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])  # normalize to [-1,1]
])

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return {"message": "Deepfake Detector API is running 🚀"}

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    # Preprocess
    x = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = "fake" if probs[1] > 0.5 else "real"
    confidence = float(probs[1] if pred_class == "fake" else probs[0])

    return jsonify({
        "prediction": pred_class,
        "confidence": confidence
    })


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
