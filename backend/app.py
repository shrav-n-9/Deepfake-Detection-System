import torch
from timm import create_model
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable CuDNN benchmark for slight inference speedup on RTX 2050
torch.backends.cudnn.benchmark = True

# Load model once
model = create_model("xception", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("work_pytorch/xception_best.pth", map_location=device))
model.eval().to(device)

# Preprocessing
tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

app = Flask(__name__)
CORS(app)  # allow frontend to call backend

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error":"No file uploaded"}), 400
    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")

    x = tfms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    result = "fake" if probs[1] > 0.5 else "real"
    return jsonify({
        "prediction": result,
        "confidence": float(probs[1] if result=="fake" else probs[0])
    })

if __name__ == "__main__":
    app.run(debug=True)