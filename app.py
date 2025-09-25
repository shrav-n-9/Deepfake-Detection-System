import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS  # Added for CORS support
from PIL import Image
import torchvision.transforms as transforms
from timm import create_model

# Initialize Flask app
app = Flask(__name__, static_folder='front end', static_url_path='/')
CORS(app, resources={r"/predict": {"origins": "*"}})  # Allow CORS for /predict endpoint

# Model configuration (matching notebook)
IMG_SIZE = 299
MODEL_PATH = os.path.join("work_pytorch", "xception_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["fake", "real"]

# Load model
try:
    model = create_model("xception", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
except FileNotFoundError:
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Ensure 'xception_best.pth' exists in 'work_pytorch/'")
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

# Image preprocessing (matching notebook)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"error": "Unsupported file type. Use PNG, JPG, or JPEG"}), 400

        # Load and preprocess image
        try:
            img = Image.open(file).convert('RGB')
        except Exception as e:
            return jsonify({"error": f"Invalid image: {str(e)}"}), 400

        img = transform(img)
        img = img.unsqueeze(0)  # Add batch dimension

        # Run inference
        with torch.no_grad():
            img = img.to(DEVICE)
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = probs.max(dim=1)
            prediction = CLASSES[pred_idx.item()]

        # Log prediction
        log_path = os.path.join("work_pytorch", "predictions.log")
        try:
            with open(log_path, "a") as f:
                f.write(f"[{request.remote_addr}] Prediction: {prediction}, Confidence: {confidence.item()}, Filename: {file.filename}\n")
        except Exception as e:
            print(f"Warning: Failed to log prediction: {str(e)}")

        return jsonify({
            "prediction": prediction,
            "confidence": confidence.item()
        })

    except Exception as e:
        # Log server error
        print(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)