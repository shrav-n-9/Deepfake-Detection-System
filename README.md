# Deepfake Detection System

![Python](https://img.shields.io/badge/python-3.10-blue) ![License](https://img.shields.io/badge/license-MIT-green)

A **Deepfake Detection System** built on the FaceForensics++ dataset to identify and classify manipulated faces using state-of-the-art computer vision techniques. This project demonstrates deepfake detection through training, inference, and an interactive app interface.

---

## Features

* Detects deepfakes across multiple manipulation methods:

  * FaceShifter
  * Face2Face
  * FaceSwap
  * NeuralTextures
  * DeepFakes
  * Originals
* Trained on the **FaceForensics++** dataset.
* Supports **CPU** and **GPU** environments.
* Generates `.pth` model weights from training.
* Interactive app (`app.py`) for real-time deepfake detection.

---

## Project Structure

```plaintext
Deepfake-Detection-System/
├── app.py                # Main application script
├── image_detection.ipynb # Training notebook for deepfake detection
├── requirements.txt      # Dependencies for CPU
├── requirements-gpu.txt  # Dependencies for GPU
├── models/               # Saved .pth model weights
├── data/                 # FaceForensics++ dataset (after download)
└── README.md             # Project documentation
```

---

## Dataset Preparation (FaceForensics++)

The system uses the **FaceForensics++ dataset**.

1. Request access from the [FaceForensics++ website](https://github.com/ondyari/FaceForensics).
2. Once access is granted, download datasets with:

```bash
python download_FF++.py --dataset FaceShifter
python download_FF++.py --dataset Face2Face
python download_FF++.py --dataset FaceSwap
python download_FF++.py --dataset NeuralTextures
python download_FF++.py --dataset DeepFakes
python download_FF++.py --dataset Originals
```

Place all datasets under the `data/` directory.

---

## Environment Setup

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/MacOS
venv\Scripts\activate      # Windows
```

### Install Dependencies

* For CPU-only systems:

```bash
pip install -r requirements.txt
```

* For GPU-enabled systems:

```bash
pip install -r requirements-gpu.txt
```

---

## Model Training

Open and run the Jupyter Notebook `image_detection.ipynb` sequentially:

1. Preprocess dataset.
2. Train the detection model.
3. Save the trained model weights (`.pth` files).

Trained weights will be stored in the `models/` directory.

---

## Running the Application

After training, launch the app:

```bash
python app.py
```

The app will load the trained `.pth` model weights and perform deepfake detection.

---
## 📸 Demo / Screenshots

### App Interface
![App Interface](assets/Screenshot%202025-09-26%20223946.png)

### Detection Examples
![Detection Example 1](assets/Screenshot%202025-09-26%20225128.png)  
![Detection Example 2](assets/Screenshot%202025-09-26%20225146.png)  
![Detection Example 3](assets/Screenshot%202025-09-26%20225256.png)

---

## Results / Benchmarks

- **ROC AUC:** 0.876  
- **Overall Accuracy:** 79%  

### Class-wise Performance
| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Real  | 0.53      | 0.94   | 0.68     |
| Fake  | 0.97      | 0.75   | 0.85     |

### Confusion Matrix

[[3628 248]
[3175 9621]]

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

* [FaceForensics++ dataset](https://github.com/ondyari/FaceForensics)
* Relevant research papers and contributors to deepfake detection.

---

You are now ready to detect deepfakes with the Deepfake Detection System!
