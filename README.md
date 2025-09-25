# Deepfake Detection System
## Overview

This repository implements an image-based deepfake detection system using the FaceForensics++ dataset. It provides tools for preprocessing and detecting manipulated images using a pre-trained detection model.

## Dataset: FaceForensics++

Request Access
Visit the FaceForensics++ project page
 and fill out the access form.

Download Dataset
After receiving approval, use the provided download script to obtain the dataset.

Organize Files
Ensure the dataset directories are structured as follows:

DeepFakeDetection
├── Deepfakes
├── Face2Face
├── FaceSwap
├── NeuralTextures
├── FaceShifter
└── Originals


## Installation

Clone the Repository
```
git clone https://github.com/shrav-n-9/Deepfake-Detection-System.git
cd Deepfake-Detection-System
```

Create a Virtual Environment
```
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Install Dependencies
```
pip install -r requirements.txt
```
## Usage

Launch Jupyter Notebook:
```
jupyter notebook
```

Open image_detection.ipynb and follow the instructions to:

Load the FaceForensics++ dataset

Preprocess images

Run the deepfake detection model

Save the detection results

Project Structure

image_detection.ipynb — Notebook for deepfake detection

preprocess.py — Image preprocessing utilities

requirements.txt — Python dependencies

README.md — Project documentation