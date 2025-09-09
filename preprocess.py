import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import torch
from tqdm import tqdm
import logging
import random

# Set up logging
logging.basicConfig(filename='preprocess.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
DATASET_ROOT = "FFPlus"
OUTPUT_ROOT = "data"
IMG_SIZE = 224
FRAME_RATE = 1  # Extract 1 frame per second
TRAIN_SPLIT = 0.8  # 80% train, 20% test
SEED = 42
random.seed(SEED)

# Categories
CATEGORIES = {
    'real': ['original_sequences/youtube/c23/videos'],
    'fake': ['manipulated_sequences/Deepfakes/c23/videos',
             'manipulated_sequences/FaceShifter/c23/videos',
             'manipulated_sequences/Face2Face/c23/videos',
             'manipulated_sequences/FaceSwap/c23/videos',
             'manipulated_sequences/NeuralTextures/c23/videos']
}

# Output directories
for split in ['train', 'test']:
    for label in ['real', 'fake']:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, label), exist_ok=True)

# Initialize MTCNN (CPU to avoid RTX 2050 VRAM issues)
device = torch.device('cpu')
mtcnn = MTCNN(image_size=IMG_SIZE, margin=20, min_face_size=20, device=device)

def extract_frames(video_path, frame_rate=FRAME_RATE):
    """Extract frames from a video at specified frame rate."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Cannot open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / frame_rate) if fps > 0 else 1
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_count += 1
        
        cap.release()
        return frames
    except Exception as e:
        logging.error(f"Error extracting frames from {video_path}: {str(e)}")
        return []

def detect_and_save_faces(frame, mtcnn, output_path, max_faces=1):
    """Detect faces with MTCNN and save them."""
    try:
        frame_pil = Image.fromarray(frame)
        faces, _ = mtcnn.detect(frame_pil)
        if faces is None:
            return 0
        
        saved_faces = 0
        for i, face in enumerate(faces[:max_faces]):
            x1, y1, x2, y2 = map(int, face)
            if x2 - x1 < 20 or y2 - y1 < 20:  # Skip small detections
                continue
            face_img = frame_pil.crop((x1, y1, x2, y2))
            face_img = face_img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            face_img.save(output_path.replace('.jpg', f'_face{i}.jpg'), 'JPEG')
            saved_faces += 1
        return saved_faces
    except Exception as e:
        logging.error(f"Error detecting faces in frame: {str(e)}")
        return 0

def process_videos(category, video_dir, label, train_videos, test_videos):
    """Process videos for a given category and split."""
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]
    random.shuffle(video_files)
    
    train_count = int(len(video_files) * TRAIN_SPLIT)
    train_files = video_files[:train_count]
    test_files = video_files[train_count:]
    
    for split, files in [('train', train_files), ('test', test_files)]:
        output_dir = os.path.join(OUTPUT_ROOT, split, label)
        for video in tqdm(files, desc=f"Processing {label}/{split} in {video_dir}"):
            video_path = os.path.join(video_dir, video)
            frames = extract_frames(video_path)
            
            for i, frame in enumerate(frames):
                face_path = os.path.join(output_dir, f"{video}_{i}_face0.jpg")
                if os.path.exists(face_path):
                    continue
                output_path = os.path.join(output_dir, f"{video}_{i}.jpg")
                saved_faces = detect_and_save_faces(frame, mtcnn, output_path)
                if saved_faces > 0:
                    logging.info(f"Saved {saved_faces} faces from {video_path}, frame {i}")

def main():
    # Process each category
    for label, paths in CATEGORIES.items():
        for path in paths:
            video_dir = os.path.join(DATASET_ROOT, path)
            if not os.path.exists(video_dir):
                logging.error(f"Directory not found: {video_dir}")
                continue
            logging.info(f"Processing {label} videos in {video_dir}")
            process_videos(path, video_dir, label, [], [])

if __name__ == "__main__":
    main()