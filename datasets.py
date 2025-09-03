import os
import random
import argparse
import subprocess
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image
import torch
from tqdm import tqdm

# Step 1: Frame Extraction
def extract_frames(video_path, frame_dir, fps=1, reencode=True):
    frame_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_path.resolve()  # Ensure absolute path
    fixed_video = frame_dir / "fixed.mp4"

    if reencode:
        # Re-encode the video
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "veryfast",
            "-an", str(fixed_video)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg re-encoding failed for {video_path}: {result.stderr}")
        input_video = fixed_video
    else:
        input_video = video_path

    # Extract frames at 1 fps
    cmd_extract = [
        "ffmpeg", "-i", str(input_video),
        "-vf", f"fps={fps}",
        str(frame_dir / "frame_%04d.jpg")
    ]
    result = subprocess.run(cmd_extract, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg frame extraction failed for {video_path}: {result.stderr}")

    # Verify frames were created
    frames = list(frame_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames extracted for {video_path}. Check video or FFmpeg.")
    
    # Clean up fixed video if used
    if reencode:
        fixed_video.unlink(missing_ok=True)

    return len(frames)

# Step 2: Face Cropping (with batching for speed)
def crop_faces(frame_dir, face_dir, mtcnn):
    face_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all frames as list of PIL images
    frame_files = sorted(frame_dir.glob("*.jpg"))
    if not frame_files:
        return
    
    imgs = [Image.open(f).convert("RGB") for f in frame_files]
    
    # Batch detect and extract faces
    faces = mtcnn(imgs)
    
    # Save each (faces is list of tensors or None)
    for frame_file, face in zip(frame_files, faces):
        try:
            if face is not None:
                # Convert [0,1] float tensor to uint8 numpy array
                face_img = (face.permute(1, 2, 0) * 255).byte().cpu().numpy()
                face_pil = Image.fromarray(face_img)
                face_pil.save(face_dir / frame_file.name)
        except Exception as e:
            print(f"Skipping {frame_file}: {e}")

# Step 3: Dataset Builder
def build_dataset(data_root, output_root, test_ratio=0.2, reencode=True):
    all_videos = list(Path(data_root).rglob("*.mp4"))
    print(f"Found {len(all_videos)} videos.")

    # Split into train/test at video level
    random.shuffle(all_videos)
    split_idx = int(len(all_videos) * (1 - test_ratio))
    train_videos = all_videos[:split_idx]
    test_videos = all_videos[split_idx:]

    # Init MTCNN with post_process=False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(image_size=224, margin=20, device=device, post_process=False)
    
    for split_name, split_videos in [("train", train_videos), ("test", test_videos)]:
        for vid in tqdm(split_videos, desc=f"Processing {split_name}"):
            # Determine Label: 'real' or 'fake'
            label = "real" if "original" in str(vid).lower() else "fake"

            # Paths
            frame_dir = output_root / "frames" / split_name / label / vid.stem
            face_dir = output_root / split_name / label / vid.stem

            # Extract frames
            try:
                num_frames = extract_frames(vid, frame_dir, fps=1, reencode=reencode)
                print(f"Extracted {num_frames} frames for {vid}")
            except Exception as e:
                print(f"Frame extraction failed for {vid}: {e}")
                continue
                
            # Crop faces (batched)
            crop_faces(frame_dir, face_dir, mtcnn)
    
    print("Dataset build complete!")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to FaceForensics++ videos")
    parser.add_argument("--output_root", type=str, required=True, help="Where to save dataset")
    parser.add_argument("--no-reencode", action="store_false", dest="reencode", help="Skip video re-encoding")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    build_dataset(data_root, output_root, reencode=args.reencode)