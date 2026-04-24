from ultralytics import YOLO
import tempfile
import os

# Load model once
model = YOLO("models/best.pt")

def run_detection(video_file):
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, video_file.name)

    with open(temp_path, "wb") as f:
        f.write(video_file.read())

    # Run YOLO prediction
    results = model.predict(
        source=temp_path,
        save=True,
        conf=0.25
    )

    # Get output path (Ultralytics saves in runs/detect/)
    output_dir = results[0].save_dir
    output_files = os.listdir(output_dir)

    # Find video output
    for file in output_files:
        if file.endswith((".mp4", ".avi", ".mov")):
            return os.path.join(output_dir, file)

    return None
