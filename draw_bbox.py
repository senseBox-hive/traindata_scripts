import argparse
import os, re, shutil
import glob
from pathlib import Path
import cv2

FRAME_DIR = "uncropped"
CROP_DIR = "unsorted"
OUT_DIR = "frames_bbox"

CROP_SIZE = 32

def extract_metadata(path):
    # information stored in the title
    filename = os.path.basename(path)

    exp = "(.*)__x([0-9]+)_y([0-9]+)_a([0-9]+)\\.(?:png|jpg)"
    match = re.search(exp, filename)
    if match is None:
        raise ValueError(f"Filename does not contain crop metadata: {filename}")

    frame_id = match.group(1)
    x = int(match.group(2))
    y = int(match.group(3))

    return {
        "filename": filename,
        "frame_id": frame_id,
        "x": x,
        "y": y,
        "width": CROP_SIZE,
        "height": CROP_SIZE
    }

def main():
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes from extracted crops over raw frames."
    )
    parser.add_argument("--inference", action=argparse.BooleanOptionalAction, help=f"Whether crops are also inferred")
    args = parser.parse_args()

    INFERENCE = args.inference
    if INFERENCE:
        OUT_DIR = frames_bbox_inferred

    frame_paths = sorted(
        glob.glob(os.path.join(FRAME_DIR, "*.jpg")) +
        glob.glob(os.path.join(FRAME_DIR, "*.png"))
    )
    crop_paths = sorted(
        glob.glob(os.path.join(CROP_DIR, "*.jpg")) +
        glob.glob(os.path.join(CROP_DIR, "*.png"))
    )

    for frame in frame_paths:
        frame_id = Path(frame).stem
        # get all crops matching id
        crops = filter(
            lambda cr: extract_metadata(cr)["frame_id"] == frame_id, 
            crop_paths
        )
        crop_metadata = [extract_metadata(cr) for cr in crops]

        # draw bounding boxes
        img = cv2.imread(frame)        
        for metadata in crop_metadata:
            x, y = metadata["x"], metadata["y"]
            cv2.rectangle(img,
                          (x - CROP_SIZE // 2, y - CROP_SIZE // 2),
                          (x + CROP_SIZE // 2, y + CROP_SIZE // 2),
                          (200, 50, 255), 1)
        
        cv2.imwrite(os.path.join(OUT_DIR, f"{frame_id}.png"), img)

if __name__ == "__main__":
    main()
