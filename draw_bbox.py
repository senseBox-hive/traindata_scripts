import argparse
import os, re, shutil
import glob
from pathlib import Path
import cv2
import torch
from tqdm.auto import tqdm

FRAME_DIR = "uncropped"
CROP_DIR = "unsorted"
OUT_DIR = "frames_bbox"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CROP_SIZE = 32
LABELS = ["bg","flying","crawling"]
LABEL_COLORS = [
    (200,200,200), #grey
    (0,0,255), #red
    (0,125,255), #yello
]

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
        "filepath": path,
        "frame_id": frame_id,
        "x": x,
        "y": y,
        "width": CROP_SIZE,
        "height": CROP_SIZE,
        "inference": None,
        "color": None
    }

def main():
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes from extracted crops over raw frames."
    )
    parser.add_argument("--inference", action=argparse.BooleanOptionalAction, help=f"Whether crops are also inferred")
    args = parser.parse_args()

    out_dir = OUT_DIR
    model = None

    message = f"Drawing bboxes to {out_dir}"

    INFERENCE = args.inference
    if INFERENCE:
        message = f"Drawing inferred bboxes to {out_dir}"
        #conditional import for better performance
        from cnn_helpers import Net, infer
        out_dir = "frames_bbox_inferred"
        model_path = Path(__file__).parent / "model_loss0.18_acc93.66_hl48.pth"
        model = Net(
            input_shape=3, 
            hidden_units=48,
            output_shape=3
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

    frame_paths = sorted(
        glob.glob(os.path.join(FRAME_DIR, "*.jpg")) +
        glob.glob(os.path.join(FRAME_DIR, "*.png"))
    )
    crop_paths = sorted(
        glob.glob(os.path.join(CROP_DIR, "*.jpg")) +
        glob.glob(os.path.join(CROP_DIR, "*.png"))
    )

    for frame in tqdm(
        frame_paths,
        desc=message, 
        position=0, 
        leave=False
        ):
        frame_id = Path(frame).stem
        # get all crops matching id
        crops = filter(
            lambda cr: extract_metadata(cr)["frame_id"] == frame_id, 
            crop_paths
        )
        crop_metadata = [extract_metadata(cr) for cr in crops]

        # infer images
        if INFERENCE:
            for metadata in crop_metadata:
                label_index = infer(metadata["filepath"], model, CROP_SIZE)
                metadata["inference"] = LABELS[label_index]
                metadata["color"] = LABEL_COLORS[label_index]

            #sort by inference
            crop_metadata.sort(
                key=lambda x: x["inference"]
            )

        # draw bounding boxes
        img = cv2.imread(frame)        
        for metadata in crop_metadata:
            color = metadata["color"] or (200, 50, 255)
            x, y = metadata["x"], metadata["y"]
            cv2.rectangle(img,
                          (x - CROP_SIZE // 2, y - CROP_SIZE // 2),
                          (x + CROP_SIZE // 2, y + CROP_SIZE // 2),
                          color, 1)
            if INFERENCE:
                #draw label text with background
                text_size = 0.25
                (text_width, text_height), baseline = cv2.getTextSize(metadata["inference"], cv2.FONT_HERSHEY_SIMPLEX, text_size, 1)
                cv2.rectangle(img, 
                              (x - CROP_SIZE // 2, y - CROP_SIZE // 2 - text_height - baseline),
                              (x + CROP_SIZE // 2, y - CROP_SIZE // 2), 
                              color, -1)
                cv2.putText(
                    img, 
                    metadata["inference"], 
                    (x - CROP_SIZE // 2, y - CROP_SIZE // 2 - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    text_size, (255,255,255), 2, cv2.LINE_AA)        
        
        cv2.imwrite(os.path.join(out_dir, f"{frame_id}.png"), img)

if __name__ == "__main__":
    main()
