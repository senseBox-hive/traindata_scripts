import argparse
import os, re, shutil
import glob
from pathlib import Path
import cv2
import torch
from torchvision import datasets, models, transforms
from PIL import Image

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

#see bee_detection_training repo
class Net(torch.nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,  # how big is the square that's going over the image?
                      stride=1,  # default
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,  # how big is the square that's going over the image?
                      stride=1,  # default
                      padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,
                stride=2)
        )
        self.block_3 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our input data.
            torch.nn.Linear(in_features=hidden_units*4*4, 
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.block_3(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

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

def infer(crop_filepath, model: torch.nn.Module):
    #img = cv2.imread(crop_filepath) #TODO: check RGB - BGR flip
    img = Image.open(crop_filepath)
    # First prepare the transformations: resize the image to what the model was trained on and convert it to a tensor
    data_transform = transforms.Compose(
        [transforms.Resize((CROP_SIZE, CROP_SIZE)), 
        transforms.ToTensor()]
    )
    image = data_transform(img).unsqueeze(0).to(DEVICE)

    #predict and return most confident
    output = model(image)
    return(output.argmax())

def main():
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes from extracted crops over raw frames."
    )
    parser.add_argument("--inference", action=argparse.BooleanOptionalAction, help=f"Whether crops are also inferred")
    args = parser.parse_args()

    out_dir = OUT_DIR
    model = None

    INFERENCE = args.inference
    if INFERENCE:
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

    for frame in frame_paths:
        frame_id = Path(frame).stem
        # get all crops matching id
        crops = filter(
            lambda cr: extract_metadata(cr)["frame_id"] == frame_id, 
            crop_paths
        )
        crop_metadata = [extract_metadata(cr) for cr in crops]

        # infer images
        if INFERENCE:
            print(f"Inferring {len(crop_metadata)} crops for frame {frame_id}...")
            for metadata in crop_metadata:
                label_index = infer(metadata["filepath"], model)
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
