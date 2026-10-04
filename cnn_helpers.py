import torch
from torchvision import datasets, models, transforms
from PIL import Image
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def infer(crop_filepath, model: torch.nn.Module, crop_size):
    #img = cv2.imread(crop_filepath) #TODO: check RGB - BGR flip
    img = Image.open(crop_filepath)
    # First prepare the transformations: resize the image to what the model was trained on and convert it to a tensor
    data_transform = transforms.Compose(
        [transforms.Resize((crop_size, crop_size)), 
        transforms.ToTensor()]
    )
    image = data_transform(img).unsqueeze(0).to(DEVICE)

    #predict and return most confident
    output = model(image)
    return(output.argmax())

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