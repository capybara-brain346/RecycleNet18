from typing import OrderedDict
from io import BytesIO
import torch
from torchvision.transforms import InterpolationMode
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# dictionary mapping class indices to waste item categories
class_map = {
    0: "aerosol cans",
    1: "aluminum food_cans",
    2: "aluminum soda cans",
    3: "cardboard boxes",
    4: "cardboard packaging",
    5: "clothing",
    6: "coffee grounds",
    7: "disposable plastic cutlery",
    8: "eggshells",
    9: "food waste",
    10: "glass beverage bottles",
    11: "glass cosmetic containers",
    12: "glass food jars",
    13: "magazines",
    14: "newspaper",
    15: "office paper",
    16: "paper cups",
    17: "plastic cup lids",
    18: "plastic detergent bottles",
    19: "plastic food containers",
    20: "plastic shopping bags",
    21: "plastic soda bottles",
    22: "plastic straws",
    23: "plastic trash bags",
    24: "plastic water bottles",
    25: "shoes",
    26: "steel food cans",
    27: "styrofoam cups",
    28: "styrofoam food containers",
    29: "tea bags",
}


def classify(image_bytes: bytes) -> tuple[int, str, float] | None:
    """
    classifies an image of waste items into one of 30 categories using a fine-tuned resnet18 model.

    args:
        image_bytes (bytes): raw bytes of the input image to be classified

    returns:
        tuple[int, str, float] | none: a tuple containing:
            - class_idx (int): the predicted class index
            - class_name (str): the human-readable class name
            - confidence (float): the model's confidence score for the prediction
        returns none if classification fails
    """
    device = torch.device("cpu")

    transform = transforms.Compose(
        [
            transforms.Resize(
                256, interpolation=InterpolationMode.BILINEAR
            ),  # resize image to 256x256
            transforms.CenterCrop(
                224
            ),  # crop center to 224x224 (standard resnet input size)
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # imagenet normalization values
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # initialise model
    recycle_net = resnet18(weights=ResNet18_Weights.DEFAULT)
    recycle_net = recycle_net.to(device)

    # add custom prediction head
    num_features = recycle_net.fc.in_features
    recycle_net.fc = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(num_features, 512)),
                ("relu1", nn.ReLU()),
                (
                    "bn1",
                    nn.BatchNorm1d(512),
                ),
                ("fc2", nn.Linear(512, 256)),
                ("relu2", nn.ReLU()),
                ("bn2", nn.BatchNorm1d(256)),
                ("fc3", nn.Linear(256, 128)),
                ("relu3", nn.ReLU()),
                ("bn3", nn.BatchNorm1d(128)),
                ("fc4", nn.Linear(128, 64)),
                ("relu4", nn.ReLU()),
                ("bn4", nn.BatchNorm1d(64)),
                ("fc5", nn.Linear(64, 30)),
            ]
        )
    )

    # load weights
    recycle_net.load_state_dict(
        torch.load(
            "api/recyclenet18_model.pth",
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    )
    recycle_net.to(device)
    recycle_net.eval()

    # preprocess image
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # perform inference
    with torch.no_grad():
        output = recycle_net(input_tensor)

    # convert logits to probabilities
    logits_to_probablities = torch.nn.functional.softmax(output[0], dim=0)

    # get index of predicted value
    class_idx = torch.argmax(logits_to_probablities).item()

    return class_idx, class_map[class_idx], logits_to_probablities[class_idx].item()
