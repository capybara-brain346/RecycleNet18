from flask import Flask, render_template, request
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from PIL import Image
from typing import OrderedDict, Annotated

app = Flask(__name__)

CLASS_MAP = {
    0: "aerosol cans",
    1: "aluminum food cans",
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


def get_classification(
    image_bytes,
) -> Annotated[tuple[int, float], "Range 0 to 29"] | None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on GPU..." if torch.cuda.is_available() else "Running on CPU...")

    BATCH_NORM_MEAN: list[float] = [0.485, 0.456, 0.406]
    BATCH_NORM_STD: list[float] = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=BATCH_NORM_MEAN, std=BATCH_NORM_STD),
        ]
    )
    RecycleNet = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    num_features = RecycleNet.fc.in_features
    RecycleNet.fc = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(num_features, 256)),
                ("relu1", nn.ReLU()),
                ("fc2", nn.Linear(256, 30)),
            ]
        )
    )
    RecycleNet.load_state_dict(torch.load("./app/RecycleNet18.pth"))
    RecycleNet.to(device)
    RecycleNet.eval()

    image = Image.open(image_bytes).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = RecycleNet(input_tensor)

    logits_to_probablities = torch.nn.functional.softmax(output[0], dim=0)
    class_idx = torch.argmax(logits_to_probablities).item()
    return class_idx, logits_to_probablities[class_idx].item()


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/upload/")
def upload_image():
    if "file" not in request.files:
        return "File not found"

    predicted_class, class_confidence = get_classification(request.files["file"])
    return render_template("index.html", result=CLASS_MAP[predicted_class].title())


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
