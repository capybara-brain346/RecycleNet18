"""
Run the main function to perform model evaluation on the test dataset.

Args:
    None

Returns:
    None
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from custom_data_loader import ImageLoader
from typing import OrderedDict
from config import Config
import logging
import tqdm

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.basicConfig(
    filename="./logs/testing.log",
    encoding="utf-8",
    level=logging.DEBUG,
    filemode="a",
)
logger = logging.getLogger(__name__)


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on GPU..." if torch.cuda.is_available() else "Running on CPU...")

    BATCH_NORM_MEAN: list[float] = [0.485, 0.456, 0.406]
    BATCH_NORM_STD: list[float] = [0.229, 0.224, 0.225]
    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=BATCH_NORM_MEAN, std=BATCH_NORM_STD),
        ]
    )

    test_dataset = ImageLoader(
        path=Config.DATA_BASE_DIR, data_transform=test_transforms, split="test"
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(num_features, 256)),
                ("relu1", nn.ReLU()),
                ("fc2", nn.Linear(256, 30)),
            ]
        )
    )
    # print(model)
    model.load_state_dict(torch.load("./saved_models/02_7_24_resnet_model_3.pth"))
    model.to(device)

    loss_func = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(preds == labels.data)

    test_accuracy = 100 * correct / total
    average_loss = running_loss / total
    print(f"Validation accuracy: {test_accuracy:.2f} %")
    print(f"Validation loss: {average_loss:.4f}")


if __name__ == "__main__":
    main()
