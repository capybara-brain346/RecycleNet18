import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from custom_data_loader import ImageLoader
from typing import OrderedDict
from config import Config
import logging
import datetime
import tqdm
from torchsummary import summary

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.basicConfig(
    filename="./logs/training.log",
    encoding="utf-8",
    level=logging.DEBUG,
    filemode="a",
)
logger = logging.getLogger(__name__)


def training_loop(model, loss_func, optimizer, epochs, dataloader, device):
    logger.info("Starting training loop...")
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs - 1}")
        print("-" * 10)
        running_loss = 0.0
        running_correct = 0

        for images, labels in tqdm.tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            running_correct += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_correct.double() / len(dataloader.dataset)

        if epoch % 5 == 0:
            logger.info(f"Epoch: {epoch} Loss: {epoch_loss} Accuracy: {epoch_acc}")

        print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    logger.info("Training loop completed!")
    return model


def validate(model, dataset, device, loss_func):
    logger.info("Starting validation...")
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in dataset:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(preds == labels.data)

    accuracy = 100 * correct / total
    average_loss = running_loss / total

    logger.info(f"Validation accuracy: {accuracy:.2f} %")
    logger.info(f"Validation loss: {average_loss:.4f}")
    logger.info("Validation completed!")
    print(f"Validation accuracy: {accuracy:.2f} %")
    print(f"Validation loss: {average_loss:.4f}")


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on GPU..." if torch.cuda.is_available() else "Running on CPU...")

    BATCH_SIZE: int = 64
    EPOCHS: int = 20
    LEARNING_RATE: float = 0.001
    BATCH_NORM_MEAN: list[float] = [0.485, 0.456, 0.406]
    BATCH_NORM_STD: list[float] = [0.229, 0.224, 0.225]
    DATA_TRANSFORMS: dict = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=BATCH_NORM_MEAN,
                    std=BATCH_NORM_STD,
                ),
            ]
        ),
        "validation": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=BATCH_NORM_MEAN, std=BATCH_NORM_STD),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=BATCH_NORM_MEAN, std=BATCH_NORM_STD),
            ]
        ),
    }

    logger.info(f"Started training run at {datetime.datetime.now()}")
    logger.info("Loading images...")

    train_dataset = ImageLoader(
        path=Config.DATA_BASE_DIR,
        data_transform=DATA_TRANSFORMS["train"],
        split="train",
    )
    train_loader = DataLoader(
        dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True
    )

    validation_dataset = ImageLoader(
        path=Config.DATA_BASE_DIR,
        data_transform=DATA_TRANSFORMS["validation"],
        split="validation",
    )
    validation_loader = DataLoader(
        dataset=validation_dataset, shuffle=False, batch_size=BATCH_SIZE
    )

    logger.info(
        f"Loaded {len(train_loader.dataset)} training images and {len(validation_loader.dataset)} validation images"
    )

    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    logger.info("Loaded ResNet18 with DEFAULT weights")

    for param in model.parameters():
        param.requires_grad = False

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
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    trained_model = training_loop(
        model=model,
        loss_func=loss_function,
        optimizer=optimizer,
        epochs=EPOCHS,
        dataloader=train_loader,
        device=device,
    )

    validate(
        model=trained_model,
        dataset=validation_loader,
        device=device,
        loss_func=loss_function,
    )

    torch.save(trained_model.state_dict(), "./saved_models/02_7_24_resnet_model_3.pth")


if __name__ == "__main__":
    main()
