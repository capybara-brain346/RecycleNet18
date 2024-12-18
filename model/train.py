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
# from torchsummary import summary

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.basicConfig(
    filename="./logs/training.log",
    encoding="utf-8",
    level=logging.DEBUG,
    filemode="a",
)
logger = logging.getLogger(__name__)


def training_loop(
    model, loss_func, optimizer, epochs, dataloader, device, logger_object=logger
):
    """
    Execute the training loop for a given model.

    Args:
        model: The model object to be trained.
        loss_func: The loss function used to compute the model's loss.
        optimizer: The optimizer used to update the model's weights.
        epochs (int): The number of epochs to train the model.
        dataloader: The DataLoader providing the training data.
        device: The device on which to perform training (e.g., 'cuda' or 'cpu').
        logger: The logger object for logging training progress.

    Returns:
        The trained model.
    """

    logger_object.info("Starting training loop...")
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
            logger_object.info(
                f"Epoch: {epoch} Loss: {epoch_loss} Accuracy: {epoch_acc}"
            )

        print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    logger_object.info("Training loop completed!")
    return model


def validate(model, dataset, device, loss_func, logger_object=logger):
    """
    Validate the model on validation dataset.

    Args:
        model: The neural network model to be validated.
        dataset: The dataset to use for validation.
        device: The device on which to perform validation (e.g., 'cuda' or 'cpu').
        loss_func: The loss function used to compute the model's loss.
        logger: The logger object for logging validation results.

    Returns:
        None
    """

    logger_object.info("Starting validation...")
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

    logger_object.info(f"Validation accuracy: {accuracy:.2f} %")
    logger_object.info(f"Validation loss: {average_loss:.4f}")
    logger_object.info("Validation completed!")
    print(f"Validation accuracy: {accuracy:.2f} %")
    print(f"Validation loss: {average_loss:.4f}")


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on GPU..." if torch.cuda.is_available() else "Running on CPU...")

    BATCH_SIZE: int = 64
    EPOCHS: int = 20
    LEARNING_RATE: float = 0.001
    # BATCH_NORM_MEAN: list[float] = [0.485, 0.456, 0.406]
    # BATCH_NORM_STD: list[float] = [0.229, 0.224, 0.225]

    DATA_TRANSFORMS: dict = {
        "train": transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
        "validation": transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
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
                ("fc1", nn.Linear(num_features, 512)),
                ("relu1", nn.ReLU()),
                ("bn1", nn.BatchNorm1d(512)),
                # ("dropout1", nn.Dropout(0.3)),
                ("fc2", nn.Linear(512, 256)),
                ("relu2", nn.ReLU()),
                ("bn2", nn.BatchNorm1d(256)),
                # ("dropout2", nn.Dropout(0.3)),
                ("fc3", nn.Linear(256, 128)),
                ("relu3", nn.ReLU()),
                ("bn3", nn.BatchNorm1d(128)),
                # ("dropout3", nn.Dropout(0.2)),
                ("fc4", nn.Linear(128, 64)),
                ("relu4", nn.ReLU()),
                ("bn4", nn.BatchNorm1d(64)),
                ("fc5", nn.Linear(64, 30)),
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

    torch.save(trained_model.state_dict(), "./saved_models/18_11_2024_2.pth")


if __name__ == "__main__":
    main()
