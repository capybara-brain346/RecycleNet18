import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, random_split
from custom_data_loader import ImageLoader
from config import Config
import logging
import datetime
import tqdm
# from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.basicConfig(
    filename="./logs/training.log", encoding="utf-8", level=logging.DEBUG, filemode="a"
)
logger = logging.getLogger(__name__)


DATA_TRANSFORMS = {
    "train": transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.7485, 0.7274, 0.7051], std=[0.2481, 0.2566, 0.2747]
            ),
        ]
    ),
    "validate": transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.7485, 0.7274, 0.7051], std=[0.2481, 0.2566, 0.2747]
            ),
        ]
    ),
}


def training_loop(model, loss_func, optimizer, epochs, dataloader):
    logger.info("Starting training loop...")
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs - 1}")
        print("-" * 10)
        model.train()
        running_loss = 0.0
        running_correct = 0

        for images, labels in tqdm.tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            idx, preds = torch.max(outputs, 1)
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


def validate(model, dataset, device):
    logger.info("Starting validation...")
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataset:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            idx, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(preds == labels.data)

    accuracy = 100 * correct / total
    average_loss = running_loss / total

    logger.info(f"Validation accuracy: {accuracy:.2f} %")
    logger.info(f"Validation loss: {average_loss:.4f}")
    logger.info("Validation completed!")
    print(f"Validation accuracy: {accuracy:.2f} %")
    print(f"Validation loss: {average_loss:.4f}")


logger.info(f"\nStarted training run at {datetime.datetime.now()}")
logger.info("Loading images...")
data = ImageLoader(path=Config.DATA_BASE_DIR, data_transform=DATA_TRANSFORMS["train"])
train_dataset, val_dataset = random_split(data, (10000, 5000))
train_loader = DataLoader(
    dataset=train_dataset, shuffle=True, batch_size=64, drop_last=True
)
val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=64)
logger.info(
    f"Loaded {len(train_loader.dataset)} training images and {len(val_loader.dataset)} validation images"
)


model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
logger.info("Loaded ResNet18 with DEFAULT weights")

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 30)
model = model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

trained_model = training_loop(
    model=model,
    loss_func=loss_function,
    optimizer=optimizer,
    epochs=20,
    dataloader=train_loader,
)

validate(model=trained_model, dataset=val_loader, device=device)
torch.save(trained_model.state_dict(), "./saved_models/30_6_24_resnet_model.pth")
