import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader, random_split
from custom_data_loader import ImageLoader
from torchsummary import summary
from config import Config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs - 1}")
        print("-" * 10)
        model.train()
        running_loss = 0.0
        running_correct = 0

        for images, labels in dataloader:
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

        print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    print("Training complete")
    return model


def validate(model, dataset):
    model.eval()
    correct = 0
    total = 0
    for images, labels in dataset:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Validation accuracy: {} %".format(100 * correct / total))


data = ImageLoader(path=Config.DATA_BASE_DIR, data_transform=DATA_TRANSFORMS["train"])
train_dataset, val_dataset = random_split(data, (10000, 5000))
train_loader = DataLoader(
    dataset=train_dataset, shuffle=True, batch_size=64, drop_last=True
)
val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=64)


model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 30)
model = model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)

training_loop(
    model=model,
    loss_func=loss_function,
    optimizer=optimizer,
    epochs=10,
    dataloader=train_loader,
)
