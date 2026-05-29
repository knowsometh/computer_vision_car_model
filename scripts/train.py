import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.models import resnet50, ResNet50_Weights

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# preprocessing + Augmentation
IMG_SIZE = 224  # for ResNet50
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

val_test_transforms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# load datasets
train_dataset = datasets.ImageFolder('../data/train', transform=train_transforms)
val_dataset   = datasets.ImageFolder('../data/val', transform=val_test_transforms)     # ✅ fixed
test_dataset  = datasets.ImageFolder('../data/test', transform=val_test_transforms)    # ✅ fixed

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

dataloaders = {"train": train_loader, "val": val_loader}

# visualise augmented images
def show_batch(loader):
    images, labels = next(iter(loader))
    grid = torchvision.utils.make_grid(images[:8], nrow=4, normalize=True)
    plt.figure(figsize=(8,4))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis('off')
    plt.title("Example Augmented Images")
    plt.show()

show_batch(train_loader)

# set up ResNet50
# Use the official new syntax
weights = ResNet50_Weights.DEFAULT  # pretrained weights
model = resnet50(weights=weights)

# freeze early layers
for param in model.parameters():
    param.requires_grad = False

# unfreeze the last block + final layer
for param in model.layer4.parameters():
    param.requires_grad = True

# replace the last layer to match number of classes
num_classes = len(train_dataset.classes)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

model = model.to(device)
print(model.fc)

# loss + optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001,
    momentum=0.9
)

# training loop
def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'train':
                train_acc_history.append(epoch_acc.cpu().numpy())
            else:
                val_acc_history.append(epoch_acc.cpu().numpy())

    # showing training curve
    plt.figure()
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(val_acc_history, label='Val Acc')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    return model

model = train_model(model, dataloaders, criterion, optimizer, num_epochs=5)

# save model
os.makedirs('../outputs/models', exist_ok=True)  # ensure folder exists
torch.save(model.state_dict(), '../outputs/models/resnet50_best.pth')

print("Model saved successfully!")
