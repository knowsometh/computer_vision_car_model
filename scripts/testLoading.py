from torchvision import datasets, transforms

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder('../data/train', transform=train_transforms)
print(f"Classes: {len(train_dataset.classes)}")
print(f"Samples: {len(train_dataset)}")
print("Example classes:", train_dataset.classes[:5])
