import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLES_PER_IMAGE = 8
EPOCHS = 15
LR = 0.001

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

augment_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=12),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_path = Path(DATA_DIR)
if not data_path.exists():
    raise FileNotFoundError(f"No se encontro la carpeta '{DATA_DIR}'.")

classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

train_tensors = []
train_labels = []

print(f"Clases detectadas en '{DATA_DIR}': {classes}")

for cls_name in classes:
    cls_folder = data_path / cls_name
    files = [f for f in cls_folder.iterdir() if f.suffix.lower() in valid_extensions]
    print(f"-> Clase '{cls_name}': {len(files)} imagenes.")
    
    idx = class_to_idx[cls_name]
    for file_path in files:
        img = Image.open(file_path).convert("RGB")
        train_tensors.append(base_transform(img))
        train_labels.append(idx)
        for _ in range(SAMPLES_PER_IMAGE):
            train_tensors.append(augment_transform(img))
            train_labels.append(idx)

class MemoryDataset(Dataset):
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]

dataset = MemoryDataset(train_tensors, train_labels)
loader = DataLoader(dataset, batch_size=8, shuffle=True)
print(f"Total de imagenes con aumentacion: {len(dataset)}")

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features, len(classes))
)

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR, weight_decay=1e-3)

model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == targets.data)
        total += targets.size(0)

    acc = correct.double() / total
    print(f"Epoca {epoch+1:02d}/{EPOCHS} - Acc: {acc:.4f}")

torch.save(model.state_dict(), "model.pth")
print("\nLISTO: Archivo 'model.pth' generado con exito.")
