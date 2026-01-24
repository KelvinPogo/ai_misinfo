import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import resnet, mobilenet
from utils import get_loaders
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import random
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Hyperparameters
# -----------------------------
batch_size = 32
learning_rate = 1e-4
num_epochs = 10
seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Data Loaders
train_loader, val_loader = get_loaders(batch_size=batch_size)


# Model, Loss, Optimizer
model = resnet.to(device) 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training Loop
for epoch in range(num_epochs):
    model.train()
    train_losses = []
    all_preds = []
    all_labels = []

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # clears accumulated gradients from previous training steps
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Epoch {epoch+1} Train Loss: {np.mean(train_losses):.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")

    # Validation
    model.eval()
    val_losses = []
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())

            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    print(f"Epoch {epoch+1} Val Loss: {np.mean(val_losses):.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")


