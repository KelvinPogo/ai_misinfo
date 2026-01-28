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

def load_latest_checkpoint(model, optimizer, checkpoint_dir='checkpoints'):
    """Load the most recent checkpoint if it exists"""
    if not os.path.exists(checkpoint_dir):
        print("No checkpoint directory found. Starting fresh training.")
        return 0
    
    # Find the latest checkpoint
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
    if not checkpoints:
        print("No checkpoints found. Starting fresh training.")
        return 0
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    
    print(f"Loading checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    print(f"Resumed from epoch {epoch} with train_loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    return epoch

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

if __name__ == '__main__':
    # Data Loaders
    train_loader, val_loader = get_loaders(batch_size=batch_size)

    # Model, Loss, Optimizer
    model = resnet.to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load checkpoint if exists
    start_epoch = load_latest_checkpoint(model, optimizer)

    # Training Loop
    for epoch in range(start_epoch, num_epochs):
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
        

        # Save model checkpoint
        checkpoint_path = f'checkpoints/model_epoch_{epoch+1}.pth'
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': np.mean(train_losses),
            'val_loss': np.mean(val_losses) if val_losses else float('nan'),
            'train_acc': train_acc,
            'val_acc': val_acc if not np.isnan(val_acc) else 0.0,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_checkpoint_path = 'checkpoints/model_final.pth'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_train_loss': np.mean(train_losses),
        'final_val_loss': np.mean(val_losses) if val_losses else float('nan'),
        'final_train_acc': train_acc,
        'final_val_acc': val_acc if 'val_acc' in locals() and not np.isnan(val_acc) else 0.0,
    }, final_checkpoint_path)
    print(f"Final model saved: {final_checkpoint_path}")
    print("Training completed!")


