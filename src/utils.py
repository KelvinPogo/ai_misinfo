import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Image preprocessing transforms for ResNet50/MobileNetV3 (ImageNet pre-trained)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 (ImageNet standard)
    transforms.ToTensor(),          # Convert to tensor (scales to [0,1])
    transforms.Normalize(           # Normalize with ImageNet mean and std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom dataset for deepfake detection
        Assumes directory structure:
        root_dir/
            real/     # Real images
            fake/     # Fake/deepfake images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['real', 'fake']  # 0: real, 1: fake
        self.image_paths = []
        self.labels = []

        # Collect all image paths and labels
        for label, class_name in enumerate(self.classes): # (0, 'real'), (1, 'fake')
            class_dir = os.path.join(root_dir, class_name) # e.g., data/train/real and data/train/fake
            if os.path.exists(class_dir): # Check if directory exists
                for img_name in os.listdir(class_dir): # Iterate over images
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')): # Check for valid image extensions
                        self.image_paths.append(os.path.join(class_dir, img_name)) # Full image path
                        self.labels.append(label) # Corresponding label (0 or 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def get_loaders(batch_size=32, data_dir='data'):
    """
    Create train and validation data loaders
    Assumes data is organized as:
    data_dir/
        train/
            real/
            fake/
        val/
            real/
            fake/
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # Create datasets
    train_dataset = DeepfakeDataset(train_dir, transform=transform)
    val_dataset = DeepfakeDataset(val_dir, transform=transform)

    # train_dataset = DeepfakeDataset(root_dir="data/train", transform=transform)
    # val_dataset = DeepfakeDataset(root_dir="data/test", transform=transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Adjust based on your CPU cores
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader
