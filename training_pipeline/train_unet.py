"""
U-Net Training Script for Coastline Segmentation

Trains a U-Net model for coastline segmentation on satellite imagery.
Automatically pairs images with masks and uses configurable parameters.

Usage: python train_unet.py

Configuration: All parameters in config_template.json
Data: Images and masks in results_augment_tiles folder
Naming: Images (*_XX-of-YY_[aug].tif) and Masks (*_concatenated_ndwi_mask_XX-of-YY_[aug].tif)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import glob
import re
import json
import pickle
from datetime import datetime

# Add parent directory to path to import load_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_config import load_config, get_augment_tiles_output_folder, get_training_config, get_model_save_path

# ----------------------------
# Dataset Class
# ----------------------------
class SegmentationDataset(Dataset):
    """
    Dataset for coastline segmentation training.
    
    Automatically pairs satellite images with their corresponding masks
    based on naming convention. Handles augmented data.
    
    Args:
        data_dir (str): Directory containing images and masks
        transform (callable, optional): Transform to apply to data
    """
    
    def __init__(self, data_dir, transform=None):
        """Initialize dataset with data directory and optional transforms."""
        self.data_dir = data_dir
        self.transform = transform
        self.image_mask_pairs = self._find_image_mask_pairs()

    def _find_image_mask_pairs(self):
        """
        Find matching image and mask pairs based on naming convention.
        
        Naming: Images (*_XX-of-YY_[aug].tif) -> Masks (*_concatenated_ndwi_mask_XX-of-YY_[aug].tif)
        Returns: List of (image_path, mask_path) tuples
        """
        pairs = []
        
        # Get all image files (tiles without mask in name)
        image_files = glob.glob(os.path.join(self.data_dir, "*.tif"))
        image_files = [f for f in image_files if "_concatenated_ndwi_mask_" not in os.path.basename(f)]
        
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            
            # Extract the base name and augmentation suffix
            # Pattern: base_name_XX-of-YY_[augmentation].tif
            base_match = re.match(r'(.+)_\d+-of-\d+(_[^_]+)?\.tif$', img_name)
            if base_match:
                base_name = base_match.group(1)
                augmentation = base_match.group(2) if base_match.group(2) else ""
                
                # Construct corresponding mask name
                mask_name = f"{base_name}_concatenated_ndwi_mask_{img_name.split('_')[-2]}_{img_name.split('_')[-1]}"
                mask_path = os.path.join(self.data_dir, mask_name)
                
                if os.path.exists(mask_path):
                    pairs.append((img_path, mask_path))
                else:
                    print(f"Warning: Mask not found for {img_name}")
        
        print(f"Found {len(pairs)} image-mask pairs")
        return pairs

    def __len__(self):
        """Return number of image-mask pairs in dataset."""
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        """
        Get image-mask pair by index.
        
        Returns: (image_tensor, mask_tensor) - RGB image and binary mask tensors
        """
        img_path, mask_path = self.image_mask_pairs[idx]
        
        # Load image and convert to RGB
        image = Image.open(img_path).convert("RGB")
        
        # Load mask and convert to grayscale
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convert mask to binary (0 or 1)
        mask = (mask > 0).float()
        return image, mask

# ----------------------------
# U-Net Model
# ----------------------------
class DoubleConv(nn.Module):
    """
    Double convolution block for U-Net architecture.
    
    Two consecutive 3x3 convolutions with batch normalization and ReLU.
    Used in encoder and decoder paths.
    
    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
    """
    
    def __init__(self, in_channels, out_channels):
        """Initialize double convolution block."""
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward pass through double convolution block."""
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    
    Encoder-decoder structure with skip connections for precise segmentation.
    Architecture: 4 downsampling blocks -> bottleneck -> 4 upsampling blocks -> final layer.
    
    Args:
        n_channels (int): Input channels (default: 3 for RGB)
        n_classes (int): Output classes (default: 1 for binary segmentation)
    """
    
    def __init__(self, n_channels=3, n_classes=1):
        """Initialize U-Net model with encoder-decoder structure."""
        super().__init__()
        # Encoder path
        self.down1 = DoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.middle = DoubleConv(512, 1024)

        # Decoder path with skip connections
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        # Final classification layer
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass through U-Net with skip connections."""
        # Encoder path
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        mid = self.middle(self.pool4(d4))

        # Decoder path with skip connections
        u4 = self.conv4(torch.cat([self.up4(mid), d4], dim=1))
        u3 = self.conv3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), d1], dim=1))

        # Final output with sigmoid activation
        return torch.sigmoid(self.final(u1))

# ----------------------------
# Checkpoint Functions
# ----------------------------
def save_training_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                           best_val_loss, config, checkpoint_path):
    """
    Save training checkpoint with model state and training progress.
    
    Args:
        model: U-Net model to save
        optimizer: Optimizer state
        epoch: Current epoch number
        train_loss: Current training loss
        val_loss: Current validation loss
        best_val_loss: Best validation loss so far
        config: Training configuration
        checkpoint_path: Path to save checkpoint
    """
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

def load_training_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load training checkpoint to resume training.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: U-Net model to load state into
        optimizer: Optimizer to load state into (optional)
    
    Returns:
        tuple: (epoch, train_loss, val_loss, best_val_loss, config) or None if not found
    """
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        return (checkpoint['epoch'], 
                checkpoint['train_loss'], 
                checkpoint['val_loss'], 
                checkpoint['best_val_loss'], 
                checkpoint['config'])
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

def get_checkpoint_path(model_save_path, epoch=None):
    """
    Generate checkpoint file path.
    
    Args:
        model_save_path: Base model save path
        epoch: Epoch number for specific checkpoint (optional)
    
    Returns:
        str: Checkpoint file path
    """
    base_dir = os.path.dirname(model_save_path)
    base_name = os.path.splitext(os.path.basename(model_save_path))[0]
    
    if epoch is not None:
        return os.path.join(base_dir, f"{base_name}_checkpoint_epoch_{epoch}.pth")
    else:
        return os.path.join(base_dir, f"{base_name}_checkpoint_latest.pth")

# ----------------------------
# Training Loop
# ----------------------------
def train_model(model, train_loader, val_loader, config, model_save_path, resume_from_checkpoint=True):
    """
    Train U-Net model for coastline segmentation with checkpoint support.
    
    Implements complete training loop with forward pass, loss computation,
    backpropagation, and validation. Uses configurable parameters and progress tracking.
    Supports resuming from checkpoints and saving best model.
    
    Args:
        model (UNet): U-Net model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        config (dict): Configuration with training parameters
        model_save_path (str): Path to save trained model
        resume_from_checkpoint (bool): Whether to resume from checkpoint if available
    """
    training_config = get_training_config(config)
    epochs = training_config.get('epochs', 30)
    lr = training_config.get('learning_rate', 1e-4)
    device = training_config.get('device', 'auto')
    save_every_n_epochs = training_config.get('save_every_n_epochs', 5)
    
    # Set device
    if device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    print(f"Training on device: {device}")
    
    # Initialize training variables
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Try to load checkpoint if resume is enabled
    checkpoint_path = get_checkpoint_path(model_save_path)
    if resume_from_checkpoint:
        checkpoint_data = load_training_checkpoint(checkpoint_path, model, optimizer)
        if checkpoint_data is not None:
            start_epoch, _, _, best_val_loss, _ = checkpoint_data
            start_epoch += 1  # Start from next epoch
            print(f"Resuming training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        # Calculate average losses
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # Save checkpoint every N epochs
        if (epoch + 1) % save_every_n_epochs == 0:
            save_training_checkpoint(model, optimizer, epoch, avg_train_loss, 
                                   avg_val_loss, best_val_loss, config, checkpoint_path)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved! Val Loss: {avg_val_loss:.4f}")

    # Final checkpoint save
    save_training_checkpoint(model, optimizer, epochs-1, avg_train_loss, 
                           avg_val_loss, best_val_loss, config, checkpoint_path)
    
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to {model_save_path}")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    """
    Main execution block for U-Net training.
    
    Handles complete training pipeline: config loading, dataset creation,
    train/val split, model training, and saving. Auto-detects GPU/CPU.
    """
    # Load configuration
    config = load_config()
    training_config = get_training_config(config)
    
    # Get paths from config
    data_dir = get_augment_tiles_output_folder(config)
    model_save_path = get_model_save_path(config)
    
    # Get training parameters from config
    image_size = training_config.get('image_size', [256, 256])
    batch_size = training_config.get('batch_size', 8)
    train_split = training_config.get('train_split', 0.8)
    
    print(f"Data directory: {data_dir}")
    print(f"Model save path: {model_save_path}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Train split: {train_split}")
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    # Create dataset
    dataset = SegmentationDataset(data_dir, transform=transform)
    
    if len(dataset) == 0:
        print("No image-mask pairs found! Please check your data directory and file naming.")
        exit(1)

    # Split into train and validation sets
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    print(f"Dataset split: {train_size} training, {val_size} validation samples")

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Create and train model
    model = UNet(n_channels=3, n_classes=1)
    train_model(model, train_loader, val_loader, config, model_save_path)
