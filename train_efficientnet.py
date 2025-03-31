import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torchvision import models, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import pyarrow.parquet as pq
from PIL import Image
import cv2
from tqdm import tqdm
import random
import time
import copy

# Set random seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check CUDA version
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # Dynamically set memory fraction based on GPU total memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory < 6:  # Less than 6GB (like RTX 2050 with 4GB)
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of GPU memory
        print(f"Limited GPU memory usage to 70% for smaller GPUs")
    else:
        torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of GPU memory for larger GPUs
    torch.cuda.empty_cache()
else:
    print("CUDA is not available. Training will proceed on CPU, which may be slow.")
    print("Mixed precision training will be disabled.")

# Flag to check if we should use mixed precision
use_mixed_precision = torch.cuda.is_available()

# Data preprocessing
class SpectrogramDataset(Dataset):
    def __init__(self, csv_file, spectrogram_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            spectrogram_dir (string): Directory with all the spectrogram parquet files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.spectrogram_dir = spectrogram_dir
        self.transform = transform
        
        # Map class labels to integers
        self.class_map = {
            'Seizure': 0,
            'GPD': 1,
            'LRDA': 2,
            'Other': 3,
            'GRDA': 4,
            'LPD': 5
        }
        
        # Create a mapping from spectrogram_id to file path
        self.spectrogram_files = {}
        for file_name in os.listdir(spectrogram_dir):
            if file_name.endswith('.parquet'):
                spectrogram_id = int(file_name.split('.')[0])
                self.spectrogram_files[spectrogram_id] = os.path.join(spectrogram_dir, file_name)
        
        # Filter dataframe to include only records with available spectrogram files
        self.data_frame = self.data_frame[self.data_frame['spectrogram_id'].isin(self.spectrogram_files.keys())]
        
        print(f"Dataset initialized with {len(self.data_frame)} samples")
        print(f"Class distribution: {self.data_frame['expert_consensus'].value_counts()}")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get spectrogram ID and file path
        spectrogram_id = self.data_frame.iloc[idx]['spectrogram_id']
        spectrogram_path = self.spectrogram_files[spectrogram_id]
        
        # Load spectrogram from parquet file
        try:
            # Read parquet file
            table = pq.read_table(spectrogram_path)
            df = table.to_pandas()
            
            # Convert to numpy array and reshape to image dimensions
            spectrogram_array = df.values
            
            # Create grayscale image from spectrogram array
            # Normalize to 0-255 range
            img_min = spectrogram_array.min()
            img_max = spectrogram_array.max()
            if img_max > img_min:  # Avoid division by zero
                spectrogram_array = ((spectrogram_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                spectrogram_array = np.zeros_like(spectrogram_array, dtype=np.uint8)
            
            # Convert to PIL Image (grayscale)
            img = Image.fromarray(spectrogram_array)
            
            # Resize to 300x300
            img = img.resize((300, 300), Image.LANCZOS)
            
            # Convert grayscale to RGB (3 channels) by duplicating the single channel
            img_rgb = Image.new("RGB", img.size)
            img_rgb.paste(img)
            
            # Apply transformations
            if self.transform:
                img_rgb = self.transform(img_rgb)
            
            # Get label
            label = self.class_map[self.data_frame.iloc[idx]['expert_consensus']]
            
            return img_rgb, label
            
        except Exception as e:
            print(f"Error loading spectrogram {spectrogram_id}: {e}")
            # Return a black image and the most common class as fallback
            img = torch.zeros(3, 300, 300)
            label = 0  # Default to most common class
            return img, label

# Define data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ]),
}

# Create dataset
dataset = SpectrogramDataset(
    csv_file='train.csv',
    spectrogram_dir='train_spectrograms',
    transform=data_transforms['train']
)

# Split into train and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Apply different transforms to validation set
val_dataset.dataset.transform = data_transforms['val']

# Create data loaders - optimize based on GPU memory
num_workers = 2 if device.type == 'cuda' else 0

# Set batch size based on available GPU memory
if device.type == 'cuda':
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory < 6:  # Less than 6GB
        batch_size = 8  # Smaller batch size for RTX 2050 4GB
    elif gpu_memory < 12:  # 8GB GPUs
        batch_size = 16
    else:  # 12GB+ GPUs
        batch_size = 32
else:
    batch_size = 8  # Default for CPU

print(f"Using batch size of {batch_size} based on available memory")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == 'cuda'))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == 'cuda'))

# Model architecture
def create_model(num_classes=6):
    # Load pre-trained EfficientNetB3
    model = models.efficientnet_b3(weights='IMAGENET1K_V1')
    
    # Modify classifier for our number of classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes)
    )
    
    return model

model = create_model()
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# Initialize GradScaler for mixed precision training
scaler = GradScaler() if use_mixed_precision else None
if use_mixed_precision:
    print("Using mixed precision training with GradScaler")
else:
    print("Mixed precision training disabled")

# Training and evaluation functions
def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # For early stopping
    patience = 5
    counter = 0
    min_val_loss = float('inf')
    
    # For plotting
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    # For full dataset training
    print(f"Training with ALL {len(train_loader.dataset)} samples, validating with ALL {len(val_loader.dataset)} samples")
    
    # Checkpoint directory
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Resume from checkpoint if available
    start_epoch = 0
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            train_loss_history = checkpoint['train_loss_history']
            val_loss_history = checkpoint['val_loss_history']
            train_acc_history = checkpoint['train_acc_history']
            val_acc_history = checkpoint['val_acc_history']
            print(f"Resuming training from epoch {start_epoch+1}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
    
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            
            # Iterate over data
            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                # Track history only if in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    if use_mixed_precision and phase == 'train':
                        # Use mixed precision for forward pass in training
                        with autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        
                        _, preds = torch.max(outputs, 1)
                        
                        # Backward pass + optimize only in training phase
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Regular forward pass without mixed precision
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        _, preds = torch.max(outputs, 1)
                        
                        # Backward pass + optimize only in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Store predictions and labels for F1 score calculation
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Clear GPU cache periodically to prevent OOM
                if device.type == 'cuda' and phase == 'train' and (len(all_preds) % 1000 == 0):
                    torch.cuda.empty_cache()
            
            if phase == 'train' and scheduler is not None:
                scheduler.step()
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}')
            
            # Store history for plotting
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.cpu().numpy())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.cpu().numpy())
                
                # Early stopping check
                if epoch_loss < min_val_loss:
                    min_val_loss = epoch_loss
                    counter = 0
                    # Save best model
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(model.state_dict(), 'best_model.pth')
                        print(f'New best model saved with accuracy: {best_acc:.4f}')
                else:
                    counter += 1
                    print(f'EarlyStopping counter: {counter} out of {patience}')
                    if counter >= patience:
                        print('Early stopping')
                        # Load best model weights
                        model.load_state_dict(best_model_wts)
                        
                        # Plot training curves
                        plot_training_curves(train_loss_history, val_loss_history, 
                                             train_acc_history, val_acc_history)
                        
                        time_elapsed = time.time() - since
                        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                        print(f'Best val Acc: {best_acc:.4f}')
                        
                        return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history
        
        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint.pth'))
        print(f"Checkpoint saved after epoch {epoch+1}")
        
        print()
        
        # Clear GPU cache after each epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
    # After all epochs
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Plot training curves
    plot_training_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history)
    
    # Save the final model
    torch.save(model.state_dict(), 'final_model.pth')
    print("Final model saved as 'final_model.pth'")
    
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

def plot_training_curves(train_loss, val_loss, train_acc, val_acc):
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

def evaluate_model(model, dataloader):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Class names for reference
    class_names = ['Seizure', 'GPD', 'LRDA', 'Other', 'GRDA', 'LPD']
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    print(f'Test Accuracy: {acc:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    
    return acc, f1, cm

def visualize_model_predictions(model, dataloader, num_images=6):
    model.eval()
    
    class_names = ['Seizure', 'GPD', 'LRDA', 'Other', 'GRDA', 'LPD']
    
    # Get a batch of images
    images, labels = next(iter(dataloader))
    images = images.to(device)
    labels = labels.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Plot the images
    fig = plt.figure(figsize=(15, 12))
    
    for i in range(min(num_images, len(images))):
        ax = fig.add_subplot(2, 3, i + 1, xticks=[], yticks=[])
        
        # Convert tensor to numpy array for visualization
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        # Add colored labels: green for correct, red for incorrect
        ax.set_title(f'True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}',
                    color=("green" if preds[i] == labels[i] else "red"))
    
    plt.tight_layout()
    plt.savefig('model_predictions.png')
    plt.show()

def load_test_data():
    """Load and prepare test data if available"""
    test_csv = 'test.csv'
    test_dir = 'test_spectrograms'
    
    if os.path.exists(test_csv) and os.path.exists(test_dir):
        print(f"Found test data: {test_csv} and {test_dir}")
        test_dataset = SpectrogramDataset(
            csv_file=test_csv,
            spectrogram_dir=test_dir,
            transform=data_transforms['val']  # Use validation transforms for test data
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=(device.type == 'cuda')
        )
        
        print(f"Test dataset loaded with {len(test_dataset)} samples")
        return test_loader
    else:
        print("Test data not found. Skipping test evaluation.")
        return None

if __name__ == "__main__":
    # Train the model
    print("Starting model training...")
    # Set a reasonable epoch count based on available hardware
    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory < 6:  # Less than 6GB
            num_epochs = 15
        else:
            num_epochs = 25
    else:
        num_epochs = 5
    
    print(f"Training for {num_epochs} epochs")
    
    try:
        model, train_loss, val_loss, train_acc, val_acc = train_model(
            model, criterion, optimizer, scheduler, num_epochs=num_epochs
        )
        
        # Evaluate on validation set
        print("Evaluating model on validation set...")
        val_acc, val_f1, val_cm = evaluate_model(model, val_loader)
        
        # Load and evaluate on test set if available
        test_loader = load_test_data()
        if test_loader is not None:
            print("Evaluating model on test set...")
            test_acc, test_f1, test_cm = evaluate_model(model, test_loader)
            print(f"Test Accuracy: {test_acc:.4f}, Test F1 Score: {test_f1:.4f}")
        
        # Visualize model predictions
        print("Visualizing model predictions...")
        visualize_model_predictions(model, val_loader)
        
        print("Training completed!")
        print(f"Best model saved to 'best_model.pth'")
        print(f"Final model saved to 'final_model.pth'")
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print('🛑 CUDA out of memory error! Try reducing batch size or using a smaller dataset.')
            # Free up GPU memory and try again with smaller batch size
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                print("Cleared GPU cache. Consider adjusting these parameters in the script:")
                print("1. Reduce batch_size (currently", batch_size, ")")
                print("2. Reduce the image resolution (currently 300x300)")
                print("3. Try a smaller model like efficientnet_b0 or efficientnet_b1")
        else:
            print(f"Runtime error: {e}")
    except Exception as e:
        print(f"Error during training: {e}") 