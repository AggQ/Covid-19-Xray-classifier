import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths and parameters
DATA_DIR = "./data"  # Create this directory and place your X-ray images here
BATCH_SIZE = 16
NUM_EPOCHS = 20
IMAGE_SIZE = 224
NUM_CLASSES = 3  # Normal, Viral Pneumonia, COVID-19

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Data transformation for training and validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Custom Dataset
class ChestXRayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Function to prepare data
def prepare_data(data_dir):
    # This is a placeholder. In a real app, you would scan your data directory
    # and create a DataFrame with image paths and labels
    print("Preparing data...")
    
    # Example structure (replace with your actual data loading logic)
    data = {
        'path': [],
        'label': []
    }
    
    classes = ['normal', 'viral', 'covid']
    class_dirs = [os.path.join(data_dir, cls) for cls in classes]
    
    for i, class_dir in enumerate(class_dirs):
        if os.path.exists(class_dir):
            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                    if os.path.isfile(os.path.join(class_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg'))]
            data['path'].extend(files)
            data['label'].extend([i] * len(files))
    
    df = pd.DataFrame(data)
    print(f"Found {len(df)} images across {len(classes)} classes")
    
    # Split data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    return train_df, val_df

# Function to create data loaders
def create_data_loaders(train_df, val_df):
    print("Creating data loaders...")
    
    train_dataset = ChestXRayDataset(
        train_df['path'].values,
        train_df['label'].values,
        transform=data_transforms['train']
    )
    
    val_dataset = ChestXRayDataset(
        val_df['path'].values,
        val_df['label'].values,
        transform=data_transforms['val']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

# Model definition (using ResNet18 with transfer learning)
def create_model():
    print("Creating model...")
    
    model = models.resnet18(pretrained=True)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, NUM_CLASSES)
    )
    
    model = model.to(device)
    return model

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    print("Starting training...")
    
    train_losses = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f}")
        print(f"Val Acc: {epoch_val_acc:.4f}")
        print("-" * 20)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    
    return model

# Function to save model
def save_model(model, path="covid_xray_model.pth"):
    print(f"Saving model to {path}...")
    torch.save(model.state_dict(), path)

# Function to load model
def load_model(path="covid_xray_model.pth"):
    print(f"Loading model from {path}...")
    model = create_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# Function to predict on new images
def predict(model, image_path):
    image = Image.open(image_path).convert("RGB")
    transform = data_transforms['val']
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        
    class_names = ['Normal', 'Viral Pneumonia', 'COVID-19']
    return class_names[pred.item()]

# Function to visualize predictions on multiple images
def show_preds(model, image_paths):
    class_names = ['Normal', 'Viral Pneumonia', 'COVID-19']
    
    plt.figure(figsize=(15, 10))
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        
        # Transform for model input
        transform = data_transforms['val']
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            predicted_class = class_names[pred.item()]
        
        # Display image with prediction
        plt.subplot(2, 3, i+1)
        plt.imshow(image)
        plt.title(f'Pred: {predicted_class}')
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

# Main function to run the application
def main():
    print("COVID-19 X-Ray Classification with PyTorch")
    
    # Prepare data
    train_df, val_df = prepare_data(DATA_DIR)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_df, val_df)
    
    # Create model
    model = create_model()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # Train model
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)
    
    # Save model
    save_model(trained_model)
    
    print("Training complete!")

if __name__ == "__main__":
    main()