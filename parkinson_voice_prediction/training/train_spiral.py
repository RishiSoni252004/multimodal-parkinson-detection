import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def train_spiral_model(data_dir="dataset/spiral", epochs=20, batch_size=32, save_path="models/spiral_model.pth"):
    """
    Trains a ResNet model on spiral drawings.
    Expects directory structure:
        dataset/spiral/
            healthy/
                hr_img1.png
                ...
            parkinson/
                pr_img1.png
                ...
    """
    if not os.path.exists(data_dir):
        print(f"Dataset directory {data_dir} not found. Please create it and add 'healthy' and 'parkinson' folders.")
        return

    healthy_dir = os.path.join(data_dir, "healthy")
    parkinson_dir = os.path.join(data_dir, "parkinson")
    
    if not os.path.exists(healthy_dir) or not os.path.exists(parkinson_dir):
         print(f"Please ensure {healthy_dir} and {parkinson_dir} exist and contain images.")
         return

    print("Setting up data transforms and loaders...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # MPS (Apple Silicon) > CPU. Never CUDA.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 2)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / (total if total > 0 else 1)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully to {save_path}")

if __name__ == "__main__":
    train_spiral_model()
