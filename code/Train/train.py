import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models
from torchvision.models import ResNet18_Weights

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from DataUtil.LUNGDataUtil import create_dataloaders

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 6

class SimpleClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SimpleClassifier, self).__init__()
        self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
        validate_model(model, val_loader, device)

    print("训练完成")
    torch.save(model.state_dict(), './Result/simple_classifier_model.pth')

def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    images_dir = 'E:\pyDLW\LUNGC\dataset\images'
    labels_dir = 'E:\pyDLW\LUNGC\dataset\labels'
    batch_size = 32

    train_loader = create_dataloaders(images_dir, labels_dir, 'train', batch_size)
    val_loader = create_dataloaders(images_dir, labels_dir, 'val', batch_size)

    model = SimpleClassifier(num_classes)
    train_model(model, train_loader, val_loader, device, num_epochs=10)
