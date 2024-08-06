import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import sys
import os
# 将目录加载到系统环境中，其中os.path.dirname()是获得当前文件的父目录
# 并将其加载到环境变量中 (注：这种环境变量只在运行时生效，程序运行结束即失效)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from DataUtil.LUNGDataUtil import create_dataloaders

# 定义 YOLO 模型
class SimpleYOLO(nn.Module):
    def __init__(self, num_classes):
        super(SimpleYOLO, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*7*7, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, num_classes + 4)  # num_classes + 4 for bounding box coordinates
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    model = model.to(device)
    criterion_class = nn.CrossEntropyLoss()
    criterion_bbox = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, (labels, boxes) in train_loader:
            images = images.to(device)

            # Flatten the list of labels and boxes
            if labels.numel() == 0:
                continue  # Skip batch if no valid labels

            if boxes.numel() == 0:
                continue  # Skip batch if no valid boxes

            labels = labels.to(device)
            boxes = boxes.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Reshape the outputs and labels
            batch_size = images.size(0)
            num_classes = 6
            class_scores = outputs[:, :num_classes]  # Shape: (batch_size, num_classes)
            bbox_coords = outputs[:, num_classes:]  # Shape: (batch_size, 4)

            # Ensure class_scores and labels are the same length
            if class_scores.size(0) != labels.size(0):
                continue  # Skip batch if dimensions mismatch

            # Compute the losses
            class_loss = criterion_class(class_scores, labels)
            bbox_loss = criterion_bbox(bbox_coords, boxes)

            loss = class_loss + bbox_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
        
        validate_model(model, val_loader, device)

    print("Training complete")


def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, (labels, _) in val_loader:
            images = images.to(device)

            if labels.numel() == 0:
                continue  # Skip batch if no valid labels

            labels = labels.to(device)
            outputs = model(images)

            # Reshape the outputs and labels
            num_classes = 6
            class_scores = outputs[:, :num_classes]  # Shape: (batch_size, num_classes)

            # Collect all labels and predictions
            all_labels.append(labels.cpu())
            _, predicted = torch.max(class_scores.data, 1)
            all_predictions.append(predicted.cpu())

    # Concatenate all labels and predictions
    if len(all_labels) > 0:
        all_labels = torch.cat(all_labels)
    else:
        all_labels = torch.empty(0, dtype=torch.long)
    
    if len(all_predictions) > 0:
        all_predictions = torch.cat(all_predictions)
    else:
        all_predictions = torch.empty(0, dtype=torch.long)

    # Ensure the dimensions match before computing accuracy
    if all_labels.size(0) == all_predictions.size(0):
        total = all_labels.size(0)
        correct = (all_predictions == all_labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")
    else:
        print(f"Size mismatch: Labels size = {all_labels.size(0)}, Predictions size = {all_predictions.size(0)}")
        print("Cannot compute accuracy due to size mismatch.")



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 6

images_dir = 'E:\pyDLW\LUNGC\dataset\images'
labels_dir = 'E:\pyDLW\LUNGC\dataset\labels'
batch_size = 32

train_loader = create_dataloaders(images_dir, labels_dir, 'train', batch_size)
val_loader = create_dataloaders(images_dir, labels_dir, 'val', batch_size)

model = SimpleYOLO(num_classes)
train_model(model, train_loader, val_loader, device, num_epochs=10)

# 保存训练好的模型
torch.save(model.state_dict(), 'simple_yolo_model.pth')