import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os

class LungNoduleDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # 将 numpy 数组转换为 PIL 图像
            transforms.RandomRotation(15),  # 随机旋转进行数据增强
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_files = sorted(os.listdir(images_dir))
    
    def __len__(self):
        return len(self.image_files)
    
    def read_labels(self, label_path):
        labels = []
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                label = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                labels.append(label)
                boxes.append([x_center, y_center, width, height])
        return labels, boxes

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace('.png', '.txt'))
        
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found: {img_path}")

        labels, boxes = self.read_labels(label_path)
        
        if self.transform:
            image = self.transform(image)
        
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        return image, (labels, boxes)
