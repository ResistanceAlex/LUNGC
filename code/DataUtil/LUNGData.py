import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LungNoduleDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),
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
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                label = int(parts[0])
                labels.append(label)
        return labels

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace('.jpg', '.txt'))
        
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found: {img_path}")

        labels = self.read_labels(label_path)
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(labels[0], dtype=torch.int64)
        
        return image, label
