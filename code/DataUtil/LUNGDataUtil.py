import os
import sys
import torch
from torch.utils.data import DataLoader

# 将目录加载到系统环境中，其中os.path.dirname()是获得当前文件的父目录
# 并将其加载到环境变量中 (注：这种环境变量只在运行时生效，程序运行结束即失效)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from DataUtil.LUNGData import LungNoduleDataset

def collate_fn(batch):
    images = []
    labels = []

    for img, label in batch:
        images.append(img)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels

def create_dataloaders(images_dir, labels_dir, category, batch_size):
    data_db = LungNoduleDataset(os.path.join(images_dir, category), os.path.join(labels_dir, category))
    dataloader = DataLoader(data_db, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader

if __name__ == '__main__':
    images_dir = 'E:\pyDLW\LUNGC\dataset\images'
    labels_dir = 'E:\pyDLW\LUNGC\dataset\labels'
    batch_size = 32

    train_loader = create_dataloaders(images_dir, labels_dir, 'train', batch_size)
    val_loader = create_dataloaders(images_dir, labels_dir, 'val', batch_size)
    test_loader = create_dataloaders(images_dir, labels_dir, 'test', batch_size)
