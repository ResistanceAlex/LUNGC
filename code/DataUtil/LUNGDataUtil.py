import torch
from torch.utils.data import DataLoader

import sys
import os

# 将目录加载到系统环境中，其中os.path.dirname()是获得当前文件的父目录
# 并将其加载到环境变量中 (注：这种环境变量只在运行时生效，程序运行结束即失效)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from  DataUtil.LUNGData import LungNoduleDataset

'''
自定义批处理函数
    
param：
- batch: 一个批次的数据
    
return:
- images: 批次图像张量
- features: 批次附加特征张量
- labels: 批次标签张量
'''
def collate_fn(batch):
    images = []
    labels = []
    boxes = []

    for img, target in batch:
        images.append(img)
        
        if target[0].numel() > 0:  # 确保 labels 不为空
            labels.append(target[0].clone().detach())
        if target[1].numel() > 0:  # 确保 boxes 不为空
            boxes.append(target[1].clone().detach())

    # 如果 labels 或 boxes 是空的，则创建一个空的张量
    if len(labels) > 0:
        labels = torch.cat(labels, dim=0)  # 拼接所有标签张量
    else:
        labels = torch.empty(0, dtype=torch.long)

    if len(boxes) > 0:
        boxes = torch.cat(boxes, dim=0)  # 拼接所有边界框张量
    else:
        boxes = torch.empty(0, dtype=torch.float32)

    return torch.stack(images), (labels, boxes)


'''
加载数据集函数

param：
- images_dir: 图片路径
- labels_dir: 标签路径
- catetory: 类别名
- batch_size: 每批次数据的图片数量

return:
- dataloader: 数据集
'''
def create_dataloaders(images_dir, labels_dir, catetory, batch_size):

    data_db = LungNoduleDataset(os.path.join(images_dir, catetory), os.path.join(labels_dir, catetory))
    dataloader = DataLoader(data_db, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    images_dir = 'E:\pyDLW\LUNGC\dataset\images'
    labels_dir = 'E:\pyDLW\LUNGC\dataset\labels'
    
    batch_size = 32

    train_loader = create_dataloaders(images_dir, labels_dir, 'train', batch_size)
    val_loader = create_dataloaders(images_dir, labels_dir, 'val', batch_size)
    test_loader = create_dataloaders(images_dir, labels_dir, 'test', batch_size)
