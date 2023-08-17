import torch
import albumentations
from albumentations.pytorch import ToTensorV2
import torchvision
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset

class FedDataset(Dataset):
    
    def __init__(self, file_paths, labels, transform=None):
        
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        
        target = torch.tensor(self.labels[idx], dtype=torch.long)
        file = self.file_paths[idx]
        image = cv2.imread(file)
        
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                aug = self.transform(image=image)
                image = aug['image']
                
        except:
            image = torch.zeros(3, 100, 100)
            
        return image, target

class PredictionDataset(Dataset):
    
    def __init__(self, file_paths, transform=None):
        
        self.file_paths = file_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        
        file_path = self.file_paths[idx]
        
        image = cv2.imread(file_path)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                aug = self.transform(image=image)
                image = aug['image']
        except:
            image = torch.zeros(3, 100, 100)
        
        return image

transform = albumentations.Compose([
    albumentations.Resize(150, 150),
    albumentations.CenterCrop(100, 100),
    albumentations.OneOf([
        albumentations.HorizontalFlip(p=1),
        albumentations.RandomRotate90(p=1),
        albumentations.VerticalFlip(p=1)
    ], p=1),
    albumentations.OneOf([
        albumentations.MotionBlur(p=1),
        albumentations.OpticalDistortion(p=1),
        albumentations.GaussNoise(p=1)
    ], p=1),
    albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

test_transform = albumentations.Compose([
    albumentations.Resize(150, 150),
    albumentations.CenterCrop(100, 100),
    albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

pred_transform = albumentations.Compose([
    albumentations.Resize(150, 150),
    albumentations.CenterCrop(100, 100),
    albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])
