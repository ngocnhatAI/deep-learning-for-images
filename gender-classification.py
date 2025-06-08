import os
import random
import warnings
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

warnings.filterwarnings('ignore')

from dataset import *
from transform import *

IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = r"./data/gender-classification"


def preprocessing(batch_size, num_workers):
    # Transform 
    train_transform = gender_transforms(is_train=True)
    val_transform = gender_transforms(is_train=False)

    # Datasets
    train_dataset = GenderDataset(
        root_dir=os.path.join(data_path, 'Train'),
        transform=train_transform
    )
    val_dataset = GenderDataset(
        root_dir=os.path.join(data_path, 'Val'),
        transform=val_transform
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader

def get_model(model_name, num_classes=1):
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)  # MobileNetV2 với trọng số pre-trained
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # Thay tầng cuối của classifier
    elif model_name == 'vit':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Mô hình {model_name} không được hỗ trợ!")

    return model


# TRAIN
def train(model, model_name, train_loader, val_loader, epochs, criterion, optimizer):
    batch_count = len(train_loader)
    train_losses = []
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        running_loss = running_loss/batch_count
        train_losses.append(running_loss)
        accuracy, val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}: Train loss = {running_loss:.4f}, Validation loss = {val_loss:.4f}")
        print(f"Accuracy on val set: {accuracy:5.2f}%")

        # Lưu mô hình nếu đạt độ chính xác cao nhất
        if accuracy > best_acc:
            best_acc = accuracy
            # Tạo thư mục 'weights/gender-classification' nếu chưa tồn tại
            save_dir = os.path.join('weights', 'gender-classification')
            os.makedirs(save_dir, exist_ok=True)
            # Tạo tên file dựa trên tên mô hình
            save_path = os.path.join(save_dir, f"{model_name}_best.pth")
            # Lưu trạng thái mô hình
            torch.save(model.state_dict(), save_path)
            print(f"Mô hình được lưu tại '{save_path}' với độ chính xác: {best_acc:.3f}%")

        print("-" * 64)

    return train_losses



def evaluate(model, testloader, criterion, device):
    batch_count = len(testloader)
    test_loss = 0.0
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            prob = torch.sigmoid(outputs)  # Xác suất [0, 1]
            predicted = (prob > 0.5).long()    
                        
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_loss = test_loss/batch_count
        
    return accuracy, test_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cài đặt siêu tham số cho phân loại giới tính")
    
    # Tham số mô hình và huấn luyện
    parser.add_argument("--model", type=str, default="resnet18", help="Selected model name: resnet18, resnet34, resnet50, vgg16, mobilenet, vit")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    
    
    args = parser.parse_args()
    
    train_loader, val_loader = preprocessing(batch_size=args.batch_size, num_workers=args.num_workers)
    
    model = get_model(model_name=args.model)
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(model, args.model, train_loader, val_loader, args.epochs, criterion, optimizer)