import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import models, transforms

# External packages
from dataset import *
from models import *


warnings.filterwarnings('ignore')

# ==== CONFIGURATION ====
config = {
    "batch_size": 32,
    "num_workers": 4,
    "img_size": 224,
    "epochs": 50,
    "patience": 10,
    "lr": 0.001,
    "data_path": r"C:\Users\Admin\Desktop\FINAL-IMAGE-PROCESSING\deep-learning-for-images\data\gender-classification",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_save_path": "weights/model_best.pth"
}

# ==== TRANSFORMS ====
def gender_transforms(IMG_SIZE=224, is_train=False):
    if is_train:
        return transforms.Compose([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(-15, 15), fill=0),
            transforms.RandomPerspective(0.15, p=0.5, fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def get_dataloaders(cfg):
    train_transform = gender_transforms(cfg["img_size"], is_train=True)
    test_transform = gender_transforms(cfg["img_size"], is_train=False)

    train_dataset = GenderDataset(os.path.join(cfg["data_path"], 'Train'), transform=train_transform)
    val_dataset = GenderDataset(os.path.join(cfg["data_path"], 'Val'), transform=test_transform)
    test_dataset = GenderDataset(cfg["data_path"], transform=test_transform, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    return train_loader, val_loader, test_loader

# ==== MODEL SETUP ====
def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)

    for name, param in model.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    def set_bn_eval(m):
        if isinstance(m, nn.BatchNorm2d):
            if all(not p.requires_grad for p in m.parameters()):
                m.eval()

    model.apply(set_bn_eval)
    return model

# ==== TRAINING ====
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x).squeeze()
            loss = criterion(out, y.float())
            total_loss += loss.item()
            prob = torch.sigmoid(out)
            pred = (prob > 0.5)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100 * correct / total
    return acc, total_loss / len(loader)

def train(model, train_loader, val_loader, cfg):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"] * 0.1)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0
    no_improve = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            optimizer.zero_grad()
            out = model(x).squeeze()
            loss = criterion(out, y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        val_acc, val_loss = evaluate(model, val_loader, criterion, cfg["device"])

        print(f"Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            os.makedirs(os.path.dirname(cfg["model_save_path"]), exist_ok=True)
            torch.save(model.state_dict(), cfg["model_save_path"])
            print(f"✅ Saved new best model: {best_acc:.2f}%")
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epochs.")

        if no_improve >= cfg["patience"]:
            print("⛔ Early stopping.")
            break

# ==== MAIN ====
def main():
    print("🚀 Starting training with config:", config)
    train_loader, val_loader, test_loader = get_dataloaders(config)
    model = build_model().to(config["device"])
    train(model, train_loader, val_loader, config)

if __name__ == "__main__":
    main()
