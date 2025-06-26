# main_segmentation.py
import os
from dataclasses import dataclass
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torchinfo import summary

from dataset import MedSegDataset
from models.seg_model import MedSegLitModule  

# === CONFIG CLASS ===
@dataclass
class TrainingConfig:
    arch: str = "Unet"
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    learning_rate: float = 2e-4
    epochs: int = 40
    batch_size: int = 8
    num_workers: int = 4
    image_height: int = 384
    image_width: int = 512
    val_size: float = 0.1
    t_max: int = 50 * 123  # total training steps = epochs * len(train_dataloader)
    seed: int = 99
    ckpt_dir: str = "checkpoints/"
    data_dir: str = r"C:\Users\Admin\Desktop\FINAL-IMAGE-PROCESSING\deep-learning-for-images\data\medical-image-segmentation"

# === DATASET UTILITY ===
from glob import glob
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(height=384, width=512):
    train_transform = A.Compose([
        A.Resize(height=height, width=width),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.8, 1.2), rotate=(-40, 40), translate_percent=(0, 0.05), shear=(-5, 5), p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(10, 20, 10, p=0.5),
        A.RandomToneCurve(p=0.3),
        A.GaussNoise(std_range=(0.1, 0.2), p=0.3),
        A.ElasticTransform(p=0.3),
        A.GridDistortion(p=0.3),
        A.AdvancedBlur(p=0.3),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(height=height, width=width),
        ToTensorV2(),
    ])
    return train_transform, val_transform

def get_dataset(data_dir, val_size, seed, train_tf, val_tf):
    img_dir = os.path.join(data_dir, "Train", "Image")
    mask_dir = os.path.join(data_dir, "Train", "Mask")
    img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    mask_paths = [os.path.join(mask_dir, os.path.basename(p).replace(".jpg", ".png")) for p in img_paths]
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(img_paths, mask_paths, test_size=val_size, random_state=seed)
    return (
        MedSegDataset(train_imgs, train_masks, transform=train_tf),
        MedSegDataset(val_imgs, val_masks, transform=val_tf)
    )

# === TRAINING FUNCTION ===
def train_model(cfg: TrainingConfig):
    L.seed_everything(cfg.seed)
    train_tf, val_tf = get_transform(cfg.image_height, cfg.image_width)
    train_set, val_set = get_dataset(cfg.data_dir, cfg.val_size, cfg.seed, train_tf, val_tf)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = MedSegLitModule(
        arch=cfg.arch,
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,
        learning_rate=cfg.learning_rate,
        t_max=cfg.t_max
    )

    print(summary(model, input_size=(cfg.batch_size, 3, cfg.image_height, cfg.image_width),
                  col_names=["input_size", "output_size", "num_params"]))

    filename_best = f"{cfg.arch}_{cfg.encoder_name}_best"
    filename_last = f"{cfg.arch}_{cfg.encoder_name}_last"
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.ckpt_dir, monitor="val/loss_best", 
        filename=filename_best, save_last=True
    )
    checkpoint_cb.CHECKPOINT_NAME_LAST = filename_last

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        callbacks=[checkpoint_cb],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# === MAIN ===
def main():
    config = TrainingConfig(
        arch="UnetPlusPlus",
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        epochs=40,
    )
    train_model(config)

if __name__ == "__main__":
    main()
