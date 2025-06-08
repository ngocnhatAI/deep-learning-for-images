from torchvision import transforms

IMG_SIZE = 224

def gender_transforms(IMG_SIZE = 224, is_train=False):
    if is_train:
        return transforms.Compose([
            # 1. Apply color adjustments (brightness, contrast, saturation, hue)
            transforms.ColorJitter(
                brightness=(0.8, 1.2),  # Adjust brightness by ±20%
                contrast=(0.8, 1.2),    # Adjust contrast by ±20%
                saturation=(0.8, 1.2),  # Adjust saturation by ±20%
                hue=(-0.1, 0.1)        # Adjust hue by ±0.1
            ),
            # 2. Crop to square to maintain aspect ratio before resizing
            transforms.CenterCrop((178, 178)),  # Crop to square (178x178) from input (218x178)
            # 3. Resize image to target output size
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
            # 4. Apply geometric transformations
            transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip image horizontally with 50% probability
            transforms.RandomAffine(
                degrees=(-20, 20),           # Rotate image by ±20 degrees
                translate=(0.2, 0.2),        # Translate by up to 20% of image size
                scale=(0.8, 1.2),            # Scale image by 80%-120%
                shear=(-15, 15),             # Shear by ±15 degrees
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0                       # Fill empty areas with black
            ),
            transforms.RandomPerspective(
                distortion_scale=0.5,        # Apply perspective distortion with 50% scale
                p=0.5,                       # Apply with 50% probability
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0                       # Fill empty areas with black
            ),
            # 5. Adjust image sharpness
            transforms.RandomAdjustSharpness(
                sharpness_factor=1.5,        # Slightly increase sharpness
                p=0.3                        # Apply with 30% probability
            ),
            # 6. Convert to tensor and normalize
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean
                std=[0.229, 0.224, 0.225]    # Normalize with ImageNet std
            )
        ])
    else:
        return transforms.Compose([
            # 1. Crop to square to maintain aspect ratio
            transforms.CenterCrop((178, 178)),  # Crop to square (178x178) from input (218x178)
            # 2. Resize image to target output size
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
            # 3. Convert to tensor and normalize
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet mean
                std=[0.229, 0.224, 0.225]    # Normalize with ImageNet std
            )
        ])