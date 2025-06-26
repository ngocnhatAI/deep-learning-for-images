import os
import warnings
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

class GenderDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.images = []  # Store images path
        self.labels = []  # Store labels of image (used for train and valid set)
        self.ids    = []  # Store ID of image (used for test set)
        
        if is_test:
            # Load test images without labels
            test_dir = os.path.join(root_dir, 'Test')
            for filename in sorted(os.listdir(test_dir)):
                if filename.endswith('.jpg'):
                    self.images.append(os.path.join(test_dir, filename))
                    # Extract ID from filename (remove .jpg extension)
                    image_id = filename.replace('.jpg', '')
                    self.ids.append(image_id)  # Store ID instead of label for test
        else:
            # Load train/val images with labels
            for gender in ['Female', 'Male']:
                gender_dir = os.path.join(root_dir, gender)
                if os.path.exists(gender_dir):
                    label = 1 if gender == 'Female' else 0  # Female=1, Male=0
                    for filename in os.listdir(gender_dir):
                        if filename.endswith('.jpg'):
                            self.images.append(os.path.join(gender_dir, filename))
                            self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, self.ids[idx]  # Return image and ID for test
        else:
            return image, self.labels[idx]  # Return image and label for train/val
        
class MedSegDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32) # binary mask
        
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']
            mask = np.expand_dims(mask, axis=0).astype(np.float32)

        return {
            "image": image,
            "mask": mask,
            "path": img_path
        }