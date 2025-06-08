import os
import warnings
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