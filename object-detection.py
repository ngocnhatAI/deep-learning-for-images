import os
import torch
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from ensemble_boxes import weighted_boxes_fusion

@dataclass
class Config:
    data_dir: str = "data"
    train_image_dir: str = "data/Train/images"
    train_label_dir: str = "data/Train/labels"
    test_image_dir: str = "data/Test/images"
    kfold_dir: str = "data/kfold"
    results_dir: str = "results"
    cache_dir: str = "data/cache"
    batch_size: int = 8
    epochs: int = 75
    patience: int = 15
    seed: int = 42
    workers: int = 4
    model_type: str = "yolov8x.pt"
    img_size: int = 640
    augment: bool = True
    k_folds: int = 7
    lr0: float = 0.001
    lrf: float = 0.01
    optimizer: str = "AdamW"
    weight_decay: float = 0.0005
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ensemble: bool = True
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    wbf_iou_thr: float = 0.5
    wbf_skip_box_thr: float = 0.001
    weights: list = None
    classes: list = None
    num_classes: int = None
    yaml_path: str = None

CONFIG = Config()

def apply_weighted_fusion(predictions_list, img_width, img_height, iou_thr=0.5, skip_box_thr=0.001, weights=None):
    """Apply weighted boxes fusion for ensemble predictions"""
    if not predictions_list:
        return []
    
    all_boxes = []
    all_scores = []
    all_labels = []
    
    for predictions in predictions_list:
        if not predictions:
            all_boxes.append(np.array([]).reshape(0, 4))
            all_scores.append(np.array([]))
            all_labels.append(np.array([]))
            continue
        
        # Convert to arrays for batch operations
        if isinstance(predictions[0]['xyxy'], np.ndarray):
            boxes = np.array([pred['xyxy'] for pred in predictions])
        else:
            boxes = np.array([[pred['xyxy'][0], pred['xyxy'][1], pred['xyxy'][2], pred['xyxy'][3]] 
                             for pred in predictions])
        
        scores = np.array([pred['conf'] for pred in predictions])
        labels = np.array([pred['cls'] for pred in predictions])
        
        # Normalize coordinates
        boxes[:, [0, 2]] /= img_width
        boxes[:, [1, 3]] /= img_height
        boxes = np.clip(boxes, 0.0, 1.0)
        
        # Filter valid boxes
        valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        
        all_boxes.append(boxes[valid_mask].tolist())
        all_scores.append(scores[valid_mask].tolist())
        all_labels.append(labels[valid_mask].tolist())
    
    # Apply WBF
    try:
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes,
            all_scores,
            all_labels,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr
        )
        
        # Convert back to original coordinates
        fused_boxes = np.array(fused_boxes)
        fused_scores = np.array(fused_scores)
        fused_labels = np.array(fused_labels)
        
        if len(fused_boxes) > 0:
            # Convert back to pixel coordinates
            fused_boxes[:, [0, 2]] *= img_width
            fused_boxes[:, [1, 3]] *= img_height
            
            # Create final results
            final_boxes = []
            for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
                final_boxes.append({
                    "x_min": round(float(box[0]), 2),
                    "y_min": round(float(box[1]), 2),
                    "x_max": round(float(box[2]), 2),
                    "y_max": round(float(box[3]), 2),
                    "class": int(label),
                    "confidence": round(float(score), 4)
                })
            
            return final_boxes
        else:
            return []
            
    except Exception as e:
        print(f"WBF failed: {e}")
        return []

def normalize_labels(label_files, image_dir, output_label_dir):
    print("Converting label files to YOLO format...")
    
    # Pre-compute image dimensions
    image_dims_cache = {}
    cache_file = os.path.join(cache_dir, 'image_dimensions.npy')
    
    if CONFIG['use_cache'] and os.path.exists(cache_file):
        print("Loading cached image dimensions...")
        image_dims_cache = np.load(cache_file, allow_pickle=True).item()
    else:
        print("Computing image dimensions...")
        for label_file in tqdm(label_files, desc="Computing image dimensions"):
            image_filename = os.path.basename(label_file).replace('.txt', '.jpg')
            image_path = os.path.join(image_dir, image_filename)
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    image_dims_cache[image_filename] = np.array(img.size, dtype=np.float32)
        
        if CONFIG['use_cache']:
            np.save(cache_file, image_dims_cache)
    
    # Process labels in batches
    os.makedirs(output_label_dir, exist_ok=True)
    
    for label_file in tqdm(label_files, desc="Converting labels"):
        image_filename = os.path.basename(label_file).replace('.txt', '.jpg')
        
        if image_filename not in image_dims_cache:
            continue
            
        img_width, img_height = image_dims_cache[image_filename]
        
        # Read all lines at once
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            continue
            
        # Convert to arrays for batch processing
        data = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                data.append(list(map(float, parts)))
        
        if not data:
            continue
            
        data = np.array(data, dtype=np.float32)
        
        # Batch coordinate processing
        classes = data[:, 0].astype(int)
        coords = data[:, 1:5]  # x_min, y_min, x_max, y_max
        
        # Clip coordinates to image bounds
        coords[:, [0, 2]] = np.clip(coords[:, [0, 2]], 0, img_width)
        coords[:, [1, 3]] = np.clip(coords[:, [1, 3]], 0, img_height)
        
        # Convert to YOLO format
        x_center = (coords[:, 0] + coords[:, 2]) / (2 * img_width)
        y_center = (coords[:, 1] + coords[:, 3]) / (2 * img_height)
        width = (coords[:, 2] - coords[:, 0]) / img_width
        height = (coords[:, 3] - coords[:, 1]) / img_height
        
        # Filter valid boxes
        valid_mask = (width > 0) & (height > 0)
        
        if np.any(valid_mask):
            valid_classes = classes[valid_mask]
            valid_coords = np.column_stack([
                x_center[valid_mask],
                y_center[valid_mask],
                width[valid_mask],
                height[valid_mask]
            ])
            
            # Write to file
            output_label_path = os.path.join(output_label_dir, os.path.basename(label_file))
            with open(output_label_path, 'w') as f:
                for cls, coord in zip(valid_classes, valid_coords):
                    f.write(f"{cls} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f} {coord[3]:.6f}\n")
                    
# Convert all label files
label_files = glob.glob(os.path.join(train_label_dir, '*.txt'))
normalize_labels(label_files, train_image_dir, train_label_dir)

# Define classes
classes = ['Apple', 'Banana', 'Grape', 'Orange', 'Pineapple', 'Watermelon']
num_classes = len(classes)


                    
def main():
    os.makedirs(CONFIG.kfold_dir, exist_ok=True)
    os.makedirs(CONFIG.results_dir, exist_ok=True)
    os.makedirs(CONFIG.cache_dir, exist_ok=True)

    CONFIG.classes = ['Apple', 'Banana', 'Grape', 'Orange', 'Pineapple', 'Watermelon']
    CONFIG.num_classes = len(CONFIG.classes)
    CONFIG.yaml_path = os.path.join(CONFIG.data_dir, 'data.yaml')

    # Create data.yaml
    yaml.dump({
        'names': CONFIG.classes,
        'nc': CONFIG.num_classes,
        'test': CONFIG.test_image_dir,
        'train': CONFIG.train_image_dir,
        'val': CONFIG.train_image_dir,
    }, open(CONFIG.yaml_path, 'w'))

    # Normalize labels
    label_files = glob.glob(os.path.join(CONFIG.train_label_dir, '*.txt'))
    normalize_labels(label_files, CONFIG.train_image_dir, CONFIG.train_label_dir)

    # Analyze class distribution
    labels = sorted(Path(CONFIG.data_dir).rglob("Train/labels/*.txt"))

    # KFold split
    kf = StratifiedKFold(n_splits=CONFIG.k_folds, shuffle=True, random_state=CONFIG.seed)
    kfolds = list(kf.split(labels_df, labels_df['strat_target']))
    image_paths = np.array(sorted(list(Path(CONFIG.train_image_dir).glob("*.jpg"))))

    yaml_paths = []
    for i, (train_idx, val_idx) in enumerate(kfolds):
        train_txt = os.path.join(CONFIG.kfold_dir, f"train_{i}.txt")
        val_txt = os.path.join(CONFIG.kfold_dir, f"val_{i}.txt")
        with open(train_txt, 'w') as f:
            f.writelines(str(image_paths[j]) + '\n' for j in train_idx)
        with open(val_txt, 'w') as f:
            f.writelines(str(image_paths[j]) + '\n' for j in val_idx)
        fold_yaml = os.path.join(CONFIG.kfold_dir, f"data_{i}.yaml")
        yaml.dump({
            'train': os.path.relpath(train_txt, CONFIG.kfold_dir),
            'val': os.path.relpath(val_txt, CONFIG.kfold_dir),
            'names': CONFIG.classes,
            'nc': CONFIG.num_classes
        }, open(fold_yaml, 'w'))
        yaml_paths.append(fold_yaml)

    # Train
    trained_models = []
    for i in range(CONFIG.k_folds):
        fold_dir = os.path.join(CONFIG.results_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)
        model = YOLO(CONFIG.model_type)
        print(f"Training fold {i}...")
        model.train(
            data=yaml_paths[i],
            batch=CONFIG.batch_size,
            epochs=CONFIG.epochs,
            patience=CONFIG.patience,
            project=CONFIG.results_dir,
            name=f"fold_{i}",
            lr0=CONFIG.lr0,
            lrf=CONFIG.lrf,
            optimizer=CONFIG.optimizer,
            weight_decay=CONFIG.weight_decay,
            imgsz=CONFIG.img_size,
            augment=CONFIG.augment,
            workers=CONFIG.workers,
            device=CONFIG.device,
            exist_ok=True,
            verbose=False,
            save_dir=fold_dir
        )
        best_path = os.path.join(fold_dir, 'weights/best.pt')
        trained_models.append(best_path)

    print("Training complete!")
    # Predict on test using WBF
    test_image_paths = sorted(glob.glob(os.path.join(CONFIG.test_image_dir, '*.jpg')))
    results_data = []
    for image_path in tqdm(test_image_paths, desc="Test prediction"):
        image_id = Path(image_path).stem
        img = Image.open(image_path)
        img_width, img_height = img.size
        predictions_list = []
        for model_path in trained_models:
            model = YOLO(model_path)
            results = model.predict(image_path, conf=CONFIG.conf_threshold, iou=CONFIG.iou_threshold, device=CONFIG.device, verbose=False)
            preds = []
            for r in results:
                if r.boxes is not None:
                    boxes_data = r.boxes.data.cpu().numpy()
                    for box in boxes_data:
                        preds.append({'xyxy': box[:4], 'conf': float(box[4]), 'cls': int(box[5])})
            predictions_list.append(preds)
        boxes = apply_weighted_fusion(predictions_list, img_width, img_height, iou_thr=CONFIG.wbf_iou_thr, skip_box_thr=CONFIG.wbf_skip_box_thr)
        results_data.append({"image_id": image_id, "bounding_boxes": json.dumps(boxes)})
    print("Prediction complete!")

if __name__ == '__main__':
    main()
