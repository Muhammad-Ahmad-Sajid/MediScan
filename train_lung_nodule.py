import os
import sys
import glob
import json
import time
import logging
import random
import numpy as np
import pandas as pd
import cv2
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.metrics import classification_report, confusion_matrix

import mlflow

# ==============================================================================
# CONFIGURATION & LOGGING
# ==============================================================================
BASE_DIR = "d:/X-ray ML Model/Mediscan/lung_nodule/Data"
CHECKPOINT_DIR = "d:/X-ray ML Model/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "lung_nodule_best.pth")
HISTORY_JSON_PATH = os.path.join(CHECKPOINT_DIR, "lung_nodule_training_history.json")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("mediscan.lung_nodule.train")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_WORKERS = 0
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
LR = 1e-4
WEIGHT_DECAY = 1e-5

# ==============================================================================
# PREPROCESSING & AUGMENTATION
# ==============================================================================
def get_transforms(is_small_dataset):
    if is_small_dataset:
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2), # CutOut
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def apply_noise(tensor, p):
        if torch.rand(1).item() < p:
            noise = torch.randn(tensor.size()) * 0.05
            return tensor + noise
        return tensor

    return train_transforms, val_transforms, apply_noise

def preprocess_image(img_path):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        img_gray = np.zeros((224, 224), dtype=np.uint8)
        
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    return img_rgb

class LungNoduleDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, is_train=False, noise_prob=0.0, noise_func=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_train = is_train
        self.noise_prob = noise_prob
        self.noise_func = noise_func

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img_rgb = preprocess_image(img_path)
        
        if self.transform:
            tensor = self.transform(img_rgb)
            
        if self.is_train and self.noise_func:
            tensor = self.noise_func(tensor, self.noise_prob)
            
        return tensor, torch.tensor(label, dtype=torch.float).unsqueeze(0)

# ==============================================================================
# MODEL & TRAINING UTILS
# ==============================================================================
def build_model():
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1) # Binary classification
    )
    
    # Freeze all except layer3, layer4, fc
    for name, param in model.named_parameters():
        if not any(n in name for n in ['layer3', 'layer4', 'fc']):
            param.requires_grad = False
            
    return model.to(DEVICE)

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    logger.info("Unfrozen all layers for fine-tuning.")

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, total = 0.0, 0
    all_preds, all_labels = [], []
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        total += labels.size(0)
        
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / total
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss, total = 0.0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / total
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return epoch_loss, epoch_acc, sensitivity, specificity, all_preds, all_labels

def load_split_data(split_name):
    """Loads images from a specific split folder (train, valid, test)."""
    split_dir = os.path.join(BASE_DIR, split_name)
    paths = []
    labels = []
    
    if not os.path.exists(split_dir):
        return paths, labels
        
    for class_folder in os.listdir(split_dir):
        class_path = os.path.join(split_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
            
        # Class 0: Normal, Class 1: Nodule (Adeno, Large Cell, Squamous)
        label = 0 if 'normal' in class_folder.lower() else 1
        
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png','.jpg','.jpeg','.tif')):
                paths.append(os.path.join(class_path, img_name))
                labels.append(label)
                
    return paths, labels

# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
def main():
    if not os.path.exists(BASE_DIR):
        logger.error(f"Dataset Data folder does not exist: {BASE_DIR}")
        sys.exit(1)
        
    # Load already split data
    X_train, y_train = load_split_data('train')
    X_val, y_val = load_split_data('valid')
    X_test, y_test = load_split_data('test')
    
    total_images = len(X_train) + len(X_val) + len(X_test)
    if total_images == 0:
        logger.error("No images found in the train/valid/test folders.")
        sys.exit(1)
        
    logger.info(f"Loaded {total_images} total images.")
    logger.info(f"Splits -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Class Distribution
    train_pos = sum(y_train)
    train_neg = len(y_train) - train_pos
    logger.info(f"Train Class Distribution -> Nodule: {train_pos}, Normal: {train_neg}")
    
    is_small_dataset = total_images < 1000
    if is_small_dataset:
        logger.info(f"Small dataset detected ({total_images} images). Enabled heavy augmentation.")
        global BATCH_SIZE, EPOCHS
        BATCH_SIZE = 4
        EPOCHS = 50
        
    train_tf, val_tf, noise_func = get_transforms(is_small_dataset)
    noise_prob = 0.4 if is_small_dataset else 0.2
    
    train_ds = LungNoduleDataset(X_train, y_train, transform=train_tf, is_train=True, noise_prob=noise_prob, noise_func=noise_func)
    val_ds = LungNoduleDataset(X_val, y_val, transform=val_tf)
    test_ds = LungNoduleDataset(X_test, y_test, transform=val_tf)
    
    # Class imbalance handling
    ratio = train_neg / train_pos if train_pos > 0 else 1
    
    if ratio > 2.0 or ratio < 0.5:
        logger.info(f"Class imbalance ratio is {ratio:.2f}. Using WeightedRandomSampler and pos_weight.")
        class_sample_counts = [train_neg, train_pos]
        weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
        samples_weights = weights[[0 if y == 0 else 1 for y in y_train]]
        sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
        pos_weight = torch.tensor([train_neg / train_pos]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        strategy = "WeightedRandomSampler + pos_weight"
    else:
        logger.info("Dataset is reasonably balanced. Using standard DataLoader and BCELoss.")
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        criterion = nn.BCEWithLogitsLoss()
        strategy = "Standard"
        
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Model & Optim
    model = build_model()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    mlflow.set_experiment("lung_nodule_detection")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    best_sensitivity = 0.0
    best_acc = 0.0
    history = []
    
    start_time = time.time()
    
    with mlflow.start_run(run_name="lung_nodule_v1"):
        mlflow.log_params({
            "model": "ResNet50", "dataset": "CT_Class_Dataset",
            "batch_size": BATCH_SIZE, "lr": LR, "class_balance_strategy": strategy
        })
        
        print("\nEpoch | Train Loss | Train Acc | Val Loss | Val Acc | Sensitivity | Specificity | LR")
        print("-" * 85)
        
        for epoch in range(1, EPOCHS + 1):
            if epoch == 6:
                unfreeze_all(model)
                optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
                
            current_lr = optimizer.param_groups[0]['lr']
            
            t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            v_loss, v_acc, sens, spec, _, _ = evaluate(model, val_loader, criterion)
            
            scheduler.step(v_loss)
            
            print(f"{epoch:5} | {t_loss:10.4f} | {t_acc:9.4f} | {v_loss:8.4f} | {v_acc:7.4f} | {sens:11.4f} | {spec:11.4f} | {current_lr:.2e}")
            print(f"Nodule Sensitivity: {sens*100:.2f}% — target above 80%")
            
            history.append({
                "epoch": epoch, "train_loss": t_loss, "val_loss": v_loss,
                "train_acc": t_acc, "val_acc": v_acc, "sensitivity": sens, "specificity": spec
            })
            
            if epoch >= 8 and v_acc < 0.65:
                logger.warning("Validation accuracy is below 65% after epoch 8.")
            if epoch >= 10 and sens < 0.65:
                logger.warning("Nodule sensitivity dropped below 65%.")
                
            mlflow.log_metrics({
                "train_loss": t_loss, "val_loss": v_loss,
                "train_acc": t_acc, "val_acc": v_acc,
                "nodule_sensitivity": sens, "nodule_specificity": spec
            }, step=epoch)
            
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_epoch = epoch
                best_sensitivity = sens
                best_acc = v_acc
                epochs_no_improve = 0
                torch.save({
                    'epoch': epoch, 'state_dict': model.state_dict(),
                    'val_loss': v_loss, 'val_acc': v_acc, 'sensitivity': sens
                }, BEST_MODEL_PATH)
            else:
                epochs_no_improve += 1
                
            with open(HISTORY_JSON_PATH, "w") as f:
                json.dump(history, f, indent=4)
                
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
    total_time = time.time() - start_time
    
    # TEST EVALUATION
    logger.info("=== FINAL EVALUATION ON TEST SET ===")
    checkpoint = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    
    _, test_acc, test_sens, test_spec, test_preds, test_labels = evaluate(model, test_loader, criterion)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=["No Nodule", "Nodule"]))
    print(f"\n*** Nodule Sensitivity (Recall): {test_sens*100:.2f}% ***")
    print(f"Nodule Specificity: {test_spec*100:.2f}%")
    
    if test_sens < 0.80:
        print("\n⚠️ Nodule sensitivity below clinical threshold.")
        print("Consider: (1) lowering classification threshold, (2) adding more augmentation, (3) combining with additional data.")
        
    print(f"\nTraining summary: best epoch={best_epoch}, best val acc={best_acc:.4f}, best sensitivity={best_sensitivity:.4f}, total training time={total_time/60:.2f} mins")

if __name__ == "__main__":
    main()
