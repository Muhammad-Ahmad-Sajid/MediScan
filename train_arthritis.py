import os
import time
import json
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, recall_score
import mlflow

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants and Configuration
BASE_DIR = "d:/X-ray ML Model/Mediscan"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "arthritis_best.pth")
LATEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "arthritis_latest.pth")
HISTORY_JSON_PATH = os.path.join(CHECKPOINT_DIR, "arthritis_training_history.json")

BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 7
LR = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRADE_NAMES = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
CLASS_WEIGHTS = [1.0, 2.19, 1.51, 3.02, 13.2]

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., p=0.3):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std}, p={self.p})'

# Transforms
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    AddGaussianNoise(mean=0., std=0.05, p=0.3),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ArthritisDataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        split_dir = os.path.join(base_dir, split)
        for grade in range(5):
            grade_dir = os.path.join(split_dir, str(grade))
            if os.path.exists(grade_dir):
                for filename in os.listdir(grade_dir):
                    if filename.lower().endswith('.png'):
                        self.image_paths.append(os.path.join(grade_dir, filename))
                        self.labels.append(grade)
                        
        logger.info(f"Loaded {len(self.image_paths)} images for {split} split.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load as grayscale
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            # Fallback if image fails to load
            img_gray = np.zeros((224, 224), dtype=np.uint8)
            
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)
        
        # Convert to 3-channel RGB
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        
        if self.transform:
            img_tensor = self.transform(img_rgb)
            
        return img_tensor, torch.tensor(label, dtype=torch.long)

def build_model():
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Replace final FC layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 5)
    )
    
    # Freeze all layers except layer3, layer4, and fc
    for name, param in model.named_parameters():
        if not any(n in name for n in ['layer3', 'layer4', 'fc']):
            param.requires_grad = False
            
    return model.to(DEVICE)

def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True
    logger.info("Unfrozen all model layers for full fine-tuning.")

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def main():
    logger.info("Initializing Arthritis Grading Training Pipeline...")
    logger.info(f"Using device: {DEVICE}")
    
    # DataLoaders
    train_dataset = ArthritisDataset(BASE_DIR, 'train', transform=train_transforms)
    val_dataset = ArthritisDataset(BASE_DIR, 'val', transform=val_transforms)
    test_dataset = ArthritisDataset(BASE_DIR, 'test', transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Model Setup
    model = build_model()
    
    # Loss, Optimizer, Scheduler
    weights_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    
    # MLflow Setup
    mlflow.set_experiment("arthritis_grading")
    
    history = []
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_epoch = 0
    start_epoch = 1
    
    # Auto-resume logic
    if os.path.exists(LATEST_MODEL_PATH):
        logger.info(f"Found existing checkpoint at {LATEST_MODEL_PATH}. Resuming training...")
        checkpoint = torch.load(LATEST_MODEL_PATH)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        best_val_acc = checkpoint.get('best_val_acc', best_val_acc)
        
        # If we passed the freeze period, unfreeze before loading optimizer state
        if start_epoch > 6:
            unfreeze_all_layers(model)
            # Re-initialize the optimizer to hold all parameters before loading its massive state
            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
            
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if os.path.exists(HISTORY_JSON_PATH):
            try:
                with open(HISTORY_JSON_PATH, "r") as f:
                    history = json.load(f)
            except:
                pass
                
        logger.info(f"Resumed successfully. Starting from Epoch {start_epoch}")
    
    start_time = time.time()
    
    with mlflow.start_run(run_name="arthritis_v1"):
        mlflow.log_params({
            "model": "ResNet50",
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "class_weights": CLASS_WEIGHTS,
            "dataset": "KneeOA",
            "epochs": NUM_EPOCHS,
            "optimizer": "Adam"
        })
        
        logger.info(f"{'Epoch':<5} | {'Train Loss':<10} | {'Train Acc':<9} | {'Val Loss':<8} | {'Val Acc':<7} | {'Best':<4} | {'LR'}")
        logger.info("-" * 75)
        
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            # Unfreeze after epoch 5
            if epoch == 6:
                unfreeze_all_layers(model)
                # Re-initialize optimizer with all parameters
                optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, val_preds, val_labels = evaluate_model(model, val_loader, criterion)
            
            scheduler.step(val_loss)
            
            # Save History
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc
            })
            
            # MLflow Logging
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": current_lr
            }, step=epoch)
            
            is_best = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_no_improve = 0
                is_best = True
                
                # Best Checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, BEST_MODEL_PATH)
            else:
                epochs_no_improve += 1
                
            # Latest Checkpoint (for pausing/resuming)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc
            }, LATEST_MODEL_PATH)
                
            best_marker = "*" if is_best else ""
            logger.info(f"{epoch:<5} | {train_loss:<10.4f} | {train_acc:<9.4f} | {val_loss:<8.4f} | {val_acc:<7.4f} | {best_marker:<4} | {current_lr:.2e}")
            
            # Save simple text log
            log_line = f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n"
            with open(os.path.join(CHECKPOINT_DIR, "training_log.txt"), "a") as f:
                f.write(log_line)
            
            # Save training history JSON incrementally
            with open(HISTORY_JSON_PATH, "w") as f:
                json.dump(history, f, indent=4)
            
            # Warnings
            if epoch == 10 and val_acc <= 0.40:
                logger.warning(f"Validation accuracy is only {val_acc:.2%}. Model may be failing to learn.")
                
            # Class-wise stats every 5 epochs
            if epoch % 5 == 0:
                cm = confusion_matrix(val_labels, val_preds, labels=[0, 1, 2, 3, 4])
                cm_diag = cm.diagonal()
                cm_sums = cm.sum(axis=1)
                
                logger.info("\nPer-class Validation Accuracy:")
                for i in range(5):
                    class_acc = cm_diag[i] / cm_sums[i] if cm_sums[i] > 0 else 0
                    logger.info(f"Grade {i} ({GRADE_NAMES[i]:<8}) : {class_acc:.2%}")
                
                recall_severe = cm_diag[4] / cm_sums[4] if cm_sums[4] > 0 else 0
                if recall_severe < 0.30:
                    logger.warning(f"Grade 4 (Severe) recall is below 30% ({recall_severe:.2%}). Model is ignoring the rarest class.")
                logger.info("-" * 75)
                
            # Early Stopping
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping condition met (7 epochs without improvement), but forcefully ignored to reach Epoch 30.")
                # break
                
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
        
        # --- FINAL EVALUATION ---
        logger.info("Loading best model for final evaluation on test set...")
        checkpoint = torch.load(BEST_MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_acc, test_preds, test_labels = evaluate_model(model, test_loader, criterion)
        
        overall_acc = accuracy_score(test_labels, test_preds)
        macro_f1 = f1_score(test_labels, test_preds, average='macro')
        weighted_f1 = f1_score(test_labels, test_preds, average='weighted')
        
        logger.info("\n" + "="*50)
        logger.info("FINAL EVALUATION RESULTS")
        logger.info("="*50)
        logger.info(f"Best Epoch: {best_epoch}")
        logger.info(f"Best Val Acc: {best_val_acc:.4f}")
        logger.info(f"Best Val Loss: {best_val_loss:.4f}")
        logger.info(f"Total Training Time: {total_time // 60:.0f}m {total_time % 60:.0f}s")
        logger.info("-" * 50)
        logger.info(f"Test Accuracy: {overall_acc:.4f}")
        logger.info(f"Test Macro F1: {macro_f1:.4f}")
        logger.info(f"Test Weighted F1: {weighted_f1:.4f}")
        logger.info("-" * 50)
        logger.info("Classification Report:")
        logger.info("\n" + classification_report(test_labels, test_preds, target_names=GRADE_NAMES, zero_division=0))
        logger.info("Confusion Matrix:")
        logger.info("\n" + str(confusion_matrix(test_labels, test_preds, labels=[0, 1, 2, 3, 4])))
        logger.info("="*50)

if __name__ == "__main__":
    main()
