"""
Trains the Residual Landmark ISL Classifier with 218-dim Invariant Geometric Features
and 3D Spatial Rotation & Jitter Augmentation.
"""
import sys
from pathlib import Path
import time
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from collections import Counter

from src.landmark_model import create_landmark_model
from src.dataset import classes, idx_to_class
from src.utils import get_device

DATA_PATH = PROJECT_ROOT / "data/processed/landmarks.npz"
BEST_MODEL_PATH = PROJECT_ROOT / "models/best_landmark_model.pth"

BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1.2e-3


def apply_3d_augmentation(feats, jitter_std=0.012, max_angle_deg=12.0):
    """
    Applies realistic 3D rotation and coordinate jitter to hand landmarks during training.
    """
    batch_size = feats.shape[0]
    device = feats.device
    
    # 1. Coordinate Jitter
    noise = torch.randn_like(feats) * jitter_std
    # Preserve indicators and presence flags
    noise[:, 88] = 0.0
    noise[:, 177] = 0.0
    noise[:, 217] = 0.0
    feats_aug = feats + noise
    
    # 2. Random 3D Rotation for Hand 1 coords (indices 0:63 -> 21 x 3)
    angles_z = (torch.rand(batch_size, device=device) * 2 - 1) * math.radians(max_angle_deg)
    cos_z = torch.cos(angles_z)
    sin_z = torch.sin(angles_z)
    
    h1_coords = feats_aug[:, 0:63].view(batch_size, 21, 3)
    x = h1_coords[:, :, 0]
    y = h1_coords[:, :, 1]
    z = h1_coords[:, :, 2]
    
    x_new = x * cos_z.unsqueeze(1) - y * sin_z.unsqueeze(1)
    y_new = x * sin_z.unsqueeze(1) + y * cos_z.unsqueeze(1)
    
    feats_aug[:, 0:63] = torch.stack([x_new, y_new, z], dim=2).view(batch_size, 63)
    
    return feats_aug


def train():
    print("=" * 60)
    print("Training 218-Dim Residual Landmark ISL Classifier")
    print("=" * 60)
    
    if not DATA_PATH.exists():
        print(f"Error: Processed data not found at {DATA_PATH}. Run src/extract_landmarks.py first.")
        return

    data = np.load(DATA_PATH)
    features = torch.tensor(data["features"], dtype=torch.float32)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    hand_counts = torch.tensor(data["hand_counts"], dtype=torch.long)
    
    print(f"Loaded {len(features):,} landmark vectors across {len(np.unique(labels))} classes.")
    print(f"Feature Vector Dimensions: {features.shape[1]}")
    
    dataset = TensorDataset(features, labels, hand_counts)
    
    total_len = len(dataset)
    train_len = int(0.80 * total_len)
    val_len = int(0.10 * total_len)
    test_len = total_len - train_len - val_len
    
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_set):,} | Val samples: {len(val_set):,} | Test samples: {len(test_set):,}")
    
    device = get_device()
    model = create_landmark_model(num_classes=len(classes)).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y, batch_hc in train_loader:
            batch_x = apply_3d_augmentation(batch_x)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_hc = batch_hc.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x, hand_count=batch_hc, apply_mask=False)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss = train_loss / len(train_set)
        scheduler.step()
        
        # Validation
        model.eval()
        correct = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_hc in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_hc = batch_hc.to(device)
                
                outputs = model(batch_x, hand_count=batch_hc, apply_mask=True)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                
        val_loss = val_loss / len(val_set)
        val_acc = 100.0 * correct / len(val_set)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            save_msg = f"✓ Saved (Val Acc: {val_acc:.2f}%)"
        else:
            save_msg = ""
            
        if (epoch + 1) % 5 == 0 or epoch == 0 or save_msg:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f} {save_msg}", flush=True)
            
    print("\nEvaluating Best Model on Test Set...", flush=True)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    
    test_correct = 0
    class_correct = Counter()
    class_total = Counter()
    
    with torch.no_grad():
        for batch_x, batch_y, batch_hc in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_hc = batch_hc.to(device)
            
            outputs = model(batch_x, hand_count=batch_hc, apply_mask=True)
            preds = torch.argmax(outputs, dim=1)
            
            test_correct += (preds == batch_y).sum().item()
            for true_l, pred_l in zip(batch_y.tolist(), preds.tolist()):
                class_total[true_l] += 1
                if true_l == pred_l:
                    class_correct[true_l] += 1
                    
    total_test_acc = 100.0 * test_correct / len(test_set)
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60, flush=True)
    print(f"✓ Training Complete in {elapsed:.1f}s!", flush=True)
    print(f"✓ Best Model Saved to: {BEST_MODEL_PATH}", flush=True)
    print(f"✓ Final Test Accuracy: {total_test_acc:.2f}%", flush=True)
    print("=" * 60, flush=True)
    
    print("\nClass Accuracies on Test Set:", flush=True)
    for idx in range(len(classes)):
        c_name = idx_to_class[idx]
        acc = 100.0 * class_correct[idx] / max(1, class_total[idx])
        print(f"  Class {c_name:2s}: {acc:5.1f}% ({class_correct[idx]}/{class_total[idx]})", flush=True)


if __name__ == "__main__":
    train()
