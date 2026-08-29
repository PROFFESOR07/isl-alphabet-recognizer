"""
Trains the ISL alphabet recognition model.

Responsibilities:
- Load the datasets and model
- Train the model
- Validate after each epoch
- Save the best-performing model
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import train_loader, val_loader
from src.model import create_model
from src.utils import get_device

PROJECT_ROOT = Path(__file__).parent.parent

BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoint.pth"

LEARNING_RATE = 3e-4
NUM_EPOCHS = 10

device = get_device()

model = create_model().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1,
)

best_accuracy = 0.0

total_train_batches = len(train_loader)
total_val_batches = len(val_loader)

for epoch in range(NUM_EPOCHS):

    # ==========================
    # Training
    # ==========================

    model.train()

    train_running_loss = 0.0
    train_batches = 0

    print(f"\n--- Epoch {epoch+1:02d}/{NUM_EPOCHS} ---")

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_running_loss += loss.item()
        train_batches += 1

        if train_batches % 250 == 0 or train_batches == total_train_batches:
            avg_batch_loss = train_running_loss / train_batches
            print(
                f"  [Train] Batch {train_batches:4d}/{total_train_batches:4d} | "
                f"Running Loss: {avg_batch_loss:.4f}"
            )

    train_loss = train_running_loss / train_batches

    # ==========================
    # Validation
    # ==========================

    model.eval()

    val_running_loss = 0.0
    val_batches = 0

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            val_running_loss += loss.item()
            val_batches += 1

    val_loss = val_running_loss / val_batches
    accuracy = 100 * correct / total

    # Update learning rate scheduler
    scheduler.step(accuracy)

    # ==========================
    # Save Best Model
    # ==========================

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        # Used by predict.py
        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH,
        )

        # Full checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_accuracy": best_accuracy,
            },
            CHECKPOINT_PATH,
        )

        print(f"✓ Saved new best model ({best_accuracy:.2f}%)")

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
        f"LR: {current_lr:.6f} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {accuracy:.2f}%"
    )

print("\nTraining Complete!")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
print(f"Best Model Saved To: {BEST_MODEL_PATH}")