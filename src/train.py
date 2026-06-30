"""
Trains the ISL alphabet recognition model.

Responsibilities:
- Load the datasets and model
- Train the model
- Validate after each epoch
- Save the best-performing model
"""

from src.dataset import train_loader, val_loader
from src.model import create_model
from src.utils import get_device
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

device = get_device()

model = create_model()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)#optimizes only those which has requires_grad = True
best_accuracy = 0.0
for epoch in range(NUM_EPOCHS):
    model.train()
    train_batches = 0
    train_running_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        #Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()#erase old weights

        loss.backward()#calculate new grads

        optimizer.step()#update model
        train_running_loss += loss.item()
        train_batches+=1
    
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
            val_batches+=1
    val_loss = val_running_loss / val_batches
    accuracy = 100 * correct / total
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✓ Saved new best model ({best_accuracy:.2f}%)")
    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_running_loss/train_batches:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {accuracy:.2f}%"
    )
print(f"\nBest Validation Accuracy: {best_accuracy:.2f}%")
    
    

