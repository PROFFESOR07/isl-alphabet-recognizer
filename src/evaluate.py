from src.dataset import test_loader
from src.model import create_model
from src.utils import get_device

import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

device = get_device()

model = create_model()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)
model.to(device)
model.eval()

criterion = nn.CrossEntropyLoss()

test_running_loss = 0.0
test_batches = 0
correct_predictions = 0
total_samples = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, dim=1)

        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        test_running_loss += loss.item()
        test_batches += 1

test_loss = test_running_loss / test_batches
test_accuracy = 100 * correct_predictions / total_samples

print(
    f"Test Loss: {test_loss:.4f} | "
    f"Test Accuracy: {test_accuracy:.2f}%"
)