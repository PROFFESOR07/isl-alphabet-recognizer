# Dataset loading and splitting
# /Users/pranjalchoudhary/Documents/CODE/isl-alphabet-recognizer/data/raw/indian-sign-language
from torchvision.datasets import ImageFolder
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from src.transforms import train_transform, val_transform


current_file = Path(__file__)
dataset_path = current_file.parent.parent/"data/raw/indian-sign-language" #path to file

dataset = ImageFolder(dataset_path)
classes = dataset.classes
class_to_idx = dataset.class_to_idx #loading dataset
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

generator = torch.Generator().manual_seed(42)
train_subset, val_subset, test_subset = random_split(
    dataset,
    [28800, 3600, 3600],
    generator=generator
)# dataset splitting

class TransformSubset(Dataset):
    """
Why this wrapper exists:

After random_split(), we get Subset objects. A Subset only knows
which samples belong to the train/validation/test split—it does not
support assigning different transforms.

This wrapper combines:
1. a Subset (which knows the split)
2. a transform (which knows how to preprocess images)

When a sample is requested:
TransformSubset -> Subset -> ImageFolder -> PIL Image
                 -> apply transform -> return (image, label)

This allows the training, validation, and test datasets to use
different preprocessing pipelines while sharing the same underlying
ImageFolder dataset.
"""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    


train_dataset = TransformSubset(train_subset, train_transform)
val_dataset = TransformSubset(val_subset, val_transform)
test_dataset = TransformSubset(test_subset, val_transform)

BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

