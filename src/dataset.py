# Dataset loading and splitting
from pathlib import Path
from collections import Counter
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from src.transforms import train_transform, val_transform


current_file = Path(__file__)
DATA_DIR = current_file.parent.parent / "data/raw"

DATASET_ROOTS = [
    DATA_DIR / "indian-sign-language",
    DATA_DIR / "pratham-isl",
    DATA_DIR / "Dataset",
    DATA_DIR / "Test 1-9 + a-z",
    DATA_DIR / "train",
    DATA_DIR / "HF_DATASET_1",
]

CLASSES = [
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z",
]

class_to_idx = {
    cls: idx
    for idx, cls in enumerate(CLASSES)
}

idx_to_class = {
    idx: cls
    for cls, idx in class_to_idx.items()
}

classes = CLASSES

# Mapping for HF_DATASET_1 (0-9: digits, 10-32: alphabets)
HF_1_MAPPING = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "10": "A", "11": "B", "12": "C", "13": "D", "14": "E",
    "15": "F", "16": "G", "17": "I", "18": "K", "19": "L",
    "20": "M", "21": "N", "22": "O", "23": "P", "24": "Q",
    "25": "R", "26": "S", "27": "T", "28": "U", "29": "V",
    "30": "W", "31": "X", "32": "Y"
}


class ISLDataset(Dataset):
    def __init__(self, roots, transform=None):
        self.transform = transform
        self.samples = []

        for root in roots:
            if not root.exists():
                continue

            is_hf1 = root.name == "HF_DATASET_1"

            for class_dir in sorted(root.iterdir()):
                if not class_dir.is_dir() or class_dir.name.startswith("."):
                    continue

                if is_hf1:
                    class_name = HF_1_MAPPING.get(class_dir.name)
                else:
                    class_name = class_dir.name

                if not class_name or class_name not in class_to_idx:
                    continue

                label = class_to_idx[class_name]

                for image_path in sorted(class_dir.glob("*")):
                    if image_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                        continue
                    self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


full_dataset = ISLDataset(DATASET_ROOTS)

generator = torch.Generator().manual_seed(42)

train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_subset, val_subset, test_subset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=generator,
)

train_dataset = torch.utils.data.Subset(
    ISLDataset(DATASET_ROOTS, train_transform),
    train_subset.indices,
)

val_dataset = torch.utils.data.Subset(
    ISLDataset(DATASET_ROOTS, val_transform),
    val_subset.indices,
)

test_dataset = torch.utils.data.Subset(
    ISLDataset(DATASET_ROOTS, val_transform),
    test_subset.indices,
)

BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


def print_dataset_stats():
    print(f"Total images loaded: {len(full_dataset)}")
    counter = Counter()
    for _, label in full_dataset.samples:
        counter[label] += 1

    print("\nClass distribution across all datasets:\n")
    for idx, count in sorted(counter.items()):
        print(f"{idx_to_class[idx]} : {count}")

    print(f"\nTrain samples: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"Val samples:   {len(val_dataset)} ({len(val_loader)} batches)")
    print(f"Test samples:  {len(test_dataset)} ({len(test_loader)} batches)")


if __name__ == "__main__":
    print_dataset_stats()