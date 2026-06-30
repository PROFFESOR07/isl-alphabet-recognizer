from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=(0.85,1.15),contrast=(0.85,1.15)),
    transforms.ToTensor(),#PIL->tensor
    transforms.Normalize(
        mean = IMAGENET_MEAN,
        std = IMAGENET_STD)
]
)

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),#PIL->tensor
    transforms.Normalize(
        mean = IMAGENET_MEAN,
        std = IMAGENET_STD)
]
)

