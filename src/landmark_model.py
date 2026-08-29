"""
Landmark Neural Network Classifier for Indian Sign Language.
Deep Residual Landmark Architecture with 218 Geometric Input Features.
"""
import torch
import torch.nn as nn
from src.dataset import classes


class LandmarkBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        res = x
        out = self.act(self.bn1(self.fc1(x)))
        out = self.drop(self.bn2(self.fc2(out)))
        return self.act(out + res)


class LandmarkClassifier(nn.Module):
    def __init__(self, input_dim=218, num_classes=36, hidden_dim=256, num_blocks=3, dropout_rate=0.2):
        super().__init__()
        
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.blocks = nn.ModuleList([
            LandmarkBlock(hidden_dim, dropout=dropout_rate)
            for _ in range(num_blocks)
        ])
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, hand_count=None, apply_mask=False):
        feats = self.in_proj(x)
        for block in self.blocks:
            feats = block(feats)
        return self.head(feats)


def create_landmark_model(num_classes=len(classes)):
    return LandmarkClassifier(input_dim=218, num_classes=num_classes)
