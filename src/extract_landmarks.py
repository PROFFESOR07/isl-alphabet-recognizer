"""
Extracts MediaPipe Hand Landmarks from raw dataset images.
Saves precomputed 218-dim invariant geometric feature vectors and labels to data/processed/landmarks.npz.
"""
import os
import sys
from pathlib import Path
import time
import numpy as np
import cv2
import mediapipe as mp
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DATASET_ROOTS, CLASSES, class_to_idx, HF_1_MAPPING
from src.landmark_features import extract_hand_features

OUTPUT_PATH = PROJECT_ROOT / "data/processed/landmarks.npz"
MAX_SAMPLES_PER_CLASS = 300  # 10,800 rich balanced landmark vectors across all 36 classes


def main():
    print("=" * 60)
    print("ISL 218-Dim Landmark Feature Extraction")
    print("=" * 60)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.4
    )
    
    all_features = []
    all_labels = []
    all_hand_counts = []
    
    class_counts = Counter()
    
    print("\nScanning raw datasets...")
    class_image_map = {cls: [] for cls in CLASSES}
    
    for root in DATASET_ROOTS:
        if not root.exists():
            continue
        is_hf1 = root.name == "HF_DATASET_1"
        
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith("."):
                continue
                
            cls_name = HF_1_MAPPING.get(class_dir.name) if is_hf1 else class_dir.name
            if not cls_name or cls_name not in class_to_idx:
                continue
                
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    class_image_map[cls_name].append(img_path)
                    
    print("\nExtracting 218-dim landmarks per class...")
    start_time = time.time()
    
    for idx_cls, cls_name in enumerate(CLASSES, 1):
        img_list = class_image_map[cls_name]
        np.random.seed(42)
        np.random.shuffle(img_list)
        
        target_label = class_to_idx[cls_name]
        extracted_for_class = 0
        cls_t0 = time.time()
        
        for img_path in img_list:
            if extracted_for_class >= MAX_SAMPLES_PER_CLASS:
                break
                
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            if img.shape[0] > 256 or img.shape[1] > 256:
                img = cv2.resize(img, (256, 256))
                
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            if not results.multi_hand_landmarks:
                continue
                
            feats, num_hands = extract_hand_features(results.multi_hand_landmarks)
            
            all_features.append(feats)
            all_labels.append(target_label)
            all_hand_counts.append(num_hands)
            
            extracted_for_class += 1
            class_counts[cls_name] += 1
            
        cls_time = time.time() - cls_t0
        print(f"[{idx_cls:2d}/36] Class {cls_name:2s}: {extracted_for_class:3d} landmarks extracted ({cls_time:.1f}s)", flush=True)
        
    hands.close()
    
    features_arr = np.array(all_features, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)
    hand_counts_arr = np.array(all_hand_counts, dtype=np.int32)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        features=features_arr,
        labels=labels_arr,
        hand_counts=hand_counts_arr,
        classes=np.array(CLASSES)
    )
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✓ 218-Dim Landmark Extraction Complete in {elapsed:.1f}s!")
    print(f"✓ Total Landmark Vectors: {len(features_arr):,}")
    print(f"✓ Feature Matrix Shape:  {features_arr.shape}")
    print(f"✓ Saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
