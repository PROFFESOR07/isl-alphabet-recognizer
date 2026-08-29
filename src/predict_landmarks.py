"""
Real-Time ISL Alphabet Recognition with MediaPipe Landmarks and Hand-Count Routing.
"""
import sys
from pathlib import Path
from collections import deque, Counter
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import mediapipe as mp

from src.landmark_model import create_landmark_model
from src.landmark_features import extract_hand_features
from src.dataset import classes
from src.utils import get_device

MODEL_PATH = PROJECT_ROOT / "models/best_landmark_model.pth"

CONFIDENCE_THRESHOLD = 0.65
MARGIN_THRESHOLD = 0.15
HISTORY_LEN = 5


def load_landmark_detector(device):
    model = create_landmark_model(num_classes=len(classes))
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}. Run src/train_landmarks.py first.")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def main():
    device = get_device()
    print(f"Loading Landmark Recognition Model on {device}...")
    model = load_landmark_detector(device)
    
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_draw_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Failed to open webcam.")
        return
        
    print("\n" + "=" * 50)
    print("ISL Landmark Recognition Started!")
    print("Press 'q' in the webcam window to quit.")
    print("=" * 50 + "\n")
    
    pred_history = deque(maxlen=HISTORY_LEN)
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break
            
        frame = cv2.flip(frame, 1)  # Mirror view
        h, w, _ = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        
        display_label = "No Hand"
        display_conf = 0.0
        hand_count_text = "Hands: 0"
        
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            hand_count_text = f"Hands: {num_hands}"
            
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw_styles.get_default_hand_landmarks_style(),
                    mp_draw_styles.get_default_hand_connections_style()
                )
                
            feats, hand_count = extract_hand_features(results.multi_hand_landmarks)
            feat_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(feat_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                
                top_probs, top_indices = torch.topk(probs, k=2)
                top1_conf = top_probs[0].item()
                top2_conf = top_probs[1].item()
                top1_class = classes[top_indices[0].item()]
                margin = top1_conf - top2_conf
                
                if top1_conf >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD:
                    pred_history.append(top1_class)
                    display_conf = top1_conf
                else:
                    pred_history.append("Uncertain")
                    
            if pred_history:
                most_common = Counter([p for p in pred_history if p != "Uncertain"]).most_common(1)
                if most_common:
                    display_label = most_common[0][0]
                else:
                    display_label = "Uncertain"
        else:
            pred_history.clear()
            display_label = "No Hand"
            display_conf = 0.0

        cv2.rectangle(frame, (10, 10), (360, 110), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (360, 110), (100, 255, 100), 2)
        
        status_color = (0, 255, 0) if display_label not in ["No Hand", "Uncertain"] else (0, 165, 255)
        cv2.putText(frame, f"Sign: {display_label}", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        
        conf_str = f"Conf: {display_conf*100:.1f}%" if display_conf > 0 else "Conf: --"
        cv2.putText(frame, f"{conf_str} | {hand_count_text} | FPS: {int(fps)}", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        
        cv2.imshow("ISL Real-Time Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
