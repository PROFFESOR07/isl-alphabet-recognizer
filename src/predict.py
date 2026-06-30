from src.model import create_model
from src.transforms import val_transform
from src.utils import get_device

import torch
import cv2
from PIL import Image
from pathlib import Path
from src.dataset import classes, test_subset
import mediapipe as mp

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

device = get_device()
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def load_model():
    model = create_model()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.to(device)
    model.eval()
    return model

def predict_image(model, image):
    tensor = val_transform(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)
    with torch.no_grad():
        outputs = model(tensor)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, prediction = torch.max(probabilities, dim=1)

    return classes[prediction.item()], confidence.item()

def webcam():
    model = load_model()
    prediction = "No Hand"
    confidence = 0.0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        height, width, _ = frame.shape

        if results.multi_hand_landmarks:
            x_coords = []
            y_coords = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                for landmark in hand_landmarks.landmark:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    x_coords.append(x)
                    y_coords.append(y)

            xmin = min(x_coords)
            xmax = max(x_coords)

            ymin = min(y_coords)
            ymax = max(y_coords)

            padding = 10

            xmin = max(0, xmin - padding)
            ymin = max(0, ymin - padding)

            xmax = min(width, xmax + padding)
            ymax = min(height, ymax + padding)

            box_width = xmax - xmin
            box_height = ymax - ymin
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2       
            box_size = max(box_width, box_height)
            half = box_size // 2

            xmin = max(0, center_x - half)
            xmax = min(width, center_x + half)

            ymin = max(0, center_y - half)
            ymax = min(height, center_y + half)

            
            cv2.rectangle(
                frame,
                (xmin, ymin),
                (xmax, ymax),
                (255, 0, 0),
                2
            )
            cropped = rgb_frame[ymin:ymax, xmin:xmax]

            image = Image.fromarray(cropped)

            prediction, confidence = predict_image(model, image)
            cv2.imshow("Crop", cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

        cv2.putText(
            frame,
            f"{prediction} ({confidence*100:.1f}%)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.imshow("ISL Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    webcam()
