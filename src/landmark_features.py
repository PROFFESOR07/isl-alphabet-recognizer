"""
Enhanced Landmark Feature Engineering for Indian Sign Language (ISL) Recognition.
Extracts 218 rotation/scale/position-invariant geometric features from MediaPipe Hand Landmarks.
Features include:
- Left-to-Right canonical hand sorting
- Normalized 3D coordinates
- Radial fingertip-to-wrist distances
- Adjacent fingertip distances
- 15 3D finger joint angles per hand (scale & orientation invariant)
- Comprehensive inter-hand fingertip & wrist interaction matrices
"""
import numpy as np

FEATURE_DIM = 218

TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
ADJACENT_PAIRS = [(4, 8), (8, 12), (12, 16), (16, 20), (4, 20)]

# Triplets for computing joint angles (A, B, C) where joint is at B
FINGER_JOINTS = [
    # Thumb: CMC, MCP, IP
    [(0, 1, 2), (1, 2, 3), (2, 3, 4)],
    # Index: MCP, PIP, DIP
    [(0, 5, 6), (5, 6, 7), (6, 7, 8)],
    # Middle: MCP, PIP, DIP
    [(0, 9, 10), (9, 10, 11), (10, 11, 12)],
    # Ring: MCP, PIP, DIP
    [(0, 13, 14), (13, 14, 15), (14, 15, 16)],
    # Pinky: MCP, PIP, DIP
    [(0, 17, 18), (17, 18, 19), (18, 19, 20)],
]


def compute_joint_angles(hand_coords):
    """Computes 15 cosine joint angles for a single hand (3 angles x 5 fingers)."""
    angles = []
    for finger in FINGER_JOINTS:
        for a, b, c in finger:
            v1 = hand_coords[b] - hand_coords[a]
            v2 = hand_coords[c] - hand_coords[b]
            norm1 = np.linalg.norm(v1) + 1e-7
            norm2 = np.linalg.norm(v2) + 1e-7
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            angles.append(np.clip(cos_angle, -1.0, 1.0))
    return np.array(angles, dtype=np.float32)


def extract_single_hand_features(hand_coords):
    """
    Extracts 89 features from a single hand:
    - 63 normalized coords
    - 5 fingertip-to-wrist distances
    - 5 adjacent fingertip distances
    - 15 joint angles
    - 1 presence flag
    """
    feats = np.zeros(89, dtype=np.float32)
    w = hand_coords[0]    # Wrist
    mcp = hand_coords[9]  # Middle MCP
    scale = np.linalg.norm(mcp - w) + 1e-6
    
    # 1. Normalized coords (63)
    feats[0:63] = ((hand_coords - w) / scale).flatten()
    
    # 2. Radial fingertip distances (5)
    for i, tip in enumerate(TIPS):
        feats[63 + i] = np.linalg.norm(hand_coords[tip] - w) / scale
        
    # 3. Adjacent fingertip distances (5)
    for i, (t1, t2) in enumerate(ADJACENT_PAIRS):
        feats[68 + i] = np.linalg.norm(hand_coords[t1] - hand_coords[t2]) / scale
        
    # 4. Joint angles (15)
    feats[73:88] = compute_joint_angles(hand_coords)
    
    # 5. Presence flag (1)
    feats[88] = 1.0
    
    return feats, scale, w


def extract_hand_features(landmarks_list):
    """
    Extracts a 218-dimensional feature vector from detected hands with canonical L-to-R sorting.
    
    Returns:
    - feats: np.ndarray of shape (218,), dtype float32
    - num_hands: int (0, 1, or 2)
    """
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)
    if not landmarks_list:
        return feats, 0

    # Canonical Left-to-Right sorting by wrist X-coordinate
    if len(landmarks_list) >= 2:
        sorted_lms = sorted(landmarks_list[:2], key=lambda lms: lms.landmark[0].x)
    else:
        sorted_lms = landmarks_list[:1]

    num_hands = len(sorted_lms)
    
    # Hand 1 (Leftmost hand) -> features 0:89
    h1 = np.array([[lm.x, lm.y, lm.z] for lm in sorted_lms[0].landmark], dtype=np.float32)
    h1_feats, scale1, w1 = extract_single_hand_features(h1)
    feats[0:89] = h1_feats
    
    if num_hands >= 2:
        # Hand 2 (Rightmost hand) -> features 89:178
        h2 = np.array([[lm.x, lm.y, lm.z] for lm in sorted_lms[1].landmark], dtype=np.float32)
        h2_feats, scale2, w2 = extract_single_hand_features(h2)
        feats[89:178] = h2_feats
        
        # Inter-Hand Features -> features 178:218 (40 dims)
        # 1. Inter-wrist displacement & distance (4 dims)
        feats[178:181] = (w2 - w1) / scale1
        feats[181] = np.linalg.norm(w2 - w1) / scale1
        
        # 2. Pairwise 5x5 Fingertip-to-Fingertip Distances (25 dims)
        offset = 182
        for t1 in TIPS:
            for t2 in TIPS:
                feats[offset] = np.linalg.norm(h1[t1] - h2[t2]) / scale1
                offset += 1
                
        # 3. Wrist-to-Opposite-Fingertip Distances (10 dims)
        for t2 in TIPS:
            feats[offset] = np.linalg.norm(w1 - h2[t2]) / scale1
            offset += 1
        for t1 in TIPS:
            feats[offset] = np.linalg.norm(w2 - h1[t1]) / scale2
            offset += 1
            
        # 4. Hand Count Indicator (1 dim)
        feats[217] = 2.0
    else:
        feats[217] = 1.0

    return feats, num_hands
