import numpy as np

def normalize_landmarks(landmarks):
    lm = landmarks.copy().astype(np.float32)
    wrist = lm[0, :]
    lm -= wrist
    scale = np.max(np.linalg.norm(lm[:, :2], axis=1)) + 1e-6
    lm[:, :2] /= scale
    lm[:, 2] /= scale
    return lm.reshape(-1)

def rolling_mode(labels, window=5):
    if len(labels) == 0:
        return None
    recent = labels[-window:]
    values, counts = np.unique(recent, return_counts=True)
    return values[np.argmax(counts)]
