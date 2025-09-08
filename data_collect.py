import cv2, mediapipe as mp, numpy as np, pandas as pd, os
from utils.hand_features import normalize_landmarks

DATA_PATH = "data/hand_samples.csv"
LABEL_KEYS = {ord('o'):'open_palm', ord('f'):'fist', ord('r'):'point_right',
              ord('l'):'point_left', ord('u'):'thumbs_up', ord('d'):'thumbs_down'}

def append_sample(label, feat):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    row = pd.DataFrame([{'label': label, **{f'f{i}': v for i,v in enumerate(feat)}}])
    row.to_csv(DATA_PATH, mode='a', index=False, header=not os.path.exists(DATA_PATH))

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    print("Press o/f/r/l/u/d to save a sample, q to quit.")
    while True:
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image)
        if result.multi_hand_landmarks:
            pts = np.array([[lm.x,lm.y,lm.z] for lm in result.multi_hand_landmarks[0].landmark])
            feat = normalize_landmarks(pts)
            mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        else:
            feat = None

        cv2.imshow("Collect", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if feat is not None and key in LABEL_KEYS:
            append_sample(LABEL_KEYS[key], feat)
            print("Saved", LABEL_KEYS[key])

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
