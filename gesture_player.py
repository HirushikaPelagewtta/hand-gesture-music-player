import cv2, mediapipe as mp, numpy as np, joblib, glob, pygame, os
from utils.hand_features import normalize_landmarks

MODEL_PATH = "models/gesture_svm.joblib"
MUSIC_DIR = "music"
model = joblib.load(MODEL_PATH)

# Load music
playlist = glob.glob(os.path.join(MUSIC_DIR, "*.mp3")) + glob.glob(os.path.join(MUSIC_DIR, "*.wav"))
pygame.mixer.init(); track_i = 0

def play(): pygame.mixer.music.load(playlist[track_i]); pygame.mixer.music.play()
def stop(): pygame.mixer.music.stop()
def next_track(): 
    global track_i; track_i = (track_i+1) % len(playlist); play()
def prev_track():
    global track_i; track_i = (track_i-1) % len(playlist); play()
def vol_up(): pygame.mixer.music.set_volume(min(1, pygame.mixer.music.get_volume()+0.1))
def vol_down(): pygame.mixer.music.set_volume(max(0, pygame.mixer.music.get_volume()-0.1))

actions = {
    'open_palm': lambda: pygame.mixer.music.pause() if pygame.mixer.music.get_busy() else play(),
    'fist': stop, 'point_right': next_track, 'point_left': prev_track,
    'thumbs_up': vol_up, 'thumbs_down': vol_down
}

# Webcam loop
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

gesture = None
last_gesture = None
stable_count = 0
TRIGGER_THRESHOLD = 15  # how many frames must match before triggering

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    if result.multi_hand_landmarks:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in result.multi_hand_landmarks[0].landmark])
        feat = normalize_landmarks(pts).reshape(1, -1)
        pred = model.predict(feat)[0]
        gesture = pred

        # Check stability
        if gesture == last_gesture:
            stable_count += 1
        else:
            stable_count = 0  # reset if gesture changes

        # Trigger only once when gesture is stable long enough
        if stable_count == TRIGGER_THRESHOLD and gesture in actions:
            print(f"Executing action: {gesture}")
            actions[gesture]()

        last_gesture = gesture

        # Show current gesture
        cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Player", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
