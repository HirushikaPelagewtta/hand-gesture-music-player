# 🎵 Hand Gesture Controlled Music Player

Control your music player in **real-time** using simple **hand gestures** captured from your webcam.  
This project combines **computer vision** (gesture recognition) with **Python audio playback** for an interactive experience.

---

## ✨ Features
- 🎥 **Real-time hand gesture recognition** using [MediaPipe](https://developers.google.com/mediapipe)  
- 🖐️ Supported gestures: **Play/Pause, Stop, Next Track, Previous Track, Volume Up/Down**  
- 🤖 Gesture classification with an **SVM model** trained on custom data  
- 🎶 Music playback with **pygame**  
- ⚡ Stable recognition with **debouncing mechanism** to avoid accidental triggers  

---

## 🚀 How It Works
1. **Data Collection**: Capture hand landmark data using MediaPipe (`data_collect.py`).  
2. **Training**: Train an SVM classifier with scikit-learn (`train.py`).  
3. **Run Player**: Launch the real-time player (`gesture_player.py`) to control music with gestures.  

---

## 🛠️ Tech Stack
- **Python 3.12**
- **OpenCV** – video capture & visualization  
- **MediaPipe** – hand landmark detection  
- **scikit-learn** – SVM model training  
- **pygame** – music playback  
- **NumPy / Pandas / Joblib** – data handling & model persistence  

---

## 📂 Project Structure

```

hand-gesture-music-player/
├─ data/              # Collected gesture samples (CSV)
├─ models/            # Trained gesture classifier (joblib)
├─ music/             # Add your .mp3/.wav files here
├─ utils/
│  └─ hand\_features.py
├─ data\_collect.py    # Collect labeled hand gesture samples
├─ train.py           # Train the SVM classifier
├─ gesture\_player.py  # Real-time music player controlled by gestures
├─ requirements.txt
└─ README.md

````


## ⚡ Quickstart
1. Clone the repo:
   ```bash
   git clone https://github.com/HirushikaPelagewtta/hand-gesture-music-player.git
   cd hand-gesture-music-player
   
2. Create a Python 3.12 environment and install dependencies:

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate   # (Windows)
   pip install -r requirements.txt
   ```

3. Collect gesture data:

   ```bash
   python data_collect.py
   ```

4. Train the classifier:

   ```bash
   python train.py
   ```

5. Add music files to the `music/` folder.

6. Run the player:

   ```bash
   python gesture_player.py
   ```

---

## 📸 Demo

(Add a GIF or screenshot of the system in action here)

---

## 🔮 Future Improvements

* Extend to more gestures (mute, shuffle, playlist control)
* Deploy with a lightweight GUI for non-technical users
* Experiment with deep learning models for improved accuracy

---

## 📜 License

This project is licensed under the MIT License.

```

