# Bangla Sign Language - Real-time Prediction Script
# Ei script webcam theke live hand sign dekhe ki sign seta bole dey
# Bangla text dekhanor jonno Pillow (PIL) library use kora hoyeche

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pickle
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# Trained model load kori
with open("models/sign_model.pkl", "rb") as f:
    model = pickle.load(f)

# Label mapping - terminal e ja likhechilam, tar Bangla meaning
label_map = {
    "o": "অ",
    "aa": "আ",
    "i": "ই",
    "ii": "ঈ",
    "u": "উ",
    "ka": "ক",
    "kha": "খ",
    "ga": "গ",
    "ghagha": "ঘ",
    "uma": "ঙ"
}

# Bangla font load kori (Windows er built-in font)
bangla_font = ImageFont.truetype("C:/Windows/Fonts/Nirmala.ttc", 60)

# Hand landmarker setup
base_options = mp_python.BaseOptions(model_asset_path="scripts/hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7
)
landmarker = vision.HandLandmarker.create_from_options(options)

# Camera চালু করি
cap = cv2.VideoCapture(0)

print("Camera chalu hocche... 'q' chapo bondo korte")

while True:
    success, frame = cap.read()
    if not success:
        print("Camera theke frame paoa jacche na")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect(mp_image)

    predicted_text = ""

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            h, w, _ = frame.shape
            landmark_row = []
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                landmark_row.append(lm.x)
                landmark_row.append(lm.y)
                landmark_row.append(lm.z)

            # Model diye predict kori
            landmark_array = np.array(landmark_row).reshape(1, -1)
            prediction = model.predict(landmark_array)[0]

            # Bangla label e convert kori
            predicted_text = label_map.get(prediction, prediction)
            print("Predicted:", prediction)
    # OpenCV frame ke PIL image e convert kori (Bangla text draw korar jonno)
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    draw.text((10, 10), predicted_text, font=bangla_font, fill=(0, 255, 255))

    # Abar OpenCV format e ferot ani
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("Bangla Sign Language - Live Prediction", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()