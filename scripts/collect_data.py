# Bangla Sign Language - Data Collection Script
# Ei script webcam theke hand landmarks capture kore CSV file e save kore
# Notun mediapipe Tasks API use kora hoyeche (0.10+ version er jonno)

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import csv
import os

# Hand landmarker setup (notun API)
base_options = mp_python.BaseOptions(model_asset_path="scripts/hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7
)
landmarker = vision.HandLandmarker.create_from_options(options)

# Kon sign er data collect korchi, seta change korte hobe প্রতিবার
sign_label = input("Kon sign er data collect korbe? Label likho: ")

# Data folder na thakle banao
if not os.path.exists("data"):
    os.makedirs("data")

csv_filename = "data/" + sign_label + ".csv"

# Camera চালু করি
cap = cv2.VideoCapture(0)

# Koto gulo sample collect korbo
max_samples = 200
sample_count = 0

# CSV file open kori লেখার জন্য
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)

print("Camera chalu hocche... 's' chapo sample save korte, 'q' chapo bondo korte")

while True:
    success, frame = cap.read()
    if not success:
        print("Camera theke frame paoa jacche na")
        break

    # Frame ke horizontally flip kori (mirror effect, dekhte সহজ hobe)
    frame = cv2.flip(frame, 1)

    # Mediapipe RGB format e kaj kore, tai convert kori
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Hand detect kori
    result = landmarker.detect(mp_image)

    landmark_row = []

    # Jodi hand detect hoy
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Hand er upor landmarks draw kori (dekhar jonno)
            h, w, _ = frame.shape
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                landmark_row.append(lm.x)
                landmark_row.append(lm.y)
                landmark_row.append(lm.z)

        # Screen e text dekhai koto sample hoise
        cv2.putText(frame, f"Samples: {sample_count}/{max_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Data Collection - Bangla Sign Language", frame)

    key = cv2.waitKey(1) & 0xFF

    # 's' chaple ei landmark data CSV e save hobe
    if key == ord('s') and landmark_row and sample_count < max_samples:
        landmark_row.append(sign_label)  # last column e label add kori
        csv_writer.writerow(landmark_row)
        sample_count += 1
        print(f"Sample {sample_count} saved")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
print(f"Total {sample_count} samples saved to {csv_filename}")