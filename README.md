
[ The signs used in this project follow standard BdSL alphabet charts. The dataset was self-recorded as part of an portfolio project. Detection is based on static hand pose (single frame), not continuous gesture sequences. ]


# Bangla Sign Language Recognition (BdSL)

A real-time Bangla Sign Language recognition system built using computer vision and machine learning. It detects hand signs via webcam and converts them into Bangla text output.

## Overview

This is a personal portfolio project exploring computer vision and machine learning for Bangla Sign Language (BdSL) alphabet recognition. Hand landmarks are captured using Mediapipe and classified using a Random Forest model.

## Features

- Real-time hand landmark detection using Mediapipe
- Custom, self-recorded dataset
- Random Forest classifier for sign prediction
- Live webcam-based prediction with Bangla text output

## Signs Covered

Currently supports 10 Bangla alphabets: অ, আ, ই, ঈ, উ, ক, খ, গ, ঘ, ঙ

## Tech Stack

- Python
- OpenCV
- Mediapipe (Hand Landmarker)
- Scikit-learn (Random Forest)
- Pandas / NumPy
- Pillow (for rendering Bangla text)


## Project Structure
bangla-sign-language-

├── data/ # Collected CSV files (hand landmark data)

├── models/ # Trained model file

├── scripts/

│ ├── collect_data.py # Data collection script

│ ├── train_model.py # Model training script

│ └── predict_live.py # Real-time prediction script

└── README.md




# বাংলা সাংকেতিক ভাষা শনাক্তকরণ (BdSL)

কম্পিউটার ভিশন ও মেশিন লার্নিং ব্যবহার করে রিয়েল-টাইম বাংলা সাংকেতিক ভাষা (Sign Language) শনাক্তকরণ প্রকল্প, যা হাতের ইশারা দেখে বাংলা টেক্সটে রূপান্তর করে।

## প্রকল্পের সংক্ষিপ্ত বিবরণ

এটি একটি ব্যক্তিগত পোর্টফোলিও প্রকল্প, যেখানে কম্পিউটার ভিশন ও মেশিন লার্নিং ব্যবহার করে বাংলা সাংকেতিক ভাষার (BdSL) বর্ণমালা শনাক্ত করার চেষ্টা করা হয়েছে। Mediapipe ব্যবহার করে হাতের ল্যান্ডমার্ক (landmark) সংগ্রহ করা হয় এবং Random Forest মডেল দিয়ে তা শ্রেণীবদ্ধ (classify) করা হয়।

## বৈশিষ্ট্য

- Mediapipe ব্যবহার করে রিয়েল-টাইম হাতের ল্যান্ডমার্ক শনাক্তকরণ
- নিজে তৈরি করা ডেটাসেট (self-recorded)
- সাইন শনাক্তকরণের জন্য Random Forest ক্লাসিফায়ার
- লাইভ ওয়েবক্যামের মাধ্যমে বাংলা টেক্সট আউটপুট

## অন্তর্ভুক্ত বর্ণসমূহ

বর্তমানে ১০টি বাংলা বর্ণ সমর্থিত: অ, আ, ই, ঈ, উ, ক, খ, গ, ঘ, ঙ

## ব্যবহৃত প্রযুক্তি

- Python
- OpenCV
- Mediapipe (Hand Landmarker)
- Scikit-learn (Random Forest)
- Pandas / NumPy
- Pillow (বাংলা টেক্সট রেন্ডারিং এর জন্য)

## Project Structure
bangla-sign-language-

├── data/ # Collected CSV files (hand landmark data)

├── models/ # Trained model file

├── scripts/

│ ├── collect_data.py # Data collection script

│ ├── train_model.py # Model training script

│ └── predict_live.py # Real-time prediction script
└── README.md



## Future Improvements

- Expand to the full Bangla alphabet (50+ characters)
- Add common word/phrase recognition
- Implement sequence-based (LSTM) models for dynamic gestures
- Build a simple web/Streamlit interface

## Author

Mahmudul Hasan (Arik)
Final-year CSE Student, Daffodil International University
7-23-26

