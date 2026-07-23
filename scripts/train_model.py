#Ekahne ja ja korsi :
# Bangla Sign Language - Model Training Script
# Ei script sob CSV file theke data nia ekta Random Forest model train kore kaj korsi 

import pandas as pd
import glob
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Data folder theke sob CSV file er list nei
csv_files = glob.glob("data/*.csv")
print("Pawa gelo file gulo:", csv_files)

# Sob CSV file ekshathe joro kori
all_data = []
for file in csv_files:
    df = pd.read_csv(file, header=None)
    all_data.append(df)

# Sob data ekta DataFrame e combine kori
full_data = pd.concat(all_data, ignore_index=True)
print("Total sample:", len(full_data))

# Last column label (sign name), baki sob feature (landmark coordinates)
X = full_data.iloc[:, :-1]   # sob column, last ta bade
y = full_data.iloc[:, -1]    # sudhu last column (label)

# Data ke train o test e vag kori (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest model banai
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test data diye accuracy check kori
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nDetailed Report:")
print(classification_report(y_test, predictions))


with open("models/sign_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model save hoye gelo: models/sign_model.pkl")