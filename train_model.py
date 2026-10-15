# Step-by-Step Simple Model Training Script
# This script trains a Random Forest Classifier to predict Airline Passenger Satisfaction

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Determine absolute directories
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data', 'airline_passenger_satisfaction.csv')
models_dir = os.path.join(current_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv(data_path)

# 2. Convert text columns to numbers (Label Encoding)
print("Converting categories to numbers...")
# Map Class: Business -> 0, Eco -> 1, Eco Plus -> 2
df['Class'] = df['Class'].map({'Business': 0, 'Eco': 1, 'Eco Plus': 2})
# Map Target: neutral_or_dissatisfied -> 0, satisfied -> 1
df['satisfaction'] = df['satisfaction'].map({'neutral_or_dissatisfied': 0, 'satisfied': 1})

# 3. Handle missing values (fill them with the median of each column)
print("Handling missing values...")
df = df.fillna(df.median(numeric_only=True))

# 4. Select the top 10 features for training
print("Selecting top 10 features...")
features_list = [
    'Online_boarding',
    'Cleanliness',
    'Seat_comfort',
    'Inflight_entertainment',
    'Inflight_wifi_service',
    'Ease_of_Online_booking',
    'Flight_Distance',
    'On-board_service',
    'Class',
    'Arrival_Delay_in_Minutes'
]

X = df[features_list]
y = df['satisfaction']

# 5. Split the data into Train (80%) and Test (20%) sets
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Scale the features (Normalize data to standard scale)
print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Create and train the Machine Learning Model (Random Forest)
print("Training the Random Forest model (this might take a few seconds)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 8. Test the model and print the accuracy score
print("Evaluating the model...")
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(f"\nSuccess! Model Test Accuracy is: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 9. Save the trained model, scaler, and metadata to folders
print("\nSaving files for Streamlit deployment...")
joblib.dump(model, os.path.join(models_dir, 'passenger_satisfaction_model.pkl'))
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))

# Save metadata (mapping details) for future reference
metadata = {
    'mappings': {
        'Class': {'Business': 0, 'Eco': 1, 'Eco Plus': 2},
        'satisfaction': {'neutral_or_dissatisfied': 0, 'satisfied': 1}
    },
    'feature_names': features_list
}
joblib.dump(metadata, os.path.join(models_dir, 'metadata.pkl'))
print(f"All files saved successfully inside '{models_dir}' directory!")
