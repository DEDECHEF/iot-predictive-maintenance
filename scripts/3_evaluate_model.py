"""
==============================================================================
STEP 3: MODEL ACCURACY EVALUATION (OPTIONAL)
==============================================================================
- Purpose: This script is not used in production. Its sole function is to
  evaluate the accuracy of the AI model we trained in step 2.

- How it works: It simulates a real-world scenario where the model, which 
  has only seen "healthy" data, is exposed to "faulty" data for the first time. 
  At the end, it prints a report and a confusion matrix telling us how good 
  the model is at distinguishing between states.
==============================================================================
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load both files separately
print("Loading files...")
df_healthy_file = pd.read_csv('fan_dataset_normal.csv')
df_fault_file = pd.read_csv('fan_dataset_imbalance_3.csv')

# 2. Append them together
# ignore_index=True ensures continuous row numbering
df = pd.concat([df_healthy_file, df_fault_file], ignore_index=True)

print(f"Dataset successfully merged. Total rows: {len(df)}")

# Define the physical columns we can see (< 50 Hz and Time domain)
feature_columns = ['RMS', 'Crest', 'Kurtosis', 'Low_E_0_10Hz', 'Mid_E_10_30Hz', 'High_E_30_50Hz']

# 2. THE REAL WORLD: Simulating a "Cold Start"
# Physically separate healthy data from broken data using the 'Normal' label
df_healthy = df[df['Label'] == 'Normal']
df_broken = df[df['Label'] != 'Normal']

# Extract only numerical values for the model to process
X_healthy = df_healthy[feature_columns].values
X_broken = df_broken[feature_columns].values

print(f"\nTotal healthy recordings (Baseline): {len(X_healthy)}")
print(f"Total recordings with hidden faults: {len(X_broken)}")

# Take 80% of healthy data to "learn normality" (Our training period)
X_train_healthy, X_test_healthy = train_test_split(X_healthy, test_size=0.2, random_state=42)

# The final test will be: the remaining 20% of healthy data + ALL broken data the AI has never seen
X_test = np.vstack((X_test_healthy, X_broken))

# In Isolation Forest: 1 means "Healthy/Normal", -1 means "Anomaly"
y_test_real = np.array([1] * len(X_test_healthy) + [-1] * len(X_broken))

# 3. UNSUPERVISED TRAINING (Isolation Forest)
print("\nTraining Isolation Forest ONLY with healthy data...")
# contamination = 0.01 because we know 99% of our train data is pure healthy
if_model = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
if_model.fit(X_train_healthy)

# 4. EVALUATION
predictions = if_model.predict(X_test)
accuracy = accuracy_score(y_test_real, predictions)

print(f"\nANOMALY DETECTION ACCURACY: {accuracy * 100:.2f}%\n")
print("--- ANOMALY REPORT (1 = Healthy, -1 = Fault) ---")
print(classification_report(y_test_real, predictions, target_names=['Fault Detected (-1)', 'Healthy State (1)']))

print("\nConfusion Matrix:")
print("Row 1: Real faults. Col 1 = Detected faults, Col 2 = False Healthy (Missed faults)")
print("Row 2: Real healthy. Col 1 = False Alarms, Col 2 = Correct healthy")
print(confusion_matrix(y_test_real, predictions))