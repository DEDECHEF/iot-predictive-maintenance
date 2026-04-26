"""
==============================================================================
STEP 2: AI MODEL TRAINING
==============================================================================
- Purpose: This script is the "trainer" for the Artificial Intelligence.
  It takes the operational history of the machine in "Normal" state
  (generated with script 1) and uses it to train an Isolation Forest model.

- Result: Generates a single file, `fan_model.pkl`, which is the trained
  "brain" of the AI. This file must be uploaded to the Raspberry Pi for 
  the monitoring script to use.
==============================================================================
"""
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# 1. Load ONLY perfect operation data
healthy_file = 'fan_dataset_normal.csv'
print(f"Loading healthy heartbeat history from: {healthy_file}")
df_normal = pd.read_csv(healthy_file)

feature_columns = ['RMS', 'Crest', 'Kurtosis', 'Low_E_0_10Hz', 'Mid_E_10_30Hz', 'High_E_30_50Hz']
X_train = df_normal[feature_columns]

print(f"Assimilating {len(X_train)} windows of pure data...")

# 2. Master Training
# We use n_estimators=200 to make the "network" denser and more precise
master_model = IsolationForest(n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1)

# Feed it 100% of the good data
master_model.fit(X_train)

# 3. Saving (Exporting the brain)
file_name = 'fan_model.pkl'
joblib.dump(master_model, file_name)

print("\n" + "="*50)
print(f"🧠 BRAIN SUCCESSFULLY CREATED!")
print(f"File saved as: {file_name}")
print("Ready to be sent to the Raspberry Pi.")
print("="*50 + "\n")