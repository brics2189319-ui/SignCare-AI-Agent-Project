import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

print("🏥 SignCare AI System Initializing...")

# 1. Load the data
csv_path = os.path.join('Data', 'alphabet_data.csv')
if not os.path.exists(csv_path):
    print("❌ Error: CSV file not found! Please run collect_data.py first.")
    exit()

df = pd.read_csv(csv_path)

# 2. Separate the "Features" (coordinates) from the "Label" (A, B, C, etc.)
X = df.drop('label', axis=1) 
y = df['label']              

# 3. Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

print("🧠 Training the Medical Agent's brain... this usually takes less than 5 seconds.")

# 4. Create and Train the Brain (Random Forest)
model = RandomForestClassifier(n_estimators=100) 
model.fit(X_train, y_train)

# 5. Check the Accuracy
y_predict = model.predict(X_test)
score = accuracy_score(y_test, y_predict)

print(f"✅ Training Complete! Agent Accuracy: {score * 100:.2f}%")

# 6. Save the brain into your 'model' folder
if not os.path.exists('model'):
    os.makedirs('model')

model_path = os.path.join('model', 'sign_language_model.p')
with open(model_path, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"💾 SignCare Agent model saved successfully at: {model_path}")