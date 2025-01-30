import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE  # For handling class imbalance

# 1. Load the Data from GitHub URL
github_url = "https://raw.githubusercontent.com/saumyasam/ML_For_Heart_Failure_Detect/main/heart_failure_clinical_records_dataset.csv"
try:
    df = pd.read_csv(github_url)
except Exception as e:
    print(f"Error loading data from GitHub: {e}")
    exit()  # Exit the script if data loading fails

# 2. Data Preprocessing (same as before)
# Convert boolean columns to integers (True/False to 1/0)
df['anaemia'] = df['anaemia'].astype(int)
df['diabetes'] = df['diabetes'].astype(int)
df['high_blood_pressure'] = df['high_blood_pressure'].astype(int)
df['smoking'] = df['smoking'].astype(int)
df['sex'] = df['sex'].astype(int) #0 for female 1 for male

# Define features (X) and target (y)
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# 3. Handle Class Imbalance (SMOTE) (same as before)
print(y.value_counts())
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 4. Split Data into Training and Testing Sets (same as before)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# 5. Feature Scaling (same as before)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Build and Train the Neural Network (MLPClassifier) (same as before)
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=42, max_iter=500)
mlp.fit(X_train, y_train)

# 7. Make Predictions (same as before)
y_pred = mlp.predict(X_test)

# 8. Evaluate the Model (same as before)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 9. (Optional) Hyperparameter Tuning (GridSearchCV, RandomizedSearchCV) (same as before)
# ... (uncomment and adapt as needed)