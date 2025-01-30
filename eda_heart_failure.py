import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data (replace with your actual URL or file path)
github_url = "https://raw.githubusercontent.com/saumyasam/ML_For_Heart_Failure_Detect/main/heart_failure_clinical_records_dataset.csv"
try:
    df = pd.read_csv(github_url)
except Exception as e:
    print(f"Error loading data from GitHub: {e}")
    exit()

# Basic Information
print(df.info())  # Data types, missing values
print(df.describe()) # Summary statistics for numerical features

# Check for missing values (if any)
print("\nMissing Values:")
print(df.isnull().sum())

# Class Distribution
print("\nClass Distribution:")
print(df['DEATH_EVENT'].value_counts())
sns.countplot(x='DEATH_EVENT', data=df)
plt.title('Distribution of Death Event')
plt.show()

# Distribution of Numerical Features
numerical_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[feature], kde=True)  # Use histplot for distributions with KDE
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Box plots for Numerical Features (Outlier Detection)
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[feature])
    plt.title(f'Box Plot of {feature}')
    plt.xlabel(feature)
    plt.show()

# Relationships between Numerical Features and the Target Variable
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='DEATH_EVENT', y=feature, data=df)
    plt.title(f'{feature} vs. Death Event')
    plt.xlabel('Death Event (0 = No, 1 = Yes)')
    plt.ylabel(feature)