# To run this code, ensure you have the required libraries installed.
# You can install them using pip:
# pip install pandas numpy matplotlib seaborn tensorflow scikit-learn
# This code is designed to read multiple CSV files, process the data, and build a recommendation system using machine learning.
# Import necessary libraries

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import tensorflow as tf

# Load and merge CSVs
'''csv_files = glob.glob("./data/*.csv")
dfs = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(dfs, ignore_index=True)'''

csv_files = glob.glob("./data/*.csv")
if not csv_files:
    raise FileNotFoundError("No CSV files found in the './data/' directory.")
dfs = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(dfs, ignore_index=True)
print("\nMerged dataframe shape:", merged_df.shape)
# Check for missing values
print("\nMissing values in each column:\n", merged_df.isnull().sum())
# Check data types
print("\nData types:\n", merged_df.dtypes)
# Check basic statistics
print("\nBasic statistics:\n", merged_df.describe())
# Check for unique values in categorical columns
print("\nUnique values in categorical columns:\n", merged_df.select_dtypes(include=['object']).nunique())
# Check for duplicates
print("\nNumber of duplicate rows:", merged_df.duplicated().sum())
# Check for outliers
print("\nOutliers in numerical columns:\n", merged_df.select_dtypes(include=[np.number]).describe())
# Check for date range
min_datetime = merged_df['timestamp'].min()
max_datetime = merged_df['timestamp'].max()
print("\nMinimum Timestamp (oldest Record):", min_datetime)
print("Maximum Timestamp (Latest Record):", max_datetime)
# Visualize activity over time
daily_activity = merged_df['timestamp'].apply(lambda x: pd.to_datetime(x, unit='ms')).dt.date.value_counts().sort_index()
plt.figure(figsize=(14, 6))
daily_activity.plot(kind='line', marker='o')
plt.title("Activity Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Actions")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('visualization/activity_over_time.png')
# Check for missing values again
print("\nMissing values after preprocessing:\n", merged_df.isnull().sum())
# Check for unique values in categorical columns again
print("\nUnique values in categorical columns after preprocessing:\n", merged_df.select_dtypes(include=['object']).nunique())
# Check for duplicates again
print("\nNumber of duplicate rows after preprocessing:", merged_df.duplicated().sum())


# Preprocessing
merged_df['user_answer'] = merged_df['user_answer'].fillna('No Answer')
merged_df['datetime'] = pd.to_datetime(merged_df['timestamp'], unit='ms')
merged_df['hour'] = merged_df['datetime'].dt.hour
merged_df['dayofweek'] = merged_df['datetime'].dt.dayofweek
merged_df['is_answered'] = merged_df['user_answer'].apply(lambda x: 0 if x == 'No Answer' else 1)
merged_df['action_type_encoded'] = merged_df['action_type'].astype('category').cat.codes
merged_df['platform_encoded'] = merged_df['platform'].astype('category').cat.codes
merged_df['source_encoded'] = merged_df['source'].astype('category').cat.codes

features = ['hour', 'dayofweek', 'action_type_encoded', 'platform_encoded', 'source_encoded']
target = 'is_answered'
X = merged_df[features]
y = merged_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_f1 = f1_score(y_test, y_pred_knn)
joblib.dump(knn_model, 'knn_model.pkl')

# Deep Learning Model
deep_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
deep_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
deep_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)
deeplearning_accuracy = deep_model.evaluate(X_test, y_test)[1]
y_pred_dl = deep_model.predict(X_test).round()
deeplearning_f1 = f1_score(y_test, y_pred_dl)
deep_model.save('deeplearning_model.keras')

# Save results table
comparison_df = pd.DataFrame({
    "Model": ["KNN", "Deep Learning"],
    "Accuracy": [knn_accuracy, deeplearning_accuracy],
    "F1 Score": [knn_f1, deeplearning_f1]
})
comparison_df.to_csv("model_comparison.csv", index=False)

# Visualizations
plt.figure(figsize=(10, 5))
sns.countplot(data=merged_df, x='hour')
plt.title('Activity by Hour')
plt.savefig('visualization/hourly_activity.png')

plt.figure(figsize=(10, 5))
sns.countplot(data=merged_df, x='dayofweek')
plt.title('Activity by Day of the Week')
plt.savefig('visualization/weekly_activity.png')





