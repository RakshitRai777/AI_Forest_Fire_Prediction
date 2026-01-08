import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def load_data(folder_path):
    try:
        # Parse XML files in Annotations folder
        annotations_list = []
        for file in os.listdir(folder_path + '/Annotations'):
            if file.endswith('.xml'):
                tree = ET.parse(os.path.join(folder_path, 'Annotations', file))
                root = tree.getroot()
                annotation_data = {child.tag: child.text for child in root}
                annotations_list.append(annotation_data)
        annotations = pd.DataFrame(annotations_list)
        
        # Load CSV files in Datacluster folder
        datacluster_file = [f for f in os.listdir(folder_path + '/Datacluster Fire and Smoke Sample') if f.endswith('.csv')][0]
        datacluster = pd.read_csv(os.path.join(folder_path, 'Datacluster Fire and Smoke Sample', datacluster_file))

        return annotations, datacluster
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def identify_columns(data):
    try:
        common_column = data.columns[0]  # Assume first column is the common column
        target_column = data.columns[-1]  # Assume last column is the target column
        feature_columns = [col for col in data.columns if col not in [common_column, target_column]]

        return common_column, target_column, feature_columns
    except Exception as e:
        print(f"Error identifying columns: {e}")
        raise

# Replace 'your_folder_path' with the actual path to your DataSet folder
folder_path = "C:\\Users\\raksh\\Desktop\\Coding Projects\\Forest-Fire-Prediction\\DataSet"
annotations, datacluster = load_data(folder_path)

common_column, target_column, _ = identify_columns(annotations)
data = pd.merge(annotations, datacluster, on=common_column)

# Identify columns
_, target_column, feature_columns = identify_columns(data)

# Split the Data
X = data[feature_columns]
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print('Classification Report:')
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.show()

# Save the Model
joblib.dump(model, 'forest_fire_model.pkl')

# Load the Model and Make Predictions on New Data
model = joblib.load('forest_fire_model.pkl')

# Example of making a prediction with new data
sample_data = data[feature_columns].iloc[0].values.reshape(1, -1)
new_data = scaler.transform(sample_data)
prediction = model.predict(new_data)
print(f'Prediction: {prediction}')
