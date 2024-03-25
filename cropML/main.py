import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load the dataset
dataset = pd.read_csv('output.csv')

# Train Random Forest model
X = dataset[['N', 'P', 'K', 'temperature', 'humidity', 'soil_moisture']]
y = dataset['label']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation

# Calculate mean accuracy
mean_accuracy = scores.mean()

print("Accuracy:", mean_accuracy * 100)

# Function to find the closest row with the same crop and return its pH and rainfall values
def find_closest_crop_row(crop_label, N, P, K):
    crop_rows = dataset[dataset['label'] == crop_label]
    min_distance = float('inf')
    closest_row = None
    for index, row in crop_rows.iterrows():
        distance = abs(row['N'] - N) + abs(row['P'] - P) + abs(row['K'] - K)
        if distance < min_distance:
            min_distance = distance
            closest_row = row
    return closest_row['ph'], closest_row['rainfall']

def predict_crop(N, P, K, temperature, humidity, soil_moisture):
    # Provide feature names explicitly when calling predict()
    crop_label = model.predict(pd.DataFrame([[N, P, K, temperature, humidity, soil_moisture]], columns=X.columns))[0]
    
    # Find the closest row with the same crop and return its pH and rainfall values
    optimal_ph, optimal_rainfall = find_closest_crop_row(crop_label, N, P, K)
    
    return crop_label, optimal_ph, optimal_rainfall


# Example usage:
N = 80
P = 50
K = 40
temperature = 25.89
humidity = 70.134
soil_moisture = 60.56

predicted_crop, optimal_ph, optimal_rainfall = predict_crop(N, P, K, temperature, humidity, soil_moisture)
print("Predicted Crop:", predicted_crop)
print("Optimal pH:", optimal_ph)
print("Optimal Rainfall (mm):", optimal_rainfall)
