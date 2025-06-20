import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier

# Load the dataset
df = pd.read_csv("heart (1).csv")  # Make sure you replace with the correct path
print(df.head())

#Preprocess Data
# Features and target variable
X = df.drop("target", axis=1)  # all columns except 'target'
y = df["target"]  # target column

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for neural networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the neural network
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model's performance
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

#Predict from the input
import numpy as np
def predict_heart_disease():
    # Get user input
    print("Enter the following details for heart disease prediction:")

    # Input: age, sex, cholesterol, blood pressure, etc.
    age = float(input("Age: "))
    sex = float(input("Sex (1 = Male, 0 = Female): "))
    cp = float(input("Chest pain type (1-4): "))
    trestbps = float(input("Resting blood pressure: "))
    chol = float(input("Cholesterol level: "))
    fbs = float(input("Fasting blood sugar > 120 mg/dl (1 = True, 0 = False): "))
    restecg = float(input("Resting electrocardiographic results (0-2): "))
    thalach = float(input("Maximum heart rate achieved: "))
    exang = float(input("Exercise induced angina (1 = Yes, 0 = No): "))
    oldpeak = float(input("Depression induced by exercise: "))
    slope = float(input("Slope of the peak exercise ST segment (1-3): "))
    ca = float(input("Number of major vessels colored by fluoroscopy (0-3): "))
    thal = float(input("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversable Defect): "))

    # Prepare input data
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # Scale the input using the same scaler as before
    user_input_scaled = scaler.transform(user_input)

    # Make prediction using the trained model
    prediction = model.predict(user_input_scaled)

    # Step 3: Output the prediction
    if prediction == 1:
        print("Prediction: The person is likely to have heart disease.")
    else:
        print("Prediction: The person is unlikely to have heart disease.")

# Run the predictive system
predict_heart_disease()



#Data Visualisation Heatmap

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()