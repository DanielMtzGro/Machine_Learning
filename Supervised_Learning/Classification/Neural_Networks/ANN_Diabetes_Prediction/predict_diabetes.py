import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the saved model
model = tf.keras.models.load_model('./Models/diabetes_prediction_79_f1.keras')

# Function to get user input
def get_user_input():
    print("Enter the following values for prediction:")

    pregnancies = int(input("Pregnancies: "))
    glucose = int(input("Glucose: "))
    blood_pressure = int(input("Blood Pressure: "))
    skin_thickness = int(input("Skin Thickness: "))
    insulin = int(input("Insulin: "))
    bmi = float(input("BMI: "))
    diabetes_pedigree_function = float(input("Diabetes Pedigree Function: "))
    age = int(input("Age: "))

    # Calculate new features based on the input data
    bmi_age = bmi * age
    glucose_insulin_ratio = glucose / (insulin + 1e-6)  # Avoid division by zero
    glucose_age = glucose * age
    insulin_age = insulin * age
    pregnancies_age = pregnancies * age
    pregnancies_bmi = pregnancies * bmi
    blood_pressure_glucose_ratio = blood_pressure / (glucose + 1e-6)  # Avoid division by zero
    blood_pressure_insulin_ratio = blood_pressure / (insulin + 1e-6)  # Avoid division by zero
    age_pedigree = age * diabetes_pedigree_function
    blood_pressure_skin_thickness = blood_pressure * skin_thickness

    # Return all the values as a numpy array
    return np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi,
                      diabetes_pedigree_function, age, bmi_age, glucose_insulin_ratio, glucose_age,
                      insulin_age, pregnancies_age, pregnancies_bmi, blood_pressure_glucose_ratio,
                      blood_pressure_insulin_ratio, age_pedigree, blood_pressure_skin_thickness]])

# Get the data from the user
user_data = get_user_input()

# Load the StandardScaler used during training (it should be saved or pre-adjusted previously)
scaler = StandardScaler()

# We assume that the model was trained with scaled data
# If you have the scaler adjusted previously, load it here and use only transform
scaler.fit(user_data)  # This is just an example; in a real case, you should have the scaler fitted beforehand.
user_data_scaled = scaler.transform(user_data)

# Make a prediction using the loaded model
prediction = model.predict(user_data_scaled)

# Display the prediction
if prediction > 0.5:
    print("The model predicts: Diabetes")
else:
    print("The model predicts: No Diabetes")