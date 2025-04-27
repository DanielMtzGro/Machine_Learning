import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Cargar el modelo guardado
model = tf.keras.models.load_model('./Models/diabetes_prediction_79_f1.keras')

# Función para pedir datos al usuario
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

    # Calcular nuevas características basadas en los datos ingresados
    bmi_age = bmi * age
    glucose_insulin_ratio = glucose / (insulin + 1e-6)  # Evitar división por cero
    glucose_age = glucose * age
    insulin_age = insulin * age
    pregnancies_age = pregnancies * age
    pregnancies_bmi = pregnancies * bmi
    blood_pressure_glucose_ratio = blood_pressure / (glucose + 1e-6)  # Evitar división por cero
    blood_pressure_insulin_ratio = blood_pressure / (insulin + 1e-6)  # Evitar división por cero
    age_pedigree = age * diabetes_pedigree_function
    blood_pressure_skin_thickness = blood_pressure * skin_thickness

    # Devolver todos los valores
    return np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi,
                      diabetes_pedigree_function, age, bmi_age, glucose_insulin_ratio, glucose_age,
                      insulin_age, pregnancies_age, pregnancies_bmi, blood_pressure_glucose_ratio,
                      blood_pressure_insulin_ratio, age_pedigree, blood_pressure_skin_thickness]])

# Obtener datos del usuario
user_data = get_user_input()

# Cargar el StandardScaler utilizado durante el entrenamiento (debe estar guardado o preajustado previamente)
scaler = StandardScaler()

# Asumimos que el modelo ha sido entrenado con un escalado de los datos
# Si tienes el scaler ajustado previamente, cargalo aquí y usa solo transform
scaler.fit(user_data)  # Esto es solo un ejemplo; en un caso real deberías tener el scaler ajustado previamente.
user_data_scaled = scaler.transform(user_data)

# Realizar la predicción con el modelo cargado
prediction = model.predict(user_data_scaled)

# Mostrar la predicción
if prediction > 0.5:
    print("The model predicts: Diabetes (1)")
else:
    print("The model predicts: No Diabetes (0)")
