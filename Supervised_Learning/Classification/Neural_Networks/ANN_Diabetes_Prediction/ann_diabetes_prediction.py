import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# Read CSV
df = pd.read_csv("./Datasets/diabetes.csv")

# Divide into input and output data
X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]

# Adding new features to X
X["BMI_Age"] = X["Age"] * X["BMI"]
X["Glucose_Insulin_Ratio"] = X["Glucose"] / (X["Insulin"] + 1e-6)
X['Glucose_Age'] = X['Glucose'] * X['Age']
X['Insulin_Age'] = X['Insulin'] * X['Age']
X['Pregnancies_Age'] = X['Pregnancies'] * X['Age']
X['Pregnancies_BMI'] = X['Pregnancies'] * X['BMI']
X['BloodPressure_Glucose_Ratio'] = X['BloodPressure'] / (X["Glucose"] + 1e-6)
X['BloodPressure_Insulin_Ratio'] = X['BloodPressure'] / (X["Insulin"] + 1e-6)
X['Age_Pedigree'] = X['Age'] * X['DiabetesPedigreeFunction']
X['BloodPressure_SkinThickness'] = X['BloodPressure'] * X['SkinThickness']

# Divide into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Instantiate the Standard Scaler
scaler = StandardScaler()

# Scale x_train
x_train_scaled = scaler.fit_transform(x_train)

# Scale x_test
x_test_scaled = scaler.transform(x_test)

# Build the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(x_train.shape[1],)),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'recall'])

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Fit the model with training data
history = model.fit(
    x_train_scaled, y_train, 
    epochs=500, 
    batch_size=32, 
    validation_data=(x_test_scaled, y_test), 
    callbacks=[early_stopping]
)

# Confusion matrix
y_pred = model.predict(x_test_scaled)

# Convert probabilities to class predictions
y_pred_classes = (y_pred > 0.5).astype("int32")

# Print classification report
print(classification_report(y_test, y_pred_classes))

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Plot train loss vs test loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues, colorbar=False)
plt.title("Confusion Matrix")
plt.show()

# Save model in 'Models' folder
model.save('./Models/diabetes_prediction.keras')