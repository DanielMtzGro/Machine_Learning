import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

#------------- PRE-PROCESSING---------------
# Read CSV
df = pd.read_csv("./Datasets/fraud_detection.csv")
print(df.head())

# Divide into input and output data
X = df.drop(["transaction_id", "label"], axis=1)
y = df["label"]

# One hot encoding
X = pd.get_dummies(X, columns=["merchant_type", "device_type"])

print(X.head())

# Divide into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Create SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_scaled, y_train)


#------------- TRAINING ---------------------

# Create the model
model = LogisticRegression(random_state=42, class_weight={0: 1, 1: 10})

# Fit the model with training data
model.fit(x_train_resampled, y_train_resampled)

# Compare accuracies for overfitting
print(f"Model train accuracy: {model.score(x_train_resampled, y_train_resampled)}")
print(f"Model test accuracy: {model.score(x_test_scaled, y_test)}")


#------------- RESULTS ---------------------
# Predecir probabilidades de la clase 1
y_pred_proba = model.predict_proba(x_test_scaled)[:, 1]  # Probabilidades de la clase 1

# Ajustar el umbral
threshold = 0.3  # Prueba con un umbral más bajo para capturar más fraudes
y_pred_adjusted = (y_pred_proba > threshold).astype(int)

# Mostrar el reporte de clasificación y la matriz de confusión
print(classification_report(y_test, y_pred_adjusted))

conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues, colorbar=False)
plt.title("Confusion Matrix con umbral ajustado")
plt.show()