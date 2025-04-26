import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Read CSV
df = pd.read_csv("./Datasets/diabetes.csv")

# ------------------------------
# ----- Preprocessing ---------
# ------------------------------

# Divide data into input and output data
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

#print(X.columns)

# Divide the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Instantiate the Standard Scaler
scaler = StandardScaler()

# Scale x_train
x_train_scaled = scaler.fit_transform(x_train)

# Scale x_test
x_test_scaled = scaler.transform(x_test)

# ------------------------------
# --------- Model --------------
# ------------------------------

# Build the model
model = LogisticRegression(max_iter=10000, class_weight="balanced")

# Cross validation
scores = cross_val_score(model, x_train_scaled, y_train, cv=5, scoring="f1")
print(f"Model average F1 score: {scores.mean()}")

# Fit the model with the train data
model.fit(x_train_scaled, y_train)

# Compare accuracies for overfitting
print(f"Model train accuracy: {model.score(x_train_scaled, y_train)}")
print(f"Model test accuracy: {model.score(x_test_scaled, y_test)}")

# Confusion matrix
y_pred = model.predict(x_test_scaled)

conf_matrix = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues, colorbar=False)
plt.title("Confusion Matrix")
plt.show()