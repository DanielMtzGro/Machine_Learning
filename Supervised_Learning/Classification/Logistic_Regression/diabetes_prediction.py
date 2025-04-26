import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Read CSV
df = pd.read_csv("./Datasets/diabetes.csv")

# ------------------------------
# ----- Preprocessing ---------
# ------------------------------

# Divide data into input and output data
X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]

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
model = LogisticRegression(max_iter=10000)

# Cross validation
scores = cross_val_score(model, x_train_scaled, y_train, cv=5, scoring="f1")
print(f"Model average F1 score: {scores.mean()}")

# Fit the model with the train data
model.fit(x_train_scaled, y_train)

# Compare accuracies for overfitting
print(f"Model train accuracy: {model.score(x_train_scaled, y_train)}")
print(f"Model test accuracy: {model.score(x_test_scaled, y_test)}")

