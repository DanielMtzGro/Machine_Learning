import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

#------------- PRE-PROCESSING---------------
# Read CSV
df = pd.read_csv("./Datasets/fraud_detection.csv")
print(df.head())

# Divide into input and output data
X = df.drop(["transaction_id", "label"], axis=1)
y = df["label"]


# One hot encoding
X["travel"] = X["merchant_type"] == "travel"
X["groceries"] = X["merchant_type"] == "groceries"
X["electronics"] = X["merchant_type"] == "electronics"
X["clothing"] = X["merchant_type"] == "clothing"
X["others"] = X["merchant_type"] == "others"

X["tablet"] = X["device_type"] == "tablet"
X["desktop"] = X["device_type"] == "desktop"
X["mobile"] = X["device_type"] == "mobile"

X.drop(["merchant_type"], axis=1, inplace=True)
X.drop(["device_type"], axis=1, inplace=True)

print(X.head())

#------------- TRAINING ---------------------
# Divide into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Create the model
model = LogisticRegression(class_weight="balanced")

# Fit the model with training data
model.fit(x_train, y_train)

# Compare accuracies for overfitting
print(f"Model train accuracy: {model.score(x_train, y_train)}")
print(f"Model test accuracy: {model.score(x_test, y_test)}")


#------------- RESULTS ---------------------
# Confusion matrix
y_pred = model.predict(x_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues, colorbar=False)
plt.title("Confusion Matrix")
plt.show()