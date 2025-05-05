import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

# Divide into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

