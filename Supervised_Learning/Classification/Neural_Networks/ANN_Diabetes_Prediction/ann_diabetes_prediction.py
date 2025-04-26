import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Read CSV
df = pd.read_csv("./Datasets/diabetes.csv")

# Divide into input and output data
X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.2)