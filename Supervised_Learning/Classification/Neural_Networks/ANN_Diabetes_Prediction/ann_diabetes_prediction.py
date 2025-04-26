import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

# Read CSV
df = pd.read_csv("./Datasets/diabetes.csv")

# Divide into input and output data
X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]

# Divide into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'recall'])

# Fit the model with training data
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))

# Confusion matrix
y_pred = model.predict(x_test)

# Convert probabilities to class predictions
y_pred_classes = (y_pred > 0.5).astype("int32")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues, colorbar=False)
plt.title("Confusion Matrix")
plt.show()
