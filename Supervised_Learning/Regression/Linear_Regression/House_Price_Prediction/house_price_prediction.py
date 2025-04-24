import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("./Datasets/house_prices.csv").dropna()

# ---------------------------------
#        Data Visualisation 
# ---------------------------------

fig, axs = plt.subplots(2, 2, figsize=(7, 6))

# Plot 1: Square footage
axs[0, 0].scatter(df["Square_Footage"], df["House_Price"], s=3, alpha=.5, color='teal')
axs[0, 0].set_title("Square Feet vs Price")
axs[0, 0].set_xlabel("Square Feet")
axs[0, 0].set_ylabel("Price (USD)")

# Plot 2: Year built
axs[0, 1].scatter(df["Year_Built"], df["House_Price"], s=3, alpha=.5, color='darkorange')
axs[0, 1].set_title("Year Built vs Price")
axs[0, 1].set_xlabel("Year Built")
axs[0, 1].set_ylabel("Price (USD)")

# Plot 3: Lot size
axs[1, 0].scatter(df["Lot_Size"], df["House_Price"], s=3, alpha=.5, color='indigo')
axs[1, 0].set_title("Lot Size vs Price")
axs[1, 0].set_xlabel("Lot Size (in acres)")
axs[1, 0].set_ylabel("Price (USD)")

# Plot 4: Bedrooms
axs[1, 1].axis("off")

# Adjust spacing
plt.tight_layout()
plt.show()

# ---------------------------------
#             Model 
# ---------------------------------

# Instantiate model
model = LinearRegression()

# Divide data into input (X) and output (y) data
X = df.drop(["House_Price"], axis=1)
y = df["House_Price"]

# Divide data into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit model with X and y
model.fit(x_train, y_train)

# Check accuracy scores
print(f"Train accuracy: {model.score(x_train, y_train)}")
print(f"Test accuracy: {model.score(x_test, y_test)}")

# Predict x test data
y_pred = model.predict(x_test)

# Plot real prices vs predicted prices
plt.scatter(y_test, y_pred, s=5, alpha=0.75)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Real price")
plt.ylabel("Predicted price")
plt.title("Real price vs Predicted price")
plt.grid(True)
plt.show()