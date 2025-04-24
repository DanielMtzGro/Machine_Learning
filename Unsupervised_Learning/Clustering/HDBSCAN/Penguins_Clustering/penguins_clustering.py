import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Read CSV
df = pd.read_csv("./Datasets/penguins.csv")
X = df[["flipper_length_mm", "culmen_length_mm", "culmen_depth_mm"]].dropna()

# Instantiate clusterer
clusterer = hdbscan.HDBSCAN(min_cluster_size=10)

# Fit with data
labels = clusterer.fit_predict(X)

# Plot the data in a 3D graph
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    X["culmen_length_mm"],
    X["flipper_length_mm"],
    X["culmen_depth_mm"],
    c=labels,
    cmap="viridis",
    s=10
)

# Plot configuration
ax.set_title("Penguin clustering using HDBSCAN")
ax.set_xlabel("Culmen length (mm)")
ax.set_ylabel("Flipper length (mm)")
ax.set_zlabel("Culmen depth (mm)")
plt.show()

