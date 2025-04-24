import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("customer_segmentation.csv")
print(df.head(3))

plt.scatter(df["Income"], df["Year_Birth"], s=5)
plt.show()
