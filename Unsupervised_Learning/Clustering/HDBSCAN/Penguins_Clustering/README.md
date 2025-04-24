# Penguin Clustering using HDBSCAN

This project applies the HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) clustering algorithm to a dataset of penguins. The dataset includes measurements of various physical features of penguins from different species. The goal is to identify natural groupings within the data using unsupervised learning techniques.

## Objective

The main objective of this project is to use the HDBSCAN algorithm to cluster penguins based on their physical characteristics: flipper length, culmen length, and culmen depth. This allows us to explore patterns in the data and discover potential clusters that may correspond to different species or subgroups.

## Dataset

The dataset used in this project is the popular Palmer Penguins dataset. It includes features such as:
- Flipper Length (mm)
- Culmen Length (mm)
- Culmen Depth (mm)

Missing values were removed before performing the clustering.

## Methodology

1. **Data Loading**: The dataset was loaded from a CSV file and relevant features were selected.
2. **Preprocessing**: Rows with missing values were dropped to ensure clean input for the clustering algorithm.
3. **Clustering**: The HDBSCAN algorithm was applied with `min_cluster_size=10`. The model assigned cluster labels to each data point.
4. **Visualization**: A 3D scatter plot was created to visualize the clusters using Matplotlib. Optionally, an interactive plot can be created using Plotly.

## Results

The HDBSCAN algorithm successfully identified several clusters in the data, as well as a number of outliers (labeled as noise). The 3D visualization helps to interpret the structure of the data and the separation between the groups.

Two main clusters were observed:
- **Cluster 1**: Penguins with long flippers, average-length culmens, and shallow culmen depth.
- **Cluster 2**: Penguins with short flippers and deep culmens.

These clusters suggest meaningful biological differences in the morphology of the penguins that align with species or ecological adaptations.

## Conclusion

This project demonstrates how density-based clustering can be used to uncover natural groupings in biological data. HDBSCAN is particularly well-suited for datasets with noise or clusters of varying densities.

### Possible Improvements:
- Use a scaler like `StandardScaler` to normalize the feature space.
- Perform a parameter sweep for `min_cluster_size` and `min_samples` to improve clustering performance.
- Add other features from the dataset for deeper analysis.
- Compare HDBSCAN results with other clustering methods (e.g., K-Means, DBSCAN).

## Requirements

To run this project, install the following Python packages:

- pandas
- matplotlib
- hdbscan
- scikit-learn
- plotly


## License

This project is open source and free to use under the MIT License.