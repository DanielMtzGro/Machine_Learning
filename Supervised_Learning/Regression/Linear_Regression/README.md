# House Price Prediction using Linear Regression

This project predicts house prices using a linear regression model applied to a dataset of house characteristics. The goal is to predict the price of a house based on various features such as square footage, year built, and lot size.

## Objective

The main objective of this project is to use a linear regression model to predict house prices based on several features: square footage, year built, lot size, and others. This allows us to explore relationships between house attributes and their prices, as well as evaluate the performance of the model using training and testing data.

## Dataset

The dataset used in this project contains information about house prices and various features of the houses. Some of the key features include:
- Square Footage (ft²)
- Year Built
- Lot Size (in acres)
- Number of Bedrooms (not visualized)
  
Missing values were removed before training the model.

## Methodology

1. **Data Loading**: The dataset was loaded from a CSV file, and missing values were removed to ensure a clean dataset for model training.
2. **Data Visualization**: Four scatter plots were created to explore relationships between the target variable (house price) and features such as square footage, year built, and lot size.
3. **Model Training**: A linear regression model was instantiated and trained using 80% of the data. The remaining 20% was used for testing the model’s performance.
4. **Model Evaluation**: The model’s performance was evaluated using the R² score for both the training and testing datasets. A comparison between real and predicted prices was visualized with a scatter plot.

## Results

The linear regression model showed the following results:
- **Training Accuracy**: The R² score on the training dataset.
- **Testing Accuracy**: The R² score on the testing dataset.
  
The comparison of real prices vs predicted prices was visualized using a scatter plot, where the predicted prices were plotted against the real values. A red dashed line represents the ideal case where the predicted price matches the real price.

## Conclusion

This project demonstrates how linear regression can be used to predict house prices based on various features. The model provides a good starting point for understanding how different attributes impact house pricing. However, there is room for improvement by adding more features or using more complex algorithms.

### Possible Improvements:
- Include more features such as the number of bedrooms, bathrooms, or location.
- Experiment with other regression models, like Ridge or Lasso, to improve performance.
- Perform hyperparameter tuning to improve model accuracy.
- Use cross-validation to get more robust performance metrics.
- Implement feature scaling to normalize features like square footage and lot size.

## Requirements

To run this project, install the following Python packages:

- pandas
- matplotlib
- scikit-learn

## License

This project is open source and free to use under the MIT License.