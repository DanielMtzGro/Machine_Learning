# Diabetes Prediction with Neural Networks

This project aims to predict the likelihood of diabetes in patients using machine learning models. The dataset used is the **Pima Indians Diabetes Database**, which includes data from female patients aged 21 or older of Pima Indian heritage. The primary goal is to classify whether a patient has diabetes or not, based on their health metrics.

## Dataset Description
The dataset contains the following features:
- **Pregnancies**: Number of pregnancies the patient has had.
- **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skinfold thickness (mm).
- **Insulin**: 2-hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg / height in m^2).
- **DiabetesPedigreeFunction**: A function which represents the genetic likelihood of diabetes.
- **Age**: Age of the patient (years).
- **Outcome**: Whether the patient has diabetes (1) or not (0).

## Project Overview

The project implements **Artificial Neural Networks (ANNs)** using TensorFlow/Keras to predict whether a patient has diabetes based on the input features. The dataset is pre-processed, scaled, and split into training and test sets for model evaluation.

### Key Steps:
1. **Data Preprocessing**: 
   - Created additional features based on existing ones (such as BMI * Age and Blood Pressure * Glucose Ratio).
   - Scaled the features to ensure they are on a comparable scale for neural networks.
2. **Modeling**:
   - **Neural Network** model using TensorFlow/Keras was built with multiple layers and dropout for regularization.
3. **Evaluation**:
   - Evaluated the models using **accuracy**, **precision**, **recall**, and **F1-score**.
   - The best-performing model achieved an **F1-score of 79%**.

## Model Architecture (Neural Network)

The neural network model used consists of the following layers:
- **Input Layer**: Accepts the input features (scaled).
- **Hidden Layers**: Multiple dense layers with **ReLU** activation functions and **Dropout** layers to reduce overfitting.
- **Output Layer**: A single neuron with **sigmoid** activation for binary classification (0 or 1).

### Early Stopping:
The model uses **early stopping** with the patience of 20 epochs to prevent overfitting and ensure the model doesn't train excessively without improvements.

## Evaluation Metrics

The model was evaluated using the following classification metrics:
- **Accuracy**: Percentage of correct predictions.
- **Precision**: Precision of the positive class (Diabetes).
- **Recall**: Sensitivity of detecting positive cases (Diabetes).
- **F1-Score**: Harmonic mean of precision and recall, with the best model achieving an F1-score of **79%**.

## Results

- The final neural network model achieved an **F1-score of 79%**.
- Confusion matrix and classification report were used to assess the performance.

### Confusion Matrix:
The confusion matrix was used to visualize the classification performance:
- **True Positives** (TP): Correctly predicted diabetes cases.
- **True Negatives** (TN): Correctly predicted non-diabetes cases.
- **False Positives** (FP): Incorrectly predicted diabetes when the patient did not have it.
- **False Negatives** (FN): Incorrectly predicted no diabetes when the patient actually had it.

### Model Performance:
- **F1-Score**: 79% (Best Model)
- The model demonstrates a good balance between precision and recall.