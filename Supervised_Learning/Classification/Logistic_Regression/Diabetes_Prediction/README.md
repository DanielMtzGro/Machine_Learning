# Diabetes Prediction with Logistic Regression

This project uses **Logistic Regression** to predict the likelihood of diabetes based on patient health data. The dataset used is the well-known **Pima Indians Diabetes Dataset**, where all patients are **females of at least 21 years old** and **of Pima Indian heritage**.

---

## Dataset Overview

The dataset includes the following original medical predictor variables:

- `Pregnancies`
- `Glucose`
- `Blood Pressure`
- `Skin Thickness`
- `Insulin`
- `BMI`
- `Diabetes Pedigree Function`
- `Age`

The target variable is:

- `Outcome` – whether the patient has diabetes (1) or not (0).

> ℹ️ All patients in the dataset are **females aged 21 or older** of **Pima Indian heritage**.

---

## 🛠️ Preprocessing Steps

1. **Feature Engineering**: Several new features were created to explore potential interactions:
   - `BMI_Age`
   - `Glucose_Insulin_Ratio`
   - `Glucose_Age`
   - `Insulin_Age`
   - `Pregnancies_Age`
   - `Pregnancies_BMI`
   - `BloodPressure_Glucose_Ratio`
   - `BloodPressure_Insulin_Ratio`
   - `Age_Pedigree`
   - `BloodPressure_SkinThickness`

2. **Train/Test Split**: The data is split into 80% training and 20% testing sets, stratified by the outcome variable.

3. **Feature Scaling**: All features are standardized using `StandardScaler`.

---

## 🤖 Model

- **Model**: `LogisticRegression` from `scikit-learn`
- **Cross-validation**: 5-fold cross-validation is used to evaluate the model’s performance using the **F1 score**, which balances precision and recall.
- **Final Training**: The model is trained on the full scaled training set and evaluated on the test set to detect overfitting.

---

## 📊 Results

- The F1 score from cross-validation was approximately 63%.
- The model's accuracy on the test set was around 78%.
- These metrics indicate that the model performs reasonably well, although further improvements could be made with more complex models or additional features.