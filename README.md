<div align="center">

# Credit Card Fraud Detection

</div>

---

## Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques.  
Due to the highly imbalanced nature of fraud data, traditional accuracy-based evaluation is misleading.  
The primary objective of this project is to study how **class balancing techniques** affect fraud detection performance.

The project compares models trained using **Random Undersampling** and **SMOTE**, and evaluates them using metrics suited for imbalanced datasets.

---

## Dataset

**Credit Card Fraud Detection Dataset (Kaggle)**  
https://www.kaggle.com/mlg-ulb/creditcardfraud

- Total transactions: 284,807  
- Fraudulent transactions: 492 (0.172%)  
- Target variable: `Class`  
  - 1 → Fraud  
  - 0 → Legitimate  

All input features are numerical.  
Features `V1` to `V28` are generated using **PCA transformation** to preserve confidentiality.  
Only `Amount` and `Time` were provided in original form.

---

## Data Preprocessing

The following preprocessing steps were applied:

- Dropped the `Time` feature
- Standardized the `Amount` feature using `StandardScaler`
- Separated features and target variable
- Split data into training and testing sets (80:20)

---

## Handling Class Imbalance

Since fraudulent transactions are extremely rare, two balancing techniques were applied and compared.

### Random Undersampling
- Randomly removes samples from the majority class
- Produces a balanced dataset with reduced size
- Can lead to information loss and underfitting

### SMOTE (Synthetic Minority Oversampling Technique)
- Generates synthetic fraud samples using nearest neighbors
- Improves minority class representation
- May introduce noisy or unrealistic samples if overused

Both techniques were applied **only on training data** to avoid data leakage.

---

## Machine Learning Models

The following models were trained and evaluated on both undersampled and SMOTE-balanced datasets:

- Logistic Regression  
- XGBoost Classifier  
- Artificial Neural Network (TensorFlow / Keras)

---

## Model Evaluation

Models were evaluated using metrics appropriate for imbalanced classification:

- Precision  
- Recall  
- F1-score  
- ROC-AUC  

Confusion matrices, ROC curves, and Precision–Recall curves were analyzed to compare performance.

---

## Results Summary

<hr>

<p>
The following tables summarize the performance of different machine learning models
trained using Random Undersampling and SMOTE. Evaluation was performed using
Accuracy, Precision, Recall, F1-score, and ROC-AUC metrics.
</p>

<h3>Logistic Regression</h3>

<table>
  <tr>
    <th>Sampling Method</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>AUC</th>
  </tr>
  <tr>
    <td>Undersampling</td>
    <td>0.94</td>
    <td>0.96</td>
    <td>0.89</td>
    <td>0.92</td>
    <td>0.97</td>
  </tr>
  <tr>
    <td>SMOTE</td>
    <td>0.95</td>
    <td>0.97</td>
    <td>0.89</td>
    <td>0.93</td>
    <td>0.988</td>
  </tr>
</table>

<br>

<h3>XGBoost Classifier</h3>

<table>
  <tr>
    <th>Sampling Method</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>AUC</th>
  </tr>
  <tr>
    <td>Undersampling</td>
    <td>0.95</td>
    <td>0.97</td>
    <td>0.89</td>
    <td>0.93</td>
    <td>0.975</td>
  </tr>
  <tr>
    <td>SMOTE</td>
    <td>0.99</td>
    <td>0.99</td>
    <td>1.00</td>
    <td>0.99</td>
    <td>0.99</td>
  </tr>
</table>

<br>

<h3>Artificial Neural Network (ANN)</h3>

<table>
  <tr>
    <th>Sampling Method</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>AUC</th>
  </tr>
  <tr>
    <td>Undersampling</td>
    <td>0.95</td>
    <td>0.98</td>
    <td>0.88</td>
    <td>0.93</td>
    <td>0.97</td>
  </tr>
  <tr>
    <td>SMOTE</td>
    <td>0.998</td>
    <td>0.995</td>
    <td>0.998</td>
    <td>0.997</td>
    <td>0.999</td>
  </tr>
</table>

<br>

<hr>


Key observations from the experiments:

- SMOTE consistently improved recall and ROC-AUC across all models
- Random undersampling reduced training data size, slightly limiting recall.
- Logistic Regression served as a strong baseline
- XGBoost showed stable and high performance
- The Neural Network trained on SMOTE-balanced data achieved the highest recall and AUC, indicating improved fraud detection capability

---

## Technologies Used

- Python  
- NumPy, Pandas  
- Scikit-learn  
- Imbalanced-learn  
- XGBoost  
- TensorFlow / Keras  
- Matplotlib, Seaborn  

---

## Conclusion

This project highlights the importance of handling class imbalance in fraud detection problems.  
While Random Undersampling is simple and effective, SMOTE generally provided better performance by allowing models to learn richer fraud patterns.

Evaluation using recall and AUC proved more meaningful than accuracy for this task.

---

## Author

**Mohammed Maaz Ali**  
B.Tech Computer Science Engineering  
IIIT Kottayam

