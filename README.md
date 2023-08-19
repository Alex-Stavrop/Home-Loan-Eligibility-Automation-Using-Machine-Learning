# Home Loan Eligibility Automation Using Machine Learning

## Introduction

This project aims to explore the automation of home loan eligibility processes for a financial institution using machine learning algorithms. The goal is to predict in real-time whether an applicant qualifies for a loan based on the details provided during the online application process. Three machine learning algorithms are evaluated: 
- K-Nearest Neighbor (KNN)
- Random Forest
- Extreme Gradient Boosting (XGBoost)

The objective is to determine which algorithm can best predict customer loan eligibility.

## Data Description & Preparations

The dataset used for this project contains 614 customer observations with 12 variables detailing their characteristics (as per Devzohaib, 2022). These variables include gender, marital status, education level, employment and income status, loan amount and terms, credit history, property area, presence of a co-applicant and their income, and loan eligibility status.

Post data exploration, a few key steps were taken in data preparation:
- Oversampling to address class imbalance
- Omission of observations with missing values
- Normalization of numeric variables
- Factorization of categorical variables
- Splitting the dataset into training (70%) and testing (30%) sets

## Methods

### K-Nearest Neighbors Algorithm (KNN)
KNN is a non-parametric, supervised machine learning algorithm that classifies data points based on the proximity to other data points. The decision boundaries in KNN are established by measuring distances, typically using the Euclidean distance.

### Random Forest
Random forest is an ensemble method that constructs multiple decision trees during training and outputs the mode of the classes for classification.

### Extreme Gradient Boosting (XGBoost)
XGBoost is an ensemble algorithm that uses the boosting technique. It sequentially builds decision trees, where each tree corrects the errors of the previous one.

### Cross Validation
K-fold cross validation was used to evaluate the performance of machine learning algorithms and to tune hyperparameters.

## Analysis & Results

All three models demonstrated commendable classification capabilities for loan eligibility:

1. **KNN Model**: Achieved an in-sample accuracy of 85% and an out-of-sample accuracy of 81.73%.
2. **Random Forest Model**: Achieved an accuracy of 81.05% on the training data and 79.90% on the test data.
3. **XGBoost Model**: Achieved an accuracy of 87.68% on the training data and 83.56% on the test data.

After evaluating the models, XGBoost showcased the strongest performance and is recommended for automating the loan eligibility process.

## Feature Importance

The XGBoost model indicated that an applicant's income level, in conjunction with the loan amount they are applying for, plays a crucial role in loan approval.

## Conclusions

While all models performed well, the XGBoost algorithm outperformed the others with a better overall error rate. The most significant feature impacting loan approval is the applicant's income level combined with the requested loan amount.

## Bibliography

- Devzohaib. 2022. [Eligibility Prediction for Loan](https://www.kaggle.com/datasets/devzohaib/eligibility-prediction-for-loan).
- IBM. 2022a. [What Is Gradient Descent?](https://www.ibm.com/topics/gradient-descent)
- IBM. 2022b. [What Is Random Forest?](https://www.ibm.com/topics/random-forest)
- IBM. 2022c. [What Is the k-Nearest Neighbors Algorithm?](https://www.ibm.com/topics/knn)
- NVIDIA. 2022. [What Is XGBoost?](https://www.nvidia.com/en-us/glossary/data-science/xgboost/)

### For more detailed information of this project please refer to the Home Loan Eligibility Automation Using Machine Learning report. 


