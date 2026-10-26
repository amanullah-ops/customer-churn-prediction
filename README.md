# Customer Churn Prediction

## Overview

This project explores customer churn prediction using the Telco Customer Churn dataset. The goal is to understand which customer characteristics are associated with churn and build a machine learning model that predicts whether a customer is likely to leave.

This is my first end-to-end machine learning project, focused on data analysis, preprocessing, model evaluation, and learning how to build a reproducible ML workflow.

## Dataset

**Source:** IBM Telco Customer Churn sample dataset.  
      Original dataset: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv

      The dataset represents a fictional telecommunications company and is used here for educational machine learning practice. Please refer to the original repository for source and licensing information.

The dataset contains 7,043 customer records with information about customer tenure, contract type, services, payment methods, and charges.

The target variable is `Churn`, which indicates whether a customer left the service.

The `customerID` column is excluded from model features because it is an identifier rather than a meaningful customer attribute.

## Project Workflow

1. **Data inspection:** Examined the dataset structure, data types, missing values, and duplicate records.
2. **Exploratory data analysis:** Investigated churn patterns across contract types, tenure, monthly charges, internet service, payment methods, and support-related features.
3. **Data preprocessing:** Converted `TotalCharges` to a numeric column, scaled numerical features, and one-hot encoded categorical features.
4. **Baseline model:** Built a Logistic Regression pipeline using scikit-learn.
5. **Model evaluation:** Evaluated predictions using precision, recall, F1-score, confusion matrices, ROC-AUC, and Average Precision.
6. **Experimentation:** Compared classification thresholds and tested class weighting to understand the trade-off between identifying churners and generating false positives.

## Tools and Libraries

- Python
- Pandas and NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook
- Git and GitHub

## Key Findings

- Customers on month-to-month contracts had substantially higher observed churn than customers on longer contracts.
- Customers who churned tended to have shorter tenure and higher average monthly charges.
- Churn rates also differed across internet service, payment method, and support-related features.
- Class weighting increased churn recall but reduced precision, illustrating the trade-off between finding more potential churners and generating additional false alarms.

These findings describe associations in the dataset; they do not establish that any particular feature causes churn.

## Model Evaluation

The initial Logistic Regression model achieved approximately:

- **Accuracy:** 80.27%
- **Precision:** 65.79%
- **Recall:** 53.48%
- **F1-score:** 59.00%
- **ROC-AUC:** 0.836

These are validation results from the baseline model, not guaranteed real-world performance.

The test partition was used during earlier exploratory work. It is therefore not treated as an untouched final test set, and the project does not claim an independent final test score.

## Limitations

- The project uses a single publicly available dataset.
- The treatment of 11 blank `TotalCharges` values as zero is a documented assumption that should be checked against the dataset documentation.
- The classification threshold has not been selected using a formal business cost analysis.
- The model has not been deployed or tested on live customer data.

## Running the Project

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies listed in `requirements.txt`.
4. Open the notebooks in Jupyter.
5. Run the notebooks in order, beginning with data inspection and exploratory analysis.

The notebooks load the dataset from `data/telco-customer-churn.csv` using relative paths.

## Future Improvements

- Evaluate the model using a genuinely untouched test set.
- Compare additional models if there is a clear reason to do so.
- Improve threshold selection using business costs.
- Explore serving predictions through an API in a future project.

## Acknowledgment

This project was created as part of my practical learning journey in machine learning and data science.
