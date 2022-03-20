# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project trains and analyzes a Machine Learning model that identifies credit card customers that are most likely to churn. The project follows the PEP8 coding standard. This project includes the testing code that verifies that the churn library works correctly.

## Running Files
The file churn_script_logging_and_tests.py executes and tests all the functions included in the churn_library.py

In order to execute the test code run the following command:
```
python churn_script_logging_and_tests.py
```
The log file contains the test results and possible errors.
If all the tests succeed the log should look like the following:

```
root - INFO - Testing import_data: SUCCESS
root - INFO - Testing perform_eda:
SUCCESS: executed without exceptions
root - INFO - Testing perform_eda:
SUCCESS: churn_distribution.png correctly generated
root - INFO - Testing perform_eda:
SUCCESS: customer_age_distribution.png correctly generated
root - INFO - Testing perform_eda:
SUCCESS: marital_status_distribution.png correctly generated
root - INFO - Testing perform_eda:
SUCCESS: total_transaction_distribution.png correctly generated
root - INFO - Testing perform_eda:
SUCCESS: heatmap.png correctly generated
root - INFO - Testing encoder_helper:
SUCCESS: executed without exceptions
root - INFO - Testing encoder_helper:
SUCCESS: Gender_Churn correctly added
root - INFO - Testing encoder_helper:
SUCCESS: Education_Level_Churn correctly added
root - INFO - Testing encoder_helper:
SUCCESS: Marital_Status_Churn correctly added
root - INFO - Testing encoder_helper:
SUCCESS: Income_Category_Churn correctly added
root - INFO - Testing encoder_helper:
SUCCESS: Card_Category_Churn correctly added
root - INFO - Testing perform_feature_engineering:
SUCCESS: executed without exceptions
root - INFO - Testing perform_feature_engineering:
SUCCESS: output values passed all tests
root - INFO - Testing train_models:
SUCCESS: executed without exceptions
root - INFO - Testing train_models:
SUCCESS: rfc_model.pkl correctly generated
root - INFO - Testing train_models:
SUCCESS: logistic_model.pkl correctly generated
root - INFO - Testing train_models:
SUCCESS: roc_curve_results.png correctly generated
root - INFO - Testing train_models:
SUCCESS: feature_importances.png correctly generated
root - INFO - Testing train_models:
SUCCESS: rf_results.png correctly generated
root - INFO - Testing train_models:
SUCCESS: logistic_results.png correctly generated
```

## Dependencies
The project requires the following python libraries:

- scikit-learn==0.24.1
- shap
- pylint
- autopep8
- matplotlib
- seaborn
- joblib
- numpy

It is possible to quickly instal all the dependencies running the following
command:

```
pip install -r requirments.txt
```
