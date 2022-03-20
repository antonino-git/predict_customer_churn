# library doc string
'''
Library that implement a Machine learning model that predicts credit card
customers that most likely to churn.

author: Antonino Vespoli
date: March 20, 2022
'''

# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import constants as cst
sns.set()


def import_data(pth: str) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''

    try:
        data = pd.read_csv(pth)
        # lg.info("SUCCESS: File {} opened correctly".format(pth))
        return data
    except FileNotFoundError as err:

        # lg.error("FAILED: import_data failed to open file with path", pth)
        raise err


def perform_eda(data: pd.DataFrame):
    '''
    perform eda on df and save figures to images folder
    input:
            data: pandas dataframe

    output:
            None
    '''

    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Set EDA figures to size 20x10
    plt.figure(figsize=(20, 10))

    # Generate Churn histogram and save it in images folder
    data['Churn'].hist()
    plt.savefig(cst.CHURN_DISTRIBUTION_IMG_PATH)

    # Generate customer age histogram and save it in images folder
    plt.figure(figsize=(20, 10))
    data['Customer_Age'].hist()
    plt.savefig(cst.CUSTOMER_AGE_DISTRIBUTION_IMG_PATH)

    # Generate marital status histogram and save it in images folder
    plt.figure(figsize=(20, 10))
    data.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(cst.MARITAL_DISTRIBUTION_IMG_PATH)

    # Generate total transaction distribution diagram and save it in images
    # folder
    plt.figure(figsize=(20, 10))
    sns.histplot(data['Total_Trans_Ct'], kde=True, stat="density", linewidth=0)
    plt.savefig(cst.TOTAL_TRANSACTION_IMG_PATH)

    # Generate feature heatmap and save it in images folder
    plt.figure(figsize=(20, 10))
    sns.heatmap(data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(cst.HEATMAP_IMG_PATH)


def encoder_helper(
        data: pd.DataFrame,
        category_lst: list,
        response: str = 'Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the
    notebook

    input:
            data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            data: pandas dataframe with new columns for
    '''
    for category in category_lst:
        result_lst = []
        result_group = data.groupby(category).mean()[response]

        for val in data[category]:
            result_lst.append(result_group.loc[val])

        result_col_name = category + '_' + response
        data[result_col_name] = result_lst


def perform_feature_engineering(data: pd.DataFrame, response: str):
    '''
    input:
              data: pandas dataframe
              response: string of response name [optional argument that could be
              used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    x_data = pd.DataFrame()
    y_data = data[response]

    encoder_helper(data, cat_columns, response)

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    x_data[keep_cols] = data[keep_cols]

    x_data.head()

    return train_test_split(x_data, y_data, test_size=0.3, random_state=42)


def classification_report_image(y_train: pd.DataFrame,
                                y_test: pd.DataFrame,
                                y_train_preds_lr: np.ndarray,
                                y_train_preds_rf: np.ndarray,
                                y_test_preds_lr: np.ndarray,
                                y_test_preds_rf: np.ndarray):
    '''
    produces classification report for training and testing results and stores
    report as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.clf()
    fig, axs = plt.subplots(2)
    plt.rc('figure', figsize=(5, 5))
    axs[0].text(0.01, 1.25, str('Random Forest Train'), {
        'fontsize': 10}, fontproperties='monospace')
    axs[0].text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    axs[0].axis('off')
    axs[1].text(0.01, 0.6, str('Random Forest Test'), {
        'fontsize': 10}, fontproperties='monospace')
    axs[1].text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    axs[1].axis('off')
    fig.savefig(cst.RF_RES_IMG_PATH)

    plt.clf()
    fig, axs = plt.subplots(2)
    plt.rc('figure', figsize=(5, 5))
    axs[0].text(0.01, 1.25, str('Logistic Regression Train'),
                {'fontsize': 10}, fontproperties='monospace')
    axs[0].text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    axs[0].axis('off')
    axs[1].text(0.01, 0.6, str('Logistic Regression Test'), {
        'fontsize': 10}, fontproperties='monospace')
    axs[1].text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    axs[1].axis('off')
    plt.savefig(cst.LOGISTIC_RES_IMG_PATH)


def feature_importance_plot(
        model: GridSearchCV,
        x_data: pd.DataFrame,
        output_pth: str):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)


def train_models(
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: x training data
              x_test: x testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))

    lrc_plot = plot_roc_curve(lrc, x_test, y_test)

    # plots
    plt.figure(figsize=(15, 8))
    axis_x = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=axis_x,
        alpha=0.8)
    lrc_plot.plot(ax=axis_x, alpha=0.8)
    plt.show()

    # save best model
    joblib.dump(cv_rfc.best_estimator_, cst.RFC_MODEL_PATH)
    joblib.dump(lrc, cst.LOGISTIC_MODEL_PATH)

    rfc_model = joblib.load(cst.RFC_MODEL_PATH)
    lr_model = joblib.load(cst.LOGISTIC_MODEL_PATH)

    lrc_plot = plot_roc_curve(lr_model, x_test, y_test)

    plt.figure(figsize=(15, 8))
    axis_x = plt.gca()
    plot_roc_curve(rfc_model, x_test, y_test, ax=axis_x, alpha=0.8)
    lrc_plot.plot(ax=axis_x, alpha=0.8)
    plt.savefig(cst.ROC_CURVE_RES_IMG_PATH)
    plt.show()

    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar")

    feature_importance_plot(cv_rfc, x_train, cst.FEATURE_IMPORT_IMG_PATH)

    classification_report_image(y_train, y_test, y_train_preds_lr,
                                y_train_preds_rf, y_test_preds_lr,
                                y_test_preds_rf)
