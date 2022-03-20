'''
This module tests and verifies churn_library code.

author: Antonino Vespoli
date: March 20, 2022
'''

from os.path import exists, basename
from os import stat
import logging as lg

import churn_library as cls
import constants as cst

lg.basicConfig(
    filename='./logs/churn_library.log',
    level=lg.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def is_file_valid(path):
    '''
    check if a file exists and it is not empty
    '''
    return exists(path) and stat(path).st_size > 0


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the
    other test functions
    '''
    try:
        data = import_data("./data/bank_data.csv")
        lg.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        lg.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert not data.isnull().values.any()
        assert not data.isna().values.any()
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        lg.error("Testing import_data: The file doesn't appear to have rows "
                 "and columns")
        raise err

    return data


def test_eda(perform_eda, data):
    '''
    test perform eda function
    '''

    try:
        perform_eda(data)
        lg.info("Testing perform_eda:\nSUCCESS: executed without exceptions")
    except Exception as err:
        lg.error(
            "Testing perform_eda:\nERROR: execution generated exception %s",
            err)

    # List of file that perform_eda is expected to generate
    list_eda_generated_files = [cst.CHURN_DISTRIBUTION_IMG_PATH,
                                cst.CUSTOMER_AGE_DISTRIBUTION_IMG_PATH,
                                cst.MARITAL_DISTRIBUTION_IMG_PATH,
                                cst.TOTAL_TRANSACTION_IMG_PATH,
                                cst.HEATMAP_IMG_PATH]

    # Check perform_eda has generated the expected files
    for file_path in list_eda_generated_files:
        file_name = basename(file_path)
        try:
            assert is_file_valid(file_path)

            lg.info("Testing perform_eda:\nSUCCESS: %s correctly generated",
                    file_name)
        except AssertionError as err:
            lg.error("Testing perform_eda:\nERROR: %s has not been generated "
                     "correctly", file_name)
            raise err


def test_encoder_helper(encoder_helper, data, response):
    '''
    test encoder helper
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    try:
        encoder_helper(data, cat_columns, response)
        lg.info("Testing encoder_helper:\nSUCCESS: executed without exceptions")
    except Exception as err:
        lg.error("Testing encoder_helper:\nERROR: execution generated "
                 "exception %s", err)

    for category in cat_columns:
        col_name = category + '_' + response
        try:
            assert col_name in data
            assert not data[col_name].isna().any()

            lg.info("Testing encoder_helper:\nSUCCESS: %s correctly added",
                    col_name)
        except AssertionError as err:
            lg.error(
                "Testing encoder_helper:\nERROR: %s has not been generated"
                " added", col_name)
            raise err


def test_perform_feature_engineering(
        perform_feature_engineering, data, response):
    '''
    test perform_feature_engineering
    '''

    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            data, response)
        lg.info("Testing perform_feature_engineering:\nSUCCESS: executed "
                "without exceptions")
    except Exception as err:
        lg.error("Testing perform_feature_engineering:\nERROR: execution "
                 "generated exception %s", err)

    try:
        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        assert x_train.shape[0] + x_test.shape[0] == data.shape[0]
        assert y_train.shape[0] + y_test.shape[0] == data.shape[0]
        lg.info("Testing perform_feature_engineering:\nSUCCESS: output values "
                "passed all tests")
    except AssertionError as err:
        lg.error("Testing perform_feature_engineering:\nERROR: invalid output "
                 "values")

    return x_train, x_test, y_train, y_test


def test_train_models(train_models, x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''

    try:
        train_models(x_train, x_test, y_train, y_test)
        lg.info("Testing train_models:\nSUCCESS: executed without exceptions")
    except Exception as err:
        lg.error(
            "Testing train_models:\nERROR: execution generated exception %s",
            err)

    # List of file that perform_eda is expected to generate
    list_train_models_generated_files = [cst.RFC_MODEL_PATH,
                                         cst.LOGISTIC_MODEL_PATH,
                                         cst.ROC_CURVE_RES_IMG_PATH,
                                         cst.FEATURE_IMPORT_IMG_PATH,
                                         cst.RF_RES_IMG_PATH,
                                         cst.LOGISTIC_RES_IMG_PATH]

    # Check perform_eda has generated the expected files
    for file_path in list_train_models_generated_files:
        file_name = basename(file_path)
        try:
            assert is_file_valid(file_path)

            lg.info("Testing train_models:\nSUCCESS: %s correctly generated",
                    file_name)
        except AssertionError as err:
            lg.error("Testing train_models:\nERROR: %s has not been generated "
                     "correctly", file_name)
            raise err


if __name__ == "__main__":
    customer_data = test_import(cls.import_data)

    test_eda(cls.perform_eda, customer_data)

    test_encoder_helper(cls.encoder_helper, customer_data, 'Churn')

    x_data_train, x_data_test, y_data_train, y_data_test = test_perform_feature_engineering(
        cls.perform_feature_engineering, customer_data, 'Churn')

    test_train_models(
        cls.train_models,
        x_data_train,
        x_data_test,
        y_data_train,
        y_data_test)
