# This utility file should alway go where the env file is.

# For funtion annotations
from binascii import a2b_qp
from typing import Union
from typing import Tuple

# Python libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Mac libraries
import os

# Personal libraries
import acquire_
import env

#  Data preparation
# -----------------------------------------------------------------
def internal_data_prep_():
    return

# return prepared data based on input type
def prep_data_(df: pd.DataFrame = None, fileName: str = None):
    if df and fileName:
        print("Enter dataFrame or file name NOT both...!")
    elif fileName:
        df = acquire_.get_existing_csv_file_(fileName=fileName)
        return internal_data_prep_(df)
    else:
        return internal_data_prep_(df)

# -----------------------------------------------------------------
# Split the data into train, validate and train
def split_data_(df: pd.DataFrame, test_size: float =.2, validate_size: float =.2, stratify_col: str =None, random_state: int =None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    parameters:
        df: pandas dataframe you wish to split
        test_size: size of your test dataset
        validate_size: size of your validation dataset
        stratify_col: the column to do the stratification on
        random_state: random seed for the data

    return:
        train, validate, test DataFrames
    '''
    # no stratification
    if stratify_col == None:
        # split test data
        train_validate, test = train_test_split(df, 
                                                test_size=test_size, 
                                                random_state=random_state)
        # split validate data
        train, validate = train_test_split(train_validate, 
                                           test_size=validate_size/(1-test_size),
                                            random_state=random_state)
    # stratify split
    else:
        # split test data
        train_validate, test = train_test_split(df,
                                                test_size=test_size,
                                                random_state=random_state, 
                                                stratify=df[stratify_col])
        # split validate data
        train, validate = train_test_split(train_validate, 
                                           test_size=validate_size/(1-test_size),
                                           random_state=random_state, 
                                           stratify=train_validate[stratify_col])
    return train, validate, test