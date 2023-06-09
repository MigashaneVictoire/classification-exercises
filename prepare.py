# For funtion annotations
from typing import Union
from typing import Tuple

import pandas as pd
import numpy as np
import acquire
from sklearn.model_selection import train_test_split


def get_prep_insight(df):
    """
    Used to plot and give value caouts of a givve dataframe
    note: this function will separete numeric from object data colums
    """
    nums_df = telco.select_dtypes("number").columns
    for i in nums_df:
        plt.figure(figsize=(5,3))
        print(i)
        print(df[i].value_counts(dropna=False))
        print(df[i].value_counts(normalize=True))
        df[i].value_counts( dropna=False).plot.bar()
        plt.show()
        print("\n\n")

    print("_" * 50)
    print("_" * 15, "Non numeric columns (objects)","_" *15)
    obj_df = telco.select_dtypes("object").columns
    for i in obj_df:
        plt.figure(figsize=(5,3))
        print(i)
        print(df[i].value_counts(dropna=False))
        df[i].value_counts( dropna=False).plot.bar()
        plt.show()
        print("\n\n")
    


# Create a function named prep_iris that accepts the untransformed iris data, and returns the data with the transformations above applied.

def prep_iris():
    iris = acquire.get_iris_data()
    
    # Drop the species_id and measurement_id columns.
    iris = iris.drop(columns=["species_id"])
    
    # Rename the species_name column to just species.
    iris = iris.rename(columns={"species_name":"species"})
    
    # get dummy variables
    get_dummies = pd.get_dummies(iris.species, drop_first=True)
    
    # Concatenate dummies to the iris dataframe.
    iris = pd.concat([iris,get_dummies], axis=1)
    return iris
    

def prep_titanic():
    '''
    This function will clean the the titanic dataset
    '''
    # Data
    titanic = acquire.get_titanic_data()
    
    # drop the class column from the dataframe becuase it is a cuplicate column
    # drop duplicates and un necessery columns
    drop_cols = ["class", "deck", "embark_town"]
    titanic = titanic.drop(columns=drop_cols, axis=1)
    
    # separeate numeric from object data types and get columns names
    object_titanic = titanic.select_dtypes("object").columns

    # make dummies for all categorical variables
    all_dummies = pd.get_dummies(titanic[object_titanic], drop_first=True)
    
    # concate the dummies back to the data frame
    titanic = pd.concat([titanic, all_dummies], axis=1)
    
    return titanic


def prep_telco():
    
    telco = acquire.get_telco()
    
    # remove the non numerics in the total charges from data frame
    print("original size:",len(telco))
    telco = telco[telco.total_charges.str.contains(" ") == False]

    # change the data type of total charges
    telco.total_charges = telco.total_charges.astype("float")
    telco.total_charges.dtype
    
    # drop duplicates and un necessery columns
    drop_cols = ["customer_id"]
    telco = telco.drop(columns=drop_cols, axis=1)

    # separate numeric from object columns
    object_telco = telco.select_dtypes("object").columns
    
    # get dummie variables
    all_dummies = pd.get_dummies(telco[object_telco], drop_first=True)
    
    # conect dummies to the originaal data
    telco = pd.concat([telco,all_dummies], axis=1)
    
    return telco



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
