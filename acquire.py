import os
from pandas import read_csv
import pandas as pd
import env


def catch_encoding_errors(fileName) -> str:
    
    """
    parameters:
        fileName: csv file name. Should look like (file.csv)
    return:
        file dataframe with no encoding errors
    """
    # import needed
    from pandas import read_csv
    
    # list of encodings to check for
    encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-16']
    
    # check encodings and return dataframe
    for encoding in encodings:
        try:
            df = pd.read_csv(fileName, encoding=encoding)
            break
        except UnicodeDecodeError:
            print(f"Failed to decode with {encoding} encoding.")
    return df

def get_existing_csv_file(fileName) -> str:
    """
    parameters:
        fileName: csv file name. Should look like (file.csv)
    return:
        file dataframe with no encoding errors after cheking for existance of file
    """
    if os.path.isfile(fileName):
        return catch_encoding_errors(fileName)
    else:
        print(f"file with name {fileName} does not exist.")
        
        
def get_titanic_data():
    query = """
    SELECT *
    FROM passengers;
    """
    titanic_data = pd.read_sql(query, env.get_db_access("titanic_db"))
    
    return titanic_data


def get_iris_data():
    query = """
    SELECT *
    FROM measurements m
    JOIN species s ON m.species_id = s.species_id;
    """
    iris_data = pd.read_sql(query, env.get_db_access("iris_db"))
    
    return iris_data


def get_telco():
    query = """
    SELECT *
    FROM customers #payment_types
    JOIN contract_types ct USING(contract_type_id)
    JOIN internet_service_types ist USING(internet_service_type_id)
    JOIN payment_types pt USING(payment_type_id);
    """
    telco_data = pd.read_sql(query, env.get_db_access("telco_churn"))
    
    return telco_data