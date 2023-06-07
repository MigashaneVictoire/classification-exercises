import pandas as pd
import env


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