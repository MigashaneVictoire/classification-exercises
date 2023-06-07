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
    '''
    This function will clean the the iris dataset
    '''
    iris = acquire.get_iris_data()
    
    # Drop the species_id and measurement_id columns.
    iris = iris.drop(columns="species_id")
    
    # Rename the species_name column to just species.
    iris = iris.rename(columns={"species_name":"species"})
    
    # Create dummy variables of the species name and concatenate onto the iris dataframe.
    return pd.concat([iris,pd.get_dummies(iris.species)], axis=1)

def prep_titanic(df):
    '''
    This function will clean the the titanic dataset
    '''
    df = df.drop(columns =['embark_town','class','age','deck'])

    df.embarked = df.embarked.fillna(value='S')
    
    # Create a function named prep_telco that accepts the raw telco data, and returns the data with the transformations above applied.
    dummy_df = pd.get_dummies(df[['sex','embarked']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df


def prep_telco(telco):
    # drop columns
    drop_cols = ["customer_id"]
    telco = telco.drop(columns=drop_cols, axis=1)
    
    # create dummy variables
    dummy_cols = ["gender","phone_service", "tech_support", "streaming_tv","streaming_movies","paperless_billing","contract_type", "internet_service_type", "churn", "payment_type"]
    all_dummies = pd.get_dummies(telco[dummy_cols], drop_first=True)
    
    telco = pd.concat([telco, all_dummies], axis=1)
    return telco
    


def train_val_test_split(df, target_col):
    """
    Takes in the titanic dataframe and return train, validate, test subset dataframes
    """
    #first split
    train_val, test = train_test_split(df, #dataframe
                    random_state= 95, #setting my random seed
                    test_size= 0.20, #setting the size of my test df
                     stratify=df[target_col]) #stratifying on my target variable
    # second split
    train,validate = train_test_split(train_val,
                                     random_state=90,
                                      test_size=0.25,
                                      stratify=train_val[target_col])
    return train, validate, test
