import acquire


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
    
