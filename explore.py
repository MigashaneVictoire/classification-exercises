import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools

import prepare


# iris
def iris_univariate_exploare():
    # get training data
    iris = prepare.prep_iris()
    train, validate, test = prepare.split_data_(df=iris, stratify_col="species", random_state=95)
    
    # separeate discrete from continuous variables
    numeric_col = []
    categorical_col = []

    for col in train.columns:
        if train[col].dtype == "o":
            categorical_col.appen(col)

        else:
            if len(train[col].unique()) < 5: #making anything with less than 4 unique values a catergorical value
                categorical_col.append(col)
            else:
                numeric_col.append(col)
     # for numeric columsn
    # count all values and normalization of the each column
    for col in numeric_col:
        plt.figure(figsize=(5,3))
        print(col.upper())
        print(train[col].value_counts(dropna=False).sort_values(ascending=False))
        print(train[col].value_counts(dropna=False, normalize=True).sort_values(ascending=False))
        train[col].value_counts(dropna=False).hist()
        plt.show()
        train[col].hist(alpha=.5)
        plt.show()
    
    # For categorical columns
    # count all values and normalization of the each column
    for col in categorical_col:
        plt.figure(figsize=(5,3))
        print(col.upper())
        print(train[col].value_counts(dropna=False).sort_values(ascending=False))
        print(train[col].value_counts(dropna=False, normalize=True).sort_values(ascending=False))
        sns.countplot(x=col, data=train)
        plt.show()
        
        
def iris_bivarial_explore():
    
    # get training data
    iris = prepare.prep_iris()
    train, validate, test = prepare.split_data_(df=iris, stratify_col="species", random_state=95)
    
    # get combination of all columns paired with the target column
    columns = train.columns[1:-3]
    target = "species"
    combinations = []
    for element in columns:
        combinations.append((target, element))
    
    # Visuals
    for col in combinations:
        # descriptive statistics
        print(col[0].upper(), "vs", col[1].upper())
        print(train[col[1]].describe())
        print(train[col[0]].describe())

        # first figure
        plt.figure(figsize=(5,3))
        sns.swarmplot(x=train[col[0]], y=train[col[1]], hue=train[col[0]])
        plt.title(f"{col[0].upper()} vs {col[1].upper()}")
        plt.show()

        # second figure
        plt.figure(figsize=(5,3))
        sns.barplot(x=train[col[0]], y=train[col[1]])
        plt.title(f"{col[0].upper()} vs {col[1].upper()}")
        plt.show()
        
        plt.figure(figsize=(5,3))
        sns.boxplot(x=train[col[0]], y=train[col[1]])
        plt.title(f"{col[0].upper()} vs {col[1].upper()}")
        plt.show()


        plt.figure(figsize=(5,3))
        sns.violinplot(x=train[col[0]], y=train[col[1]])
        plt.title(f"{col[0].upper()} vs {col[1].upper()}")
        plt.show()


# check if the variance are equal from the levene test
def check_variance_equality(p_value, alpha=0.05):
    if p_value > alpha:
        return "equal"
    else:
        return "not equal"
    
# check if the data rejects the null hypothesis from the t-test
def check_null_with_stats(t_stat,p_value, alpha=0.05):
    if (t_stat > 0) and (p_value < alpha):
        return "We have enough evidence to rejec the NULL... \n"
    else:
        return "We FAIL to reject the null... \n"
    
# full test statistics on measurements
def bivarial_stat_test():
    # get training data
    iris = prepare.prep_iris()
    train, validate, test = prepare.split_data_(df=iris, stratify_col="species", random_state=95)
    
    print("- Null-Hyp: There is a significant difference between viginical's indiviaul measurements mean and versicolor's individual measurements mean.")
    print("- Alt-Hyp: There is no significant difference between viginical petal_width mean and vrrsicolor petal_width mean\n")
          
    # set variable data
    virginica = train[train.species == "virginica"]
    versicolor = train[train.species == "versicolor"]

    # measurement columns
    measurements = train.columns[1:-3]
    
    # get visuals and discriptive statistics
    for ele in measurements:
        # get mean and standar deviation
        virginica_mean, virginica_std = virginica[ele].mean(), virginica[ele].std()
        versicolor_mean, versicolor_std = versicolor[ele].mean(), versicolor[ele].std()

        # generate distributions
        virginica_dist = stats.norm(virginica_mean,virginica_std)
        versicolor_dist = stats.norm(versicolor_mean, versicolor_std)

        # generate random variables
        virginica_random = virginica_dist.rvs(100_000)
        versicolor_random = versicolor_dist.rvs(100_000)

        # descriptive stats
        print("Virginica",ele,"number of observations:", len(virginica[ele]))
        print("Virginica",ele,"variance:",virginica[ele].var())

        # create sub-plots
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].hist(virginica_random, bins=20, alpha=0.5, color='blue', label='Virginica')
        axs[0].set_title(f'Virginica {ele}', fontsize=12)
        axs[0].set_xlabel('Distribution', fontsize=10)
        axs[0].set_ylabel('Frequency', fontsize=10)
        axs[0].tick_params(axis='both', which='major', labelsize=8)
        axs[0].grid(True, linestyle='--', linewidth=0.5)
        axs[0].legend()

        # descripitive stats
        print("\nVersicolor",ele,"number of observations:", len(virginica[ele]))
        print("Versicolor",ele,"variance:",versicolor[ele].var())

        axs[1].hist(versicolor_random, bins=20, alpha=0.5, color='green', label='Versicolor')
        axs[1].set_title(f'Versicolor {ele}', fontsize=12)
        axs[1].set_xlabel('Distribution', fontsize=10)
        axs[1].set_ylabel('Frequency', fontsize=10)
        axs[1].tick_params(axis='both', which='major', labelsize=8)
        axs[1].grid(True, linestyle='--', linewidth=0.5)
        axs[1].legend()

        plt.tight_layout()
        plt.show()
          

    # confidence level
    alpha = 0.05

    # test stats for all measurements
    for ele in measurements:
        t_stat, p_value = stats.levene(virginica[ele],versicolor[ele])

        print("levene of", ele,"p-value:", p_value)

        # check variance equality
        variance_status = check_variance_equality(p_value)
        print(variance_status, "variance")

        if variance_status == "equal":
            t_stat, p_value = stats.ttest_ind(virginica[ele],versicolor[ele])
        else:
            t_stat, p_value = stats.mannwhitneyu(virginica[ele],versicolor[ele])

        # ckec final p-value
        print(check_null_with_stats(t_stat,p_value))

    

# Visualize two numeric variables of the species. Hint: sns.relplot with hue or col

def iris_multivariate_explore():
    
    # get training data
    iris = prepare.prep_iris()
    train, validate, test = prepare.split_data_(df=iris, stratify_col="species", random_state=95)
    
    # get combination of all columns paired with one other column
    columns = train.columns[1:-3]
    target = "species"
    combinations = itertools.combinations(columns, 2)
    # Visuals
    for col in combinations:
        # descriptive statistics
        print(col[0].upper(), "vs", col[1].upper())

        # first figure
        plt.figure(figsize=(5,3))
        sns.relplot(x=train[col[0]], y=train[col[1]], hue=train[target])
        plt.title(f"{col[0].upper()} vs {col[1].upper()}")
        plt.show()

        
def melted_numeric_plots():
    # get training data
    iris = prepare.prep_iris()
    train, validate, test = prepare.split_data_(df=iris, stratify_col="species", random_state=95)
    
    melted_train = train[train.columns[1:-2]].melt(id_vars="species")

    sns.stripplot(x=melted_train.variable, y=melted_train.value, hue=melted_train.species)
    plt.ylabel("Friquency")
    plt.xlabel("Measurements")
    plt.show()
    
    sns.boxplot(x=melted_train.variable, y=melted_train.value, hue=melted_train.species)
    plt.ylabel("Friquency")
    plt.xlabel("Measurements")
    plt.show()
    
  
def sepal_area_test():
    # get training data
    iris = prepare.prep_iris()
    train, validate, test = prepare.split_data_(df=iris, stratify_col="species", random_state=95)

    print("- Null-Hyp: The sepal area is signficantly different in virginica compared to setosa?")
    print("- Alt-Hyp: The sepal area is not signficantly different in virginica compared to setosa?\n")

    # get the erea of the species sepal and add it to each of the data frames
    train["sepal_area"] = train.sepal_width * train.sepal_length

    # set variable data
    virginica = train[train.species == "virginica"]
    setosa = train[train.species == "setosa"]

    # get visuals and discriptive statistics

    # get mean and standar deviation
    virginica_mean, virginica_std = virginica.sepal_area.mean(), virginica.sepal_area.std()
    setosa_mean, setosa_std = setosa.sepal_area.mean(), setosa.sepal_area.std()

    # generate distributions
    virginica_dist = stats.norm(virginica_mean,virginica_std)
    setosa_dist = stats.norm(setosa_mean, setosa_std)

    # generate random variables
    virginica_random = virginica_dist.rvs(100_000)
    setosa_random = setosa_dist.rvs(100_000)

    # descriptive stats
    print("Virginica sepal area number of observations:", len(virginica.sepal_area))
    print("Virginica sepal area variance:",virginica.sepal_area.var())

    # create sub-plots
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(virginica_random, bins=20, alpha=0.5, color='blue', label='Virginica')
    axs[0].set_title(f'Virginica sepal area', fontsize=12)
    axs[0].set_xlabel('Distribution', fontsize=10)
    axs[0].set_ylabel('Frequency', fontsize=10)
    axs[0].tick_params(axis='both', which='major', labelsize=8)
    axs[0].grid(True, linestyle='--', linewidth=0.5)
    axs[0].legend()

    # descriptive stats
    print("\nSetosa sepal area number of observations:", len(setosa.sepal_area))
    print("Setosa sepal area variance:",setosa.sepal_area.var())

    axs[1].hist(setosa_random, bins=20, alpha=0.5, color='green', label='Setosa')
    axs[1].set_title(f'Setosa sepal area', fontsize=12)
    axs[1].set_xlabel('Distribution', fontsize=10)
    axs[1].set_ylabel('Frequency', fontsize=10)
    axs[1].tick_params(axis='both', which='major', labelsize=8)
    axs[1].grid(True, linestyle='--', linewidth=0.5)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    # confidence level
    alpha = 0.05

    # test stats for sepal area
    t_stat, p_value = stats.levene(virginica.sepal_area, setosa.sepal_area)

    print("levene of sepal_area p-value:", p_value)

    # check variance equality
    variance_status = check_variance_equality(p_value)
    print(variance_status, "variance")

    if variance_status == "equal":
        t_stat, p_value = stats.ttest_ind(virginica.sepal_area, setosa.sepal_area)
    else:
        t_stat, p_value = stats.mannwhitneyu(virginica.sepal_area, setosa.sepal_area)

    # ckec final p-value
    print(check_null_with_stats(t_stat,p_value))
    
    
    
###########################################################################
# Titanic

# Univariate test
def titanic_univariate_exploare():
    # get training data
    titanic = prepare.prep_titanic()
    train, validate, test = prepare.split_data_(df=titanic, stratify_col="survived", random_state=95)

    # separeate discrete from continuous variables
    numeric_col = []
    categorical_col = []

    for col in train.columns:
        if train[col].dtype == "o":
            categorical_col.appen(col)

        else:
            if len(train[col].unique()) < 5: #making anything with less than 4 unique values a catergorical value
                categorical_col.append(col)
            else:
                numeric_col.append(col)
    # for numeric columsn
    # count all values and normalization of the each column
    for col in numeric_col:
        plt.figure(figsize=(5,3))
        print(col.upper())
        print(train[col].value_counts(dropna=False).sort_values(ascending=False))
        print(train[col].value_counts(dropna=False, normalize=True).sort_values(ascending=False))
        train[col].value_counts(dropna=False).hist()
        plt.show()
        train[col].hist(alpha=.5)
        plt.show()

    # For categorical columns
    # count all values and normalization of the each column
    for col in categorical_col:
        plt.figure(figsize=(5,3))
        print(col.upper())
        print(train[col].value_counts(dropna=False).sort_values(ascending=False))
        print(train[col].value_counts(dropna=False, normalize=True).sort_values(ascending=False))
        sns.countplot(x=col, data=train)
        plt.show()
        
        
# bivariate test
def titanic_bivarial_explore():
    
    # get training data
    titanic = prepare.prep_titanic()
    train, validate, test = prepare.split_data_(df=titanic, stratify_col="survived", random_state=95)
    
    # get combination of all columns paired with the target column
    columns = train.columns[2:-3]
    target = "survived"
    combinations = []
    for element in columns:
        combinations.append((target, element))
    
    # Visuals
    for col in combinations:
        # descriptive statistics
        print(col[0].upper(), "vs", col[1].upper())
        print(train[col[1]].describe())
        print(train[col[1]].value_counts())

        # first figure
        plt.figure(figsize=(5,3))
        sns.stripplot(x=train[col[0]], y=train[col[1]], hue=train[col[0]])
        plt.title(f"{col[0].upper()} vs {col[1].upper()}")
        plt.show()

        # second figure
        plt.figure(figsize=(5,3))
        sns.barplot(x=train[col[0]], y=train[col[1]])
        plt.title(f"{col[0].upper()} vs {col[1].upper()}")
        plt.show()
        
        plt.figure(figsize=(5,3))
        sns.boxplot(x=train[col[0]], y=train[col[1]])
        plt.title(f"{col[0].upper()} vs {col[1].upper()}")
        plt.show()


        plt.figure(figsize=(5,3))
        sns.violinplot(x=train[col[0]], y=train[col[1]])
        plt.title(f"{col[0].upper()} vs {col[1].upper()}")
        plt.show()
    
    
def titanic_multivariate_explore():
    
    # get training data
    titanic = prepare.prep_titanic()
    train, validate, test = prepare.split_data_(df=titanic, stratify_col="survived", random_state=95)
    
    # get combination of all columns paired with one other column
    columns = train.columns[2:-3]
    target = "survived"
    combinations = itertools.combinations(columns, 2)
    # Visuals
    for cols in combinations:
        # descriptive statistics
        print(cols[0].upper(), "vs", cols[1].upper())

        # first figure
        plt.figure(figsize=(5,3))
        sns.stripplot(x=train[cols[0]], y=train[cols[1]], hue=train[target])
        plt.title(f"{cols[0].upper()} vs {cols[1].upper()}")
        plt.show()
        
   
def titanic_melted_numeric_plots():
    # get training data
    titanic = prepare.prep_titanic()
    train, validate, test = prepare.split_data_(df=titanic, stratify_col="survived", random_state=95)
    
    # select only original numeric variables and melt them
    melted_train = train[train.columns[1:-3]].select_dtypes("number").melt(id_vars="survived")
    
    sns.stripplot(x=melted_train.variable, y=melted_train.value, hue=melted_train.survived)
    plt.ylabel("Friquency")
    plt.xlabel("passanger discription")
    plt.show()
    
    sns.boxplot(x=melted_train.variable, y=melted_train.value, hue=melted_train.survived)
    plt.ylabel("Friquency")
    plt.xlabel("Measurements")
    plt.show()
    
    
def fare_and_age_relationship_test():

    # get training data
    titanic = prepare.prep_titanic()
    train, validate, test = prepare.split_data_(df=titanic, stratify_col="survived", random_state=95)

    print("- Null-Hyp: There is no linear relationship between fare and age.")
    print("- Alt-Hyp: There is is linear relationship between fare and age.\n")

    # plot
    plt.figure(figsize=(5,3))
    plt.scatter(x=train.age, y=train.fare)
    plt.title('age vs fare', fontsize=12)
    plt.xlabel('age', fontsize=10)
    plt.ylabel('fare', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    print("I don't see a linear relationship so I will do spearman's r test")

    # confidence level
    alpha = 0.05

    # test stats for sepal area
    coef_r, p_value = stats.spearmanr(train.age, train.fare)

    print("p-value:", p_value)

    if p_value < alpha:
        print("we can reject the null")
    else:
        print("we FAIL to reject the null")


###########################################################################
# Telco

def telco_univariate_exploare():
    # get training data
    telco = prepare.prep_telco()
    train, validate, test = prepare.split_data_(df=telco, stratify_col="churn", random_state=95)

    # separeate discrete from continuous variables
    numeric_col = []
    categorical_col = []

    for col in train.columns:
        if train[col].dtype == "o":
            categorical_col.appen(col)

        else:
            if len(train[col].unique()) < 5: #making anything with less than 4 unique values a catergorical value
                categorical_col.append(col)
            else:
                numeric_col.append(col)
                
                
    # for numeric columsn
    # count all values and normalization of the each column
    for col in numeric_col:
        plt.figure(figsize=(5,3))
        print(col.upper())
        print(train[col].value_counts(dropna=False).sort_values(ascending=False))
        print(train[col].value_counts(dropna=False, normalize=True).sort_values(ascending=False))
        train[col].value_counts(dropna=False).hist()
        plt.show()
        train[col].hist(alpha=.5)
        plt.show()

        
    # For categorical columns
    # count all values and normalization of the each column
    for col in categorical_col[1:-26]:
        plt.figure(figsize=(5,3))
        print(col.upper())
        print(train[col].value_counts(dropna=False).sort_values(ascending=False))
        print(train[col].value_counts(dropna=False, normalize=True).sort_values(ascending=False))
        sns.countplot(x=col, data=train)
        plt.show()

        
        
 
