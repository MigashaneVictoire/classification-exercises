import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools

import prepare

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

    


          