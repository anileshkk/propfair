import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess(train_file, test_file):
    
#list feature columns
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational_num','marital_status', 'occupation', 'relationship', 'race', 'gender','capital_gain', 'capital_loss', 'hours_per_week', 'native_country','income']

#read train and test files
    train = pd.read_csv(train_file, sep=",\s", header=None, names = column_names, engine = 'python')
    test = pd.read_csv(test_file, sep=",\s", header=None, names = column_names, engine = 'python')

#some preprocessing
    test['income'].replace(regex=True,inplace=True,to_replace=r'\.',value=r'')

#drop rows with missing columns
    train.replace('?', np.nan, inplace=True)
    train.dropna(inplace=True)
    train.reset_index(drop=True, inplace=True)

    test.replace('?', np.nan, inplace=True)
    test.dropna(inplace=True)
    test.reset_index(drop=True, inplace=True)

    #sampling a smaller training dataset
    #train = train.sample(3000, random_state = 0)

# Setting all the categorical columns to type category
    for col in set(train.columns) - set(train.describe().columns):
        train[col] = train[col].astype('category')

    for col in set(test.columns) - set(test.describe().columns):
        test[col] = test[col].astype('category')

#dropping some columns for convenience
    train_data = train.drop(columns = ['income', 'native_country', 'capital_gain'])
    train_label = train.income
    
    test_data = test.drop(columns = ['income', 'native_country', 'capital_gain'])
    test_label = test.income
    
    return train_data, train_label, test_data, test_label


def get_1hot(data):
    
    cat_1hot = pd.get_dummies(data.select_dtypes('category'))
    non_cat = data.select_dtypes(exclude = 'category')
    data_1hot = pd.concat([non_cat, cat_1hot], axis=1, join='inner')

    return data_1hot

def accuracy_randomized(classifiers, weights, data, label):
    overall_acc = 0
    for i in range(len(weights)):
        clf = classifiers[i]
        overall_acc += weights[i] * clf.score(data, label)
    return overall_acc

def sample_sizes(l, correct_size, wrong_size, samples_correct_0, samples_wrong_0):
    if samples_wrong_0 > wrong_size:
        samples_wrong = wrong_size
        samples_correct = l - samples_wrong
    elif samples_correct_0 > correct_size:
        samples_correct = correct_size
        samples_wrong = l - correct_size
    else:
        samples_wrong = samples_wrong_0
        samples_correct = samples_correct_0
        
    return samples_correct, samples_wrong
