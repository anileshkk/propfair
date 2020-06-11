import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
from scipy import stats
from sklearn.model_selection import train_test_split

def compas_preprocess(data_file, split):
    
    read_data = pd.read_csv(data_file)
    column_names = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']
    
    data = read_data[column_names]
    print(data.columns)

    for t in range(0):
        data = pd.concat([data, data.copy()], ignore_index = True)
    
    data = data[data['days_b_screening_arrest'] <= 30]
    data = data[data['days_b_screening_arrest'] >= -30]
    data = data[data['is_recid'] != -1]
    data = data[data['c_charge_degree'] != "O"]
    data = data[data['score_text'] != "N/A"]

    data['c_jail_in'] = data['c_jail_in'].astype('datetime64[ns]')
    data['c_jail_out'] = data['c_jail_out'].astype('datetime64[ns]')
    data['length_of_stay'] = data['c_jail_out']-data['c_jail_in']
    data['length_of_stay'] = data['length_of_stay'].dt.total_seconds()
    data['risk'] = (data['score_text'] != "Low").astype('float')

    data = data.drop(columns = ['decile_score', 'is_recid', 'two_year_recid','c_jail_in', 'c_jail_out', 'score_text'])

    for col in set(data.columns) - set(data.describe().columns):
        data[col] = data[col].astype('category')

    all_data = data.drop(columns = ['risk'])
    all_label = data.risk

    

    if split == True:
        train_data, test_data, train_label, test_label = train_test_split(all_data, all_label, test_size = 0.2, random_state = 0)
        return train_data, train_label, test_data, test_label
    else:
        return all_data, all_label


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
