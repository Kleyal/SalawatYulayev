from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy.linalg as LA
import re
import string
from collections import Counter
import json

skf = StratifiedKFold(n_splits=3)



def _calc_roc(model, data, target, skf):
    roc = []
    for train_ind, test_ind in skf.split(data, target):
        model.fit(data[train_ind], target[train_ind])
        roc.append(roc_auc_score(target[test_ind], model.predict_proba(data[test_ind])[:, 1]))
    print(np.array(roc).mean())

def calc_roc(data, target, algo=['logreg'], n_splits=5, print_list=[]):
    skf = StratifiedKFold(n_splits=n_splits)
    if 'logreg' in algo:
        logreg = LogisticRegression(max_iter=3000)
        print('Logistic regression:', end='')
        _calc_roc(logreg, data, target, skf)
    if 'random_forest' in algo:
        clf = RandomForestClassifier(max_depth=10, random_state=0)
        print('Random forest:', end='')
        _calc_roc(clf, data, target, skf)
    if 'naive_bayes' in algo:
        gnb = GaussianNB()
        print('Naive bayes:', end='')
        _calc_roc(gnb, data, target, skf)
    if 'knn' in algo:
        knn = KNeighborsClassifier(n_neighbors=2)
        print('KNN:', end='')
        _calc_roc(knn, data, target, skf)
    if 'decision_tree' in algo:
        clf = DecisionTreeClassifier(random_state=0)
        print('Decision tree:', end='')
        _calc_roc(clf, data, target, skf)
    if 'voting_classifier' in algo:
        clf1 = LogisticRegression(max_iter=3000, random_state=0)
        clf2 = RandomForestClassifier(max_depth=10, random_state=0)
        eclf = VotingClassifier(estimators=[('lr', clf1), ('gnb', clf2)], voting='soft')
        print('Voting Classifier:', end='')
        _calc_roc(eclf, data, target, skf)

def frequency_analysis(row, base_class, words_ones, words_zeros):
    ans = 0.0
    if base_class == 1:
        for i in row.split():
            if i in words_ones:
                ans += words_ones[i]
    elif base_class == 0:
        for i in row.split():
            if i in words_zeros:
                ans += words_zeros[i]
    return ans

def load_data():
    data_b = pd.read_csv('/home/jupyter-admin/Roma/train2.csv', engine='python')
    #data_b = pd.read_csv('/home/jupyter-admin/Roma/test3.csv', engine='python')
    
    data_b['context_text'] = data_b['context_text'].astype(str)
    data_b['phrase_len'] = data_b['phrase'].apply(lambda row : len(row))
    data_b['phrase_num_words'] = data_b['phrase'].apply(lambda row : len(row.split()))
    
    prohibited_symb = re.escape(string.punctuation)
    data_b['zhakar_dist_context_set'] = data_b['context_text'].apply(lambda row : set(re.sub(r'['+prohibited_symb+']', '',row).split()))
    data_b['zhakar_dist_phrase_set'] = data_b['phrase'].apply(lambda row : set(re.sub(r'['+prohibited_symb+']', '',row).split()))
    data_b['zhakar_dist'] = [100 * len(data_b['zhakar_dist_context_set'].iloc[i] & data_b['zhakar_dist_phrase_set'].iloc[i]) / len(data_b['zhakar_dist_context_set'].iloc[i] | data_b['zhakar_dist_phrase_set'].iloc[i] | {'.'}) for i, _ in enumerate (data_b.index)]
    
    ones_data_b = data_b[data_b['target']==1]
    zeros_data_b = data_b[data_b['target']==0]
    
    words_ones = dict(Counter(' '.join(ones_data_b['phrase']).split()))
    words_zeros = dict(Counter(' '.join(zeros_data_b['phrase']).split()))
    
    
    
    for i in words_ones:
        words_ones[i] /= sum(words_ones.values())
    for i in words_zeros:
        words_zeros[i] /= sum(words_zeros.values())
        
    data_b['1_freq'] = 100 * data_b['phrase'].apply(lambda row : frequency_analysis(row, 1, words_ones, words_zeros))
    data_b['0_freq'] = 100 * data_b['phrase'].apply(lambda row : frequency_analysis(row, 0, words_ones, words_zeros))
    data_b['freq_analysis'] = data_b['1_freq'] -  data_b['0_freq'] 
    
    data = np.load('/home/jupyter-admin/Ivan/train_emb.npy')
    
    target = np.array(data[:,-1], dtype=int)
    data = data[:,:-1]

    contexts = data[:,:768]
    phrases = data[:, 768:]
    cos_dist = np.array([1 - contexts[i,:]@phrases[i,:].T / LA.norm(contexts[i]) / LA.norm(phrases[i]) for i, _ in enumerate(contexts)]).reshape(-1, 1)


    data_b['cosine_dist'] = cos_dist
    data_b['num_words_marusya'] = data_b['phrase'].apply(lambda row : int(str.lower(row).count('маруся') > 0))
    data_b['num_other_names'] = data_b['phrase'].apply(lambda row : int(str.lower(row).count('алиса') > 0))

    '''data = np.concatenate([data, data_b['cosine_dist'].values.reshape(-1,1), data_b['phrase_len'].values.reshape(-1,1), \
                           data_b['phrase_num_words'].values.reshape(-1,1),data_b['num_words_marusya'].values.reshape(-1,1), \
                           data_b['zhakar_dist'].values.reshape(-1,1)], axis=1)'''
    
    '''data = np.concatenate([data, data_b['cosine_dist'].values.reshape(-1,1), data_b['phrase_len'].values.reshape(-1,1), \
                           data_b['phrase_num_words'].values.reshape(-1,1),data_b['num_words_marusya'].values.reshape(-1,1), \
                           data_b['zhakar_dist'].values.reshape(-1,1), data_b['1_freq'].values.reshape(-1,1), \
                           data_b['0_freq'].values.reshape(-1,1)], axis=1)'''
 
    data = np.concatenate([data, data_b['phrase_len'].values.reshape(-1,1), \
                        data_b['num_words_marusya'].values.reshape(-1,1), \
                        data_b['zhakar_dist'].values.reshape(-1,1), data_b['freq_analysis'].values.reshape(-1,1)], axis=1)
    '''data = np.concatenate([data, data_b['phrase_len'].values.reshape(-1,1), \
                           data_b['phrase_num_words'].values.reshape(-1,1),data_b['num_words_marusya'].values.reshape(-1,1), \
                           data_b['zhakar_dist'].values.reshape(-1,1), data_b['freq_analysis'].values.reshape(-1,1)], axis=1)'''
    #data_b = data_b.dropna()
    
    with open('words_ones.json', 'w') as words_ones_file: 
        words_ones = json.dump(words_ones, words_ones_file)
    with open('words_zeros.json', 'w') as words_zeros_file: 
        words_zeros = json.dump(words_zeros, words_zeros_file)
    
    
    return data_b, data, target