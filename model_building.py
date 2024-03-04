import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
import xgboost
import pickle
import os

def save_to_file(model, file_name):
    open_file = open("ensemble/" + file_name, "wb")
    pickle.dump(model, open_file)
    open_file.close()

L = 2
K = 5 # optimal parameters found in 'model_selection.ipynb'

file_dir = os.path.dirname(os.path.realpath('__file__'))

data = pd.read_csv(os.path.join(file_dir, 'data\\iris.csv'), delimiter=',', header=0)
data = data.drop('variety', axis=1)

#print(data)

distances = pdist(data.values, metric='euclidean')
dist_matrix = squareform(distances)

rep = []
for row in dist_matrix:
    mean_dist = np.mean(np.sort(row)[K])

    represent = 1 / (1 + mean_dist)

    rep.append(represent)

data['rep'] = rep

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

X_test['representativity'] = y_test
X_test.to_csv(os.path.join(file_dir, 'data\\test_data.csv'), index=False)

shuffled = X_train.sample(frac=1)
data_split = np.array_split(shuffled, L)

for i, split in enumerate(data_split) :
    distances = pdist(split.values, metric='euclidean')
    dist_matrix = squareform(distances)

    rep = []
    for row in dist_matrix:
        mean_dist = np.mean(np.sort(row)[K])

        represent = 1 / (1 + mean_dist)

        rep.append(represent)

    split['rep'] = rep

    xgb = xgboost.XGBRegressor()
    xgb.fit(split.iloc[:, :-1], split.iloc[:,-1])

    name = 'xgboost_' + str(i) + '.pkl'
    save_to_file(xgb, name)







