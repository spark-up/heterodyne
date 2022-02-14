# Copyright 2020 Vraj Shah, Arun Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import editdistance
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import DistanceMetric

from ..common import abs_limit_1000, process_stats, process_targets
from ..lazy_resources import load_test, load_train

# %%
X_train = load_train()
X_test = load_test()

X_train = X_train.sample(frac=1, random_state=100).reset_index(drop=True)

y_train = X_train[['y_act']].copy()
y_test = X_test[['y_act']].copy()

attr_train = X_train[['Attribute_name']].copy()
attr_test = X_test[['Attribute_name']].copy()


# %%
X_train = process_stats(X_train, normalize=True, abs_limit=abs_limit_1000)
y_train = process_targets(y_train)

X_test = process_stats(X_test, normalize=True, abs_limit=abs_limit_1000)
y_test = process_targets(y_test)

# %%
X_train.reset_index(inplace=True, drop=True)
y_train.reset_index(inplace=True, drop=True)
X_test.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)
attr_train.reset_index(inplace=True, drop=True)
attr_test.reset_index(inplace=True, drop=True)


X_train = X_train.values
y_train = y_train.values

X_test = X_test.values
y_test = y_test.values

attr_train = attr_train.values
attr_test = attr_test.values

# %%
K = 5
kf = KFold(n_splits=K)
avg_train_acc = avg_test_acc = 0

avgsc_lst, avgsc_train_lst, avgsc_hld_lst = [], [], []
avgsc = avgsc_train = avgsc_hld = 0

acc_val_lst, acc_test_lst = [], []

for train_index, test_index in kf.split(X_train):

    print(train_index)
    print()
    print(test_index)
    X_train_cur, X_test_cur = X_train[train_index], X_train[test_index]
    y_train_cur, y_test_cur = y_train[train_index], y_train[test_index]

    print(y_train_cur)
    atr_train_train, atr_val = attr_train[train_index], attr_train[test_index]

    X_train_train = X_train_cur
    X_val = X_test_cur

    y_train_train = y_train_cur
    y_val = y_test_cur

    Matrix = [[0 for x in range(len(X_train_train))] for y in range(len(X_val))]
    dist_euc = DistanceMetric.get_metric('euclidean')

    np_X_train = np.asmatrix(X_train_train)
    np_X_test = np.asmatrix(X_val)

    for i in range(len(X_val)):
        if i % 100 == 0:
            print(i)
        a = np_X_test[i]
        for j in range(len(X_train_train)):
            b = np_X_train[j]
            dist = np.linalg.norm(a - b)
            Matrix[i][j] = dist

    Matrix_ed = [
        [0 for x in range(len(X_train_train))] for y in range(len(X_val))
    ]

    for i in range(len(X_val)):
        if i % 100 == 0:
            print(i)
        a = atr_val[i]
        #         print(a)
        for j in range(len(X_train_train)):
            b = atr_train_train[j]
            #         print(b)
            dist = editdistance.eval(str(a), str(b))
            Matrix_ed[i][j] = dist

    Matrix_net = [
        [0 for x in range(len(X_train_train))] for y in range(len(X_val))
    ]
    alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    best_sc, best_alpha, best_neighbr = 0, 0, 0
    for alp in alpha:
        for i in range(len(Matrix)):
            for j in range(len(Matrix[i])):
                Matrix_net[i][j] = alp * Matrix[i][j] + Matrix_ed[i][j]

        for neighbr in range(1, 11):
            y_pred = []
            for i in range(len(X_val)):
                #     print('---')
                #         print(Matrix_net[i])
                dist = np.argsort(Matrix_net[i])[:neighbr]
                ys = []
                for x in dist:
                    ys.append(y_train_train[x])
                ho = stats.mode(ys)
                pred = ho[0][0]
                y_pred.append(pred)
            acc = accuracy_score(y_val, y_pred)
            print(str(neighbr) + '--->' + str(alp) + '--->' + str(acc))
            if acc > best_sc:
                best_sc = acc
                best_alpha = alp
                best_neighbr = neighbr

    print(best_sc, best_alpha, best_neighbr)

    ##################################
    X_train_train = X_train
    y_train_train = y_train
    atr_train_train = attr_train

    Matrix = [
        [0 for x in range(len(X_train_train))] for y in range(len(X_test))
    ]
    dist_euc = DistanceMetric.get_metric('euclidean')

    np_X_train = np.asmatrix(X_train_train)
    np_X_test = np.asmatrix(X_test)

    for i in range(len(X_test)):
        if i % 100 == 0:
            print(i)
        a = np_X_test[i]
        for j in range(len(X_train_train)):
            b = np_X_train[j]
            dist = np.linalg.norm(a - b)
            Matrix[i][j] = dist

    Matrix_ed = [
        [0 for x in range(len(X_train_train))] for y in range(len(X_test))
    ]

    for i in range(len(X_test)):
        if i % 100 == 0:
            print(i)
        a = attr_test[i]
        #         print(a)
        for j in range(len(X_train_train)):
            b = atr_train_train[j]
            #         print(b)
            dist = editdistance.eval(str(a), str(b))
            Matrix_ed[i][j] = dist

    #################################

    Matrix_net = [
        [0 for x in range(len(X_train_train))] for y in range(len(X_test))
    ]
    for i in range(len(Matrix)):
        for j in range(len(Matrix[i])):
            Matrix_net[i][j] = best_alpha * Matrix[i][j] + Matrix_ed[i][j]

    y_pred = []
    for i in range(len(X_test)):
        dist = np.argsort(Matrix_net[i])[:best_neighbr]
        ys = []
        for x in dist:
            ys.append(y_train_train[x])
        ho = stats.mode(ys)
        pred = ho[0][0]
        y_pred.append(pred)
    acc = accuracy_score(y_test, y_pred)
    print(acc)

    acc_val_lst.append(best_sc)
    acc_test_lst.append(acc)

    print(acc_val_lst)
    print(acc_test_lst)

    print('\n\n\n')


# %%
print(acc_val_lst)
print(acc_test_lst)
print(np.mean(acc_val_lst))
# print(np.mean(acc_test_lst))
bestValId = np.argmax(acc_val_lst)
print(acc_test_lst[bestValId])


# %%
