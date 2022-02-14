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

import numpy as np
import pandas as pd
from sklearn import metrics, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from ..common import (
    abs_limit_10000,
    extract_features,
    process_stats,
    process_targets,
)
from ..lazy_resources import load_test, load_train

# %%
name_vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer='char')
sample_vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer='char')


# %%
X_train = load_train()
X_test = load_test()

X_train = X_train.sample(frac=1, random_state=100).reset_index(drop=True)

y_train = X_train[['y_act']].copy()
y_test = X_test[['y_act']].copy()


# %%
X_train_stats = process_stats(
    X_train,
    normalize=False,
    abs_limit=abs_limit_10000,
)
X_train = extract_features(
    X_train,
    X_train_stats,
    name_vectorizer=name_vectorizer,
    fit=True,
)
y_train = process_targets(y_train)

X_test_stats = process_stats(
    X_test,
    normalize=False,
    abs_limit=abs_limit_10000,
)
X_test = extract_features(
    X_test,
    X_test_stats,
    name_vectorizer=name_vectorizer,
    fit=False,
)
y_test = process_targets(y_test)


# %%
X_train_new = X_train.reset_index(drop=True).values
y_train_new = y_train.reset_index(drop=True).values

# %%
k = 5
kf = KFold(n_splits=k)
avg_train_acc, avg_test_acc = 0, 0

cvals = [0.1, 1, 10, 100, 1000]
gamma_vals = [0.0001, 0.001, 0.01, 0.1, 1, 10]

# cvals = [1]
# gamavals = [0.0001]

best_model = svm.SVC(
    C=100, decision_function_shape="ovo", gamma=0.001, cache_size=20000
)
best_score = 0


mean_scores, mean_train_scores, mean_holdout_scores = [], [], []
mean_best_score, mean_train_score, mean_holdout_score = 0, 0, 0

for train_index, test_index in kf.split(X_train_new):
    X_train_cur, X_test_cur = X_train_new[train_index], X_train_new[test_index]
    y_train_cur, y_test_cur = y_train_new[train_index], y_train_new[test_index]
    X_train_train, X_val, y_train_train, y_val = train_test_split(
        X_train_cur, y_train_cur, test_size=0.25, random_state=100
    )

    best_model = svm.SVC(
        C=100, decision_function_shape="ovo", gamma=0.001, cache_size=20000
    )
    best_score = 0
    for cval in cvals:
        for gamma in gamma_vals:
            clf = svm.SVC(
                C=cval,
                decision_function_shape="ovo",
                gamma=gamma,
                cache_size=20000,
            )
            clf.fit(X_train_train, y_train_train)
            sc = clf.score(X_val, y_val)

            if best_score < sc:
                best_score = sc
                best_model = clf
    #                 print(bestPerformingModel)

    best_train_score = best_model.score(X_train_cur, y_train_cur)
    best_score = best_model.score(X_test_cur, y_test_cur)
    best_score_holdout = best_model.score(X_test, y_test)

    mean_train_scores.append(best_train_score)
    mean_scores.append(best_score)
    mean_holdout_scores.append(best_score_holdout)

    mean_train_score = mean_train_score + best_train_score
    mean_best_score = mean_best_score + best_score
    mean_holdout_score = mean_holdout_score + best_score_holdout

    print(best_train_score)
    print(best_score)
    print(best_score_holdout)


# %%
print(mean_train_scores)
print(mean_scores)
print(mean_holdout_scores)

print(mean_train_score / k)
print(mean_best_score / k)
print(mean_holdout_score / k)

y_pred = best_model.predict(X_test)
best_score_holdout = best_model.score(X_test, y_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print('Confusion Matrix: Actual (Row) vs Predicted (Column)')
print(cnf_matrix)


# %%
