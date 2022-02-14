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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, train_test_split

from ..common import (
    abs_limit_10000,
    create_vectorizer,
    extract_features,
    process_stats,
    process_targets,
)
from ..lazy_resources import load_test, load_train

# %%
USE_STATS = True
USE_ATTRIBUTE_NAME = True

use_sample_1 = False
use_sample_2 = False

## Using descriptive stats and attribute name

# %%
X_train = load_train()
X_test = load_test()

X_train = X_train.sample(frac=1, random_state=100).reset_index(drop=True)

y_train = X_train[['y_act']].copy()
y_test = X_test[['y_act']].copy()

# %%
name_vectorizer = create_vectorizer()
sample_vectorizer = create_vectorizer()


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

K = 5
kf = KFold(n_splits=K, random_state=100)
avg_train_acc, avg_test_acc = 0, 0

n_estimators_grid = [5, 25, 50, 75, 100, 500]
max_depth_grid = [5, 10, 25, 50, 100, 250]

# n_estimators_grid = [25,50,75,100]
# max_depth_grid = [50,100]

avgsc_lst, avgsc_train_lst, avgsc_hld_lst = [], [], []
avgsc, avgsc_train, avgsc_hld = 0, 0, 0

best_param_count = {'n_estimator': {}, 'max_depth': {}}
best_model = None
best_n_estimators = best_max_depth = best_score = 0

i = 0
for train_index, test_index in kf.split(X_train_new):
    #     if i==1: break
    i = i + 1
    X_train_cur, X_test_cur = X_train_new[train_index], X_train_new[test_index]
    y_train_cur, y_test_cur = y_train_new[train_index], y_train_new[test_index]
    X_train_train, X_val, y_train_train, y_val = train_test_split(
        X_train_cur, y_train_cur, test_size=0.25, random_state=100
    )

    best_model = RandomForestClassifier(
        n_estimators=10, max_depth=5, random_state=100
    )
    best_score = 0
    print('=' * 10)
    for ne in n_estimators_grid:
        for md in max_depth_grid:
            clf = RandomForestClassifier(
                n_estimators=ne, max_depth=md, random_state=100
            )
            clf.fit(X_train_train, y_train_train.ravel())
            sc = clf.score(X_val, y_val)
            print(f"[n_estimator: {ne}, max_depth: {md}, accuracy: {sc}]")
            if best_score < sc:
                best_n_estimators = ne
                best_max_depth = md
                best_score = sc
                best_model = clf

    if best_n_estimators in best_param_count['n_estimator']:
        best_param_count['n_estimator'][best_n_estimators] += 1
    else:
        best_param_count['n_estimator'][best_n_estimators] = 1

    if best_max_depth in best_param_count['max_depth']:
        best_param_count['max_depth'][best_max_depth] += 1
    else:
        best_param_count['max_depth'][best_max_depth] = 1

    bscr_train = best_model.score(X_train_cur, y_train_cur)
    bscr = best_model.score(X_test_cur, y_test_cur)
    bscr_hld = best_model.score(X_test, y_test)

    avgsc_train_lst.append(bscr_train)
    avgsc_lst.append(bscr)
    avgsc_hld_lst.append(bscr_hld)

    avgsc_train = avgsc_train + bscr_train
    avgsc = avgsc + bscr
    avgsc_hld = avgsc_hld + bscr_hld

    print()
    print(
        f"> Best n_estimator: {best_n_estimators} || Best max_depth: {best_max_depth}"
    )
    print(f"> Best training score: {bscr_train}")
    print(f"> Best test score: {bscr}")
    print(f"> Best held score: {bscr_hld}")
print('=' * 10)

print(avgsc_train_lst)
print(avgsc_lst)
print(avgsc_hld_lst)

print(avgsc_train / K)
print(avgsc / K)
print(avgsc_hld / K)

y_pred = best_model.predict(X_test)
bscr_hld = best_model.score(X_test, y_test)
print(bscr_hld)


# %%
best_model.score(X_test, y_test)


# %%
