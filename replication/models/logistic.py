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


from typing import Optional

import numpy as np  # linear algebra
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split

from ..common import (
    abs_limit_10000,
    extract_features,
    process_stats,
    process_targets,
)
from ..lazy_resources import load_test, load_train

# %%
X_train = load_train()
X_test = load_test()


X_train = X_train.sample(frac=1, random_state=100).reset_index(drop=True)
print(len(X_train))

y_train = X_train[['y_act']]
y_test = X_test[['y_act']]

# %%
name_vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer='char')
sample_vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer='char')


# %%
X_train_stats = process_stats(
    X_train,
    normalize=True,
    abs_limit=abs_limit_10000,
)
y_train = process_targets(y_train)

X_test_stats = process_stats(
    X_test,
    normalize=True,
    abs_limit=abs_limit_10000,
)
y_test = process_targets(y_test)


X_train = extract_features(
    X_train,
    X_train_stats,
    name_vectorizer=name_vectorizer,
    sample_vectorizer=sample_vectorizer,
    samples=2,
    fit=True,
)
X_test = extract_features(
    X_test,
    X_test_stats,
    name_vectorizer=name_vectorizer,
    sample_vectorizer=sample_vectorizer,
    samples=2,
    fit=False,
)


X_train = X_train.reset_index(drop=True).values
y_train = y_train.reset_index(drop=True).values.ravel()


K = 5
kf = KFold(n_splits=K, random_state=100)
avg_train_acc, avg_test_acc = 0, 0

val_arr = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

avgsc_lst, avgsc_train_lst, avgsc_hld_lst = [], [], []
avgsc = avgsc_train = avgsc_hld = 0

best_param_count = {'cval': {}}

best_model: Optional[LogisticRegression] = None
best_score = best_cval = 0

for train_index, test_index in kf.split(X_train):
    X_train_cur, X_test_cur = X_train[train_index], X_train[test_index]
    y_train_cur, y_test_cur = y_train[train_index], y_train[test_index]
    X_train_train, X_val, y_train_train, y_val = train_test_split(
        X_train_cur, y_train_cur, test_size=0.25, random_state=100
    )

    best_model = LogisticRegression(
        penalty='l2', multi_class='multinomial', solver='lbfgs', C=1
    )
    best_score = 0
    print('=' * 10)
    for val in val_arr:
        clf = LogisticRegression(
            penalty='l2', multi_class='multinomial', solver='lbfgs', C=val
        )
        clf.fit(X_train_train, y_train_train)
        sc = clf.score(X_val, y_val)
        print(f"[C: {val}, accuracy: {sc}]")
        if best_score < sc:
            best_cval = val
            best_score = sc
            best_model = clf

    if best_cval in best_param_count['cval']:
        best_param_count['cval'][best_cval] += 1
    else:
        best_param_count['cval'][best_cval] = 1

    bscr_train = best_model.score(X_train_cur, y_train_cur)
    bscr = best_model.score(X_test_cur, y_test_cur)
    bscr_hld = best_model.score(X_test, y_test)

    avgsc_train_lst.append(bscr_train)
    avgsc_lst.append(bscr)
    avgsc_hld_lst.append(bscr_hld)

    avgsc_train = avgsc_train + bscr_train
    avgsc = avgsc + bscr
    avgsc_hld += bscr_hld
    print()
    print(f"> Best C: {best_cval}")
    print(f"> Best training score: {bscr_train}")
    print(f"> Best test score: {bscr}")
    print(f"> Best held score: {bscr_hld}")

print('=' * 10)


# %%
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
