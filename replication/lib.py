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


import re
from collections import Counter
from functools import cache, partial
from typing import Callable, Final, cast

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler

from common import abs_limit_10000 as abs_limit
from replication.constants import LEGACY_NAME_MAP
from replication.lazy_resources import (
    fetch_nltk_data,
    load_cnn,
    load_keras_name_tokenizer,
    load_keras_sample_tokenizer,
    load_logistic_regression,
    load_random_forest,
    load_sklearn_name_vectorizer,
    load_sklearn_sample_vectorizer,
    load_stopwords,
    load_svm,
    load_test,
    load_train,
)

IS_DELIMITED_RE = re.compile(r'[^,;\|]+([,;\|][^,;\|]+)+')
DELIMITER_RE = re.compile(r'(,|;|\|)')

URL_RE = re.compile(
    r'(http|ftp|https):\/\/'
    r'([\w_-]+(?:(?:\.[\w_-]+)+))'
    r'([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'
)

EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,6}\b')

STOPWORDS = set(load_stopwords())


def summary_stats(df, keys):
    b_data = []
    for col in keys:
        nans = np.count_nonzero(pd.isnull(df[col]))
        dist_val = len(pd.unique(df[col].dropna()))
        total_val = len(df[col])
        mean = 0
        std_dev = 0
        var = 0
        min_val = 0
        max_val = 0
        if is_numeric_dtype(df[col]):
            mean = np.mean(df[col])

            if not pd.isnull(mean):
                std_dev = np.std(df[col])
                var = np.var(df[col])
                min_val = float(np.min(df[col]))
                max_val = float(np.max(df[col]))
        b_data.append(
            [total_val, nans, dist_val, mean, std_dev, min_val, max_val]
        )
    return b_data


def get_sample(df, keys):
    rand = []
    for name in keys:
        unique = df[name].unique()
        rand_sample = list(unique)
        rand_sample = rand_sample[:5]
        if len(rand_sample) < 5:
            l = np.random.choice(rand_sample, 5 - len(rand_sample))
            rand_sample.extend(l)
        rand.append(rand_sample[:5])
    return rand


# summary_stat_result has a structure like [[Total_val, nans, dist_va, ...], ...].
def get_ratio_dist_val(summary_stat_result):
    ratio_dist_val = []
    for r in summary_stat_result:
        ratio_dist_val.append(r[2] * 100.0 / r[0])
    return ratio_dist_val


def get_ratio_nans(summary_stat_result):
    ratio_nans = []
    for r in summary_stat_result:
        ratio_nans.append(r[1] * 100.0 / r[0])
    return ratio_nans


def _stopword_count(s: str) -> int:
    fetch_nltk_data()
    counts = Counter(word_tokenize(s))
    return sum(v for k, v in counts.items() if k in load_stopwords())


def _fill_up(s: pd.Series, n: int):
    remaining = n - len(s)
    if remaining <= 0:
        return s.iloc[:n]

    l = [s]
    while remaining > len(s):
        l.append(s)
        remaining -= len(s)
    if remaining:
        l.append(s.head(remaining))
    return pd.concat(l)


def sample_features_from_values(
    df: pd.DataFrame,
    /,
    *,
    use_legacy_names=False,
    _n=5,
) -> pd.DataFrame:
    """
    Extracts features from tidy Pandas DataFrame.

    Expects columns `column`, `value`.
    """
    # First, we resample up
    df = (
        df.groupby('column', as_index=False)
        .apply(partial(_fill_up, n=_n))
        .reset_index(drop=True)
    )

    # COLUMN
    values: pd.Series[str] = cast('pd.Series[str]', df['value'])

    # NOTE: Bug in original means that delimiter_count will always equal whitespace_count
    df['is_delimited'] = values.str.match(IS_DELIMITED_RE)
    df['delimiter_count'] = values.str.count(DELIMITER_RE)
    df['word_count'] = values.str.split(' ', regex=False).map(len)  # type: ignore
    df['char_count'] = values.str.len()
    df['whitespace_count'] = values.str.count('')
    df['is_url'] = values.str.match(URL_RE)
    df['is_email'] = values.str.match(EMAIL_RE)
    df['is_datetime'] = pd.to_datetime(values, errors='coerce').notnull()

    df['stopword_count'] = values.map(_stopword_count)

    aggs: dict[str, str | Callable] = {'value': list}
    is_cols = [
        'is_delimited',
        'is_url',
        'is_email',
        'is_datetime',
    ]
    count_cols = [
        'delimiter_count',
        'word_count',
        'char_count',
        'whitespace_count',
        'stopword_count',
    ]
    cols = is_cols + count_cols
    aggs.update({col: 'sum' for col in cols})

    result = df.groupby('column').agg(aggs)
    values = result.value

    for col in is_cols:
        result[col] = result[col] >= 3  # type: ignore

    for col in count_cols:
        s = result[col]
        result[f'mean_{col}'] = s.mean()
        result[f'std_{col}'] = s.std()

    result = result.drop(columns=count_cols)

    result['is_list'] = result.is_delimited & (result.mean_char_count < 100)
    result['is_long_sentence'] = result.mean_word_count > 10

    for i in range(_n):
        result[f'sample_{i}'] = values.str.get(i)

    result: pd.DataFrame = cast(
        pd.DataFrame,
        result[list(LEGACY_NAME_MAP.keys())],
    )

    if use_legacy_names:
        result = result.rename(columns=LEGACY_NAME_MAP)

    return result


def featurize_file(df: pd.DataFrame):
    stats = []
    attribute_name = []
    sample = []
    i = 0

    ratio_dist_val = []
    ratio_nans = []

    columns = df.columns.tolist()

    attribute_name.extend(columns)
    summary_stat_result = summary_stats(df, columns)
    stats.extend(summary_stat_result)
    samples = get_sample(df, columns)
    sample.extend(samples)

    ratio_dist_val.extend(get_ratio_dist_val(summary_stat_result))
    ratio_nans.extend(get_ratio_nans(summary_stat_result))

    cols = [
        'Attribute_name',
        'total_vals',
        'num_nans',
        'num_of_dist_val',
        'mean',
        'std_dev',
        'min_val',
        'max_val',
        '%_dist_val',
        '%_nans',
        'sample_1',
        'sample_2',
        'sample_3',
        'sample_4',
        'sample_5',
    ]
    df = pd.DataFrame(columns=cols)

    for i in range(len(attribute_name)):
        val_append = []
        val_append.append(attribute_name[i])
        val_append.extend(stats[i])

        val_append.append(ratio_dist_val[i])
        val_append.append(ratio_nans[i])
        val_append.extend(sample[i])

        df.loc[i] = val_append  # type: ignore

    for row in df.itertuples():
        delim_cnt = url_cnt = email_cnt = date_cnt = 0
        chars_totals = []
        word_totals = []
        stopwords = []
        whitespaces = []
        delims_count = []

        for value in row[11:16]:
            word_totals.append(len(str(value).split(' ')))
            chars_totals.append(len(str(value)))
            whitespaces.append(str(value).count(' '))

            if IS_DELIMITED_RE.match(str(value)):
                delim_cnt += 1
            if URL_RE.match(str(value)):
                url_cnt += 1
            if EMAIL_RE.match(str(value)):
                email_cnt += 1

            delims_count.append(len(DELIMITER_RE.findall(str(value))))

            tokenized = word_tokenize(str(value))
            # print(tokenized)
            stopwords.append(len([w for w in tokenized if w in STOPWORDS]))

            try:
                _ = pd.Timestamp(value)
                date_cnt += 1
            except (ValueError, TypeError):
                date_cnt += 0

        # print(delim_cnt,url_cnt,email_cnt)
        if delim_cnt > 2:
            df.at[row.Index, 'has_delimiters'] = True
        else:
            df.at[row.Index, 'has_delimiters'] = False

        if url_cnt > 2:
            df.at[row.Index, 'has_url'] = True
        else:
            df.at[row.Index, 'has_url'] = False

        if email_cnt > 2:
            df.at[row.Index, 'has_email'] = True
        else:
            df.at[row.Index, 'has_email'] = False

        if date_cnt > 2:
            df.at[row.Index, 'has_date'] = True
        else:
            df.at[row.Index, 'has_date'] = False

        df.at[row.Index, 'mean_word_count'] = np.mean(word_totals)
        df.at[row.Index, 'std_dev_word_count'] = np.std(word_totals)

        df.at[row.Index, 'mean_stopword_total'] = np.mean(stopwords)
        df.at[row.Index, 'stdev_stopword_total'] = np.std(stopwords)

        df.at[row.Index, 'mean_char_count'] = np.mean(chars_totals)
        df.at[row.Index, 'stdev_char_count'] = np.std(chars_totals)

        df.at[row.Index, 'mean_whitespace_count'] = np.mean(whitespaces)
        df.at[row.Index, 'stdev_whitespace_count'] = np.std(whitespaces)

        df.at[row.Index, 'mean_delim_count'] = np.mean(delims_count)
        df.at[row.Index, 'stdev_delim_count'] = np.std(delims_count)

        if (
            df.at[row.Index, 'has_delimiters']
            and df.at[row.Index, 'mean_char_count'] < 100
        ):
            df.at[row.Index, 'is_list'] = True
        else:
            df.at[row.Index, 'is_list'] = False

        if df.at[row.Index, 'mean_word_count'] > 10:
            df.at[row.Index, 'is_long_sentence'] = True
        else:
            df.at[row.Index, 'is_long_sentence'] = False

    df = df

    return df


def extract_features(df, use_samples=False):
    df_orig = df.copy()
    df = df_orig[
        [
            'total_vals',
            'num_nans',
            '%_nans',
            'num_of_dist_val',
            '%_dist_val',
            'mean',
            'std_dev',
            'min_val',
            'max_val',
            'has_delimiters',
            'has_url',
            'has_email',
            'has_date',
            'mean_word_count',
            'std_dev_word_count',
            'mean_stopword_total',
            'stdev_stopword_total',
            'mean_char_count',
            'stdev_char_count',
            'mean_whitespace_count',
            'stdev_whitespace_count',
            'mean_delim_count',
            'stdev_delim_count',
            'is_list',
            'is_long_sentence',
        ]
    ]
    df = df.reset_index(drop=True).fillna(0)

    arr = df_orig['Attribute_name'].values.astype(str)

    X = load_sklearn_name_vectorizer().transform(arr)
    df_attr = pd.DataFrame(X.toarray())

    if not use_samples:
        return pd.concat([df, df_attr], axis=1, sort=False)

    sample_1_values = df_orig['sample_1'].values.astype(str)
    sample_2_values = df_orig['sample_2'].values.astype(str)
    X1 = load_sklearn_sample_vectorizer().transform(sample_1_values)
    X2 = load_sklearn_sample_vectorizer().transform(sample_2_values)

    sample_1_df = pd.DataFrame(X1.toarray())
    sample_2_df = pd.DataFrame(X2.toarray())
    return pd.concat(
        [df, df_attr, sample_1_df, sample_2_df], axis=1, sort=False
    )


def predict_rf(df: pd.DataFrame):
    pred = load_random_forest().predict(df).tolist()
    return pred


def process_statistics(df: pd.DataFrame):
    df = df[
        [
            '%_dist_val',
            '%_nans',
            'max_val',
            'mean',
            'mean_char_count',
            'mean_delim_count',
            'mean_stopword_total',
            'mean_whitespace_count',
            'mean_word_count',
            'min_val',
            'num_nans',
            'num_of_dist_val',
            'std_dev',
            'std_dev_word_count',
            'stdev_char_count',
            'stdev_delim_count',
            'stdev_stopword_total',
            'stdev_whitespace_count',
            'total_vals',
        ]
    ].rename(
        columns={
            '%_nans': 'scaled_perc_nans',
            'max_val': 'scaled_max',
            'mean': 'scaled_mean',
            'mean_char_count': 'scaled_mean_char_count',
            'mean_delim_count': 'scaled_mean_delim_count',
            'mean_stopword_total': 'scaled_mean_stopword_total',
            'mean_whitespace_count': 'scaled_mean_whitespace_count',
            'mean_word_count': 'scaled_mean_token_count',
            'min_val': 'scaled_min',
            'std_dev': 'scaled_std_dev',
            'std_dev_word_count': 'scaled_std_dev_token_count',
            'stdev_char_count': 'scaled_stdev_char_count',
            'stdev_delim_count': 'scaled_stdev_delim_count',
            'stdev_stopword_total': 'scaled_stdev_stopword_total',
            'stdev_whitespace_count': 'scaled_stdev_whitespace_count',
        }
    )

    df = df.reset_index(drop=True)
    df = df.fillna(0)

    cols_to_abs_limit = [
        'num_nans',
        'num_of_dist_val',
        'scaled_max',
        'scaled_mean',
        'scaled_min',
        'scaled_std_dev',
        'total_vals',
    ]
    for col in cols_to_abs_limit:
        df[col] = df[col].apply(abs_limit)

    cols_to_normalize = [
        'total_vals',
        'num_nans',
        'num_of_dist_val',
        'scaled_mean',
        'scaled_std_dev',
        'scaled_min',
        'scaled_max',
    ]
    X = df[cols_to_normalize].values
    X = np.nan_to_num(X)
    X_scaled = StandardScaler().fit_transform(X)
    df[cols_to_normalize] = pd.DataFrame(
        X_scaled,
        columns=cols_to_normalize,
        index=df.index,
    )

    return df


def predict_cnn(df):
    from keras.preprocessing import sequence as keras_seq

    cnn = load_cnn()

    featurized = featurize_file(df)
    structured_data_test = process_statistics(featurized)

    tokenizer = load_keras_name_tokenizer()
    tokenizer_sample = load_keras_sample_tokenizer()

    names = featurized['Attribute_name'].values.astype(str)
    samples = featurized['sample_1'].values.astype(str)

    X_names = keras_seq.pad_sequences(
        tokenizer.texts_to_sequences(names),
        maxlen=256,
    )
    X_samples = keras_seq.pad_sequences(
        tokenizer_sample.texts_to_sequences(samples),
        maxlen=256,
    )

    y_pred = cnn.predict([X_names, X_samples, structured_data_test])
    y_CNN = [np.argmax(i) for i in y_pred]
    return y_CNN
