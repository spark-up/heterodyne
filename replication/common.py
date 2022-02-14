from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

DICT_LABELS = {
    'numeric': 0,
    'categorical': 1,
    'datetime': 2,
    'sentence': 3,
    'url': 4,
    'embedded-number': 5,
    'list': 6,
    'not-generalizable': 7,
    'context-specific': 8,
}


class _NoopEstimator(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return y


_NOOP_ESTIMATOR = _NoopEstimator()

_Numeric = TypeVar('_Numeric', np.ndarray, pd.Series, int, float)


def abs_limit_1000(x: _Numeric) -> _Numeric:
    if isinstance(x, (pd.Series, np.ndarray)):
        if x.dtype.kind == 'b':  # type: ignore
            return x
        if x.dtype.kind == 'u':  # type: ignore
            return np.clip(x, 0, 1000)  # type: ignore
    return np.clip(x, -1000, 1000)  # type: ignore


def abs_limit_10000(x: _Numeric) -> _Numeric:
    if isinstance(x, (pd.Series, np.ndarray)):
        if x.dtype.kind == 'b':  # type: ignore
            return x
        if x.dtype.kind == 'u':  # type: ignore
            return np.clip(x, 0, 10000)  # type: ignore
    return np.clip(x, -10000, 10000)  # type: ignore


def to_string_list(it: Any) -> List[str]:
    return list(map(str, it))


def process_stats(
    df: pd.DataFrame,
    *,
    normalize,
    abs_limit: Callable = lambda x: x,
) -> pd.DataFrame:
    df = df[
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
    if normalize:
        df = normalize_data(df, abs_limit=abs_limit)

    return df


def process_targets(y: pd.DataFrame) -> pd.DataFrame:
    y['y_act'] = y['y_act'].map(DICT_LABELS).astype(float)
    return y


def normalize_data(df: pd.DataFrame, abs_limit: Callable):
    df = cast(pd.DataFrame, df.apply(abs_limit))

    X = np.nan_to_num(df.values)
    X_scaled = StandardScaler().fit_transform(X)
    return pd.DataFrame(X_scaled, columns=df.columns, index=df.index)


def create_vectorizer() -> CountVectorizer:
    return CountVectorizer(ngram_range=(2, 2), analyzer='char')


@overload
def extract_features(
    df: pd.DataFrame,
    df_stats: pd.DataFrame,
    /,
    *,
    name_vectorizer: CountVectorizer,
    fit=False,
) -> pd.DataFrame:
    ...


@overload
def extract_features(
    df: pd.DataFrame,
    df_stats: pd.DataFrame,
    /,
    *,
    name_vectorizer: CountVectorizer,
    sample_vectorizer: CountVectorizer,
    samples: Literal[0, 1, 2] = 0,
    fit=False,
) -> pd.DataFrame:
    ...


def extract_features(
    df: pd.DataFrame,
    df_stats: pd.DataFrame,
    /,
    *,
    name_vectorizer: CountVectorizer,
    sample_vectorizer: Union[
        CountVectorizer,
        _NoopEstimator,
    ] = _NOOP_ESTIMATOR,
    samples=0,
    fit=False,
) -> pd.DataFrame:
    """
    Create a final featurized DataFrame with statistics and vectorized strings.

    Args:
        df: The dataframe to vectorize
        df_stats: The statistical features to use
        vectorizer_name:
    """
    df = df.copy()

    names = to_string_list(df['Attribute_name'].values)

    use_sample_1 = samples >= 1
    use_sample_2 = samples >= 2
    list_sample_1 = df['sample_1'].astype(str) if use_sample_1 else None
    list_sample_2 = df['sample_2'].astype(str) if use_sample_2 else None
    # list_sample_3 = df['sample_3'].astype(str) if samples >= 3 else None

    X1 = X2 = None
    if fit:
        X = name_vectorizer.fit_transform(names)
        if use_sample_1:
            X1 = sample_vectorizer.fit_transform(list_sample_1)
        if use_sample_2:
            X2 = sample_vectorizer.transform(list_sample_2)

    else:
        X = name_vectorizer.transform(names)
        if use_sample_1:
            X1 = sample_vectorizer.transform(list_sample_1)
        if use_sample_2:
            X2 = sample_vectorizer.transform(list_sample_2)

    attr_df = pd.DataFrame(X.toarray())
    sample1_df = pd.DataFrame(X1.toarray()) if use_sample_1 else None  # type: ignore
    sample2_df = pd.DataFrame(X2.toarray()) if use_sample_2 else None  # type: ignore

    out = [df_stats, attr_df]
    if sample1_df is not None:
        out.append(sample1_df)
    if sample2_df is not None:
        out.append(sample2_df)

    return pd.concat(out, axis=1, sort=False)
