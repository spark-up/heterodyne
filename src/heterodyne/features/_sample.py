import re
from collections import Counter
from functools import partial
from typing import Callable, Final, cast

import pandas as pd
from nltk.tokenize import word_tokenize
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StringType, StructField, StructType

from ._data import fetch_data, stopwords

IS_DELIMITED_RE = re.compile(r'[^,;\|]+([,;\|][^,;\|]+)+')
DELIMITER_RE = re.compile(r'(,|;|\|)')

URL_RE = re.compile(
    r'(http|ftp|https):\/\/'
    r'([\w_-]+(?:(?:\.[\w_-]+)+))'
    r'([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'
)

EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,6}\b')

# The iteration order of this map must correspond with the legacy output
_LEGACY_NAME_MAP = {
    'sample_0': 'sample_1',
    'sample_1': 'sample_2',
    'sample_2': 'sample_3',
    'sample_3': 'sample_4',
    'sample_4': 'sample_5',
    'is_delimited': 'has_delimiters',
    'is_url': 'has_url',
    'is_email': 'has_email',
    'is_datetime': 'has_date',
    'word_count': 'word_count',
    'stopword_count': 'stopword_total',
    'char_count': 'char_count',
    'whitespace_count': 'whitespace_count',
    'delimiter_count': 'delim_count',
    'is_list': 'is_list',
    'is_long_sentence': 'is_long_sentence',
}

LEGACY_NAME_MAP: Final = {}

for k, v in _LEGACY_NAME_MAP.items():
    if k.endswith('count'):
        LEGACY_NAME_MAP[f'mean_{k}'] = f'mean_{v}'
        LEGACY_NAME_MAP[f'std_{k}'] = f'stdev_{v}'
    else:
        LEGACY_NAME_MAP[k] = v

LEGACY_NAME_MAP['std_word_count'] = 'std_dev_word_count'


def _sample_values_schema() -> StructType:
    return StructType(
        [
            StructField('column', StringType(), False),
            StructField('value', StringType(), False),
        ]
    )


def _stopword_count(s: str) -> int:
    fetch_data()
    counts = Counter(word_tokenize(s))
    return sum(v for k, v in counts.items() if k in stopwords())


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


# TODO: Reimplement in PySpark
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
    df['word_count'] = values.str.split(' ', regex=False).map(len, na_action='ignore')  # type: ignore, na_action added
    df['char_count'] = values.str.len()
    df['whitespace_count'] = values.str.count(' ')
    df['is_url'] = values.str.match(URL_RE)
    df['is_email'] = values.str.match(EMAIL_RE)
    df['is_datetime'] = pd.to_datetime(values, errors='coerce').notnull()

    df['stopword_count'] = values.map(_stopword_count, na_action='ignore') #na_action added

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


def sample_with_select_distinct(
    df: SparkDataFrame,
    n: int = 5,
) -> SparkDataFrame:
    cols = df.columns

    result = df.sql_ctx.createDataFrame([], schema=_sample_values_schema())
    for name in cols:
        expr = [
            lit(name),
            col(name).cast('string'),
        ]

        result = result.union(df.select(expr).distinct().limit(n))

    return result
