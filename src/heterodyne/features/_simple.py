from typing import Final

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from .util import ColumnFn, is_struct_field_numeric

def count_all_nan(c):
    return F.count((c == '' ) | \
                            c.isNull() | \
                            F.isnan(c) |
                            (c == None))

SIMPLE_FEATURES: dict[str, ColumnFn] = {
    'count': F.count,
    'distinct': F.count_distinct,
    'distinct_percent': lambda c: 100 * F.count_distinct(c) / F.count(c),

    'nans': lambda c: count_all_nan(c),
    'nans_percent': lambda c: 100 * count_all_nan(c) / F.count(c),
}


SIMPLE_NUMERIC_FEATURES: dict[str, ColumnFn] = {
    #'nans': lambda c: F.count(F.isnan(c) | c.isNull()), # added or isNull
    #'nans': lambda c: count_all_nan(c),
    #'nans_percent': lambda c: 100 * F.count(F.isnan(c)) / F.count(c),
    #'nans_percent': lambda c: 100 * count_all_nan(c) / F.count(c),
    'mean': F.mean,
    'std': F.stddev,
    'min': F.min,
    'max': F.max,
}

_LONG_FEATURES = list(SIMPLE_FEATURES.keys()) #+ ['nans']
_DOUBLE_FEATURES = list(SIMPLE_NUMERIC_FEATURES.keys())#[1:]

_FEATURES = list(SIMPLE_FEATURES.keys()) + list(SIMPLE_NUMERIC_FEATURES.keys())

# This map's iteration order determines output order as per spec
_LEGACY_NAME_MAP: dict[str, str] = {
    'name': 'Attribute_name',
    'count': 'total_vals',
    'nans': 'num_nans',
    'nans_percent': '%_nans',
    'distinct': 'num_of_dist_val',
    'distinct_percent': '%_dist_val',
    'mean': 'mean',
    'std': 'std_dev',
    'min': 'min_val',
    'max': 'max_val',
}

N_SAMPLES: Final = 5


def _create_schema() -> StructType:
    fields = [StructField('name', StringType(), False)]
    fields.extend(StructField(k, LongType(), False) for k in _LONG_FEATURES)
    fields.extend(StructField(k, DoubleType(), False) for k in _DOUBLE_FEATURES)
    return StructType(fields)


def simple_features_impl(
    df: SparkDataFrame,
    /,
    *,
    use_legacy_names=False,
) -> SparkDataFrame:
    cols = df.columns
    s_features = SIMPLE_FEATURES
    sn_features = SIMPLE_NUMERIC_FEATURES

    if use_legacy_names:
        s_features = {_LEGACY_NAME_MAP[k]: v for k, v in s_features.items()}
        sn_features = {_LEGACY_NAME_MAP[k]: v for k, v in sn_features.items()}

    simple_aggs = [
        fn(col(c)).alias(f'{c}::{name}')
        for c in cols
        for name, fn in s_features.items()
    ]

    numeric_aggs = [
        (fn(col(c)) if is_struct_field_numeric(df.schema[c]) else lit(0)).alias(
            f'{c}::{name}'
        )
        # (fn(col(c))).alias(
        #     f'{c}::{name}'
        # )
        for c in cols
        for name, fn in sn_features.items()
    ]

    agg_df = df.agg(*simple_aggs, *numeric_aggs)

    return agg_df


def simple_features_melt(
    df: SparkDataFrame,
    /,
    *,
    _cols: list[str] = None,
    use_legacy_names=False,
) -> SparkDataFrame:
    if not _cols:
        _cols = list({c.rsplit('::')[0] for c in df.columns})

    features = _FEATURES
    if use_legacy_names:
        features = [_LEGACY_NAME_MAP.get(f, f) for f in features]

    result = df.sql_ctx.createDataFrame([], schema=_create_schema())
    for c in _cols:
        exprs = [lit(c)]
        exprs.extend(col(f'{c}::{f}') for f in features)
        tmp = df.select(exprs)
        result = result.union(tmp)

    if use_legacy_names:
        expr = [col(k).alias(v) for k, v in _LEGACY_NAME_MAP.items()]
        result = result.select(expr)

    return result
