from pyspark.sql import DataFrame as SparkDataFrame

from heterodyne.features._sample import sample_with_select_distinct
from heterodyne.features._simple import (
    simple_features_impl,
    simple_features_melt,
)

_EXPECTED_FEATURES = 'total_vals,num_nans,%_nans,num_of_dist_val,%_dist_val,mean,std_dev,min_val,max_val'
_EXPECTED_FEATURES = _EXPECTED_FEATURES.split(',')


def test_simple_features(iris_spark_df: SparkDataFrame):
    cols = iris_spark_df.columns
    _expected_columns = {f'{l}::{r}' for l in cols for r in _EXPECTED_FEATURES}

    df = simple_features_impl(iris_spark_df, use_legacy_names=True)

    assert set(df.columns) == _expected_columns

    melted = simple_features_melt(df, use_legacy_names=True)

    assert set(melted.columns[1:]) == set(_EXPECTED_FEATURES)
    assert melted.columns[0] == 'Attribute_name'
    assert melted.columns[1:] == _EXPECTED_FEATURES


def test_sample_values_on_iris(iris_spark_df: SparkDataFrame):
    n = 3

    result = sample_with_select_distinct(iris_spark_df, n=n)
    assert result.columns == ['column', 'value']
    samples = result.count()
    n_columns = len(iris_spark_df.columns)
    assert samples == n_columns * n


def test_sample_values_on_iris_with_insufficient_uniques(
    iris_spark_df: SparkDataFrame,
):
    n = 5
    expected = 23

    result = sample_with_select_distinct(iris_spark_df, n=n)
    samples = result.count()
    assert samples == expected
