from pyspark.sql import DataFrame

_IRIS_COLUMNS = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width',
    'species',
]


def test_spark_works(iris_spark_df: DataFrame):
    assert set(iris_spark_df.columns) == set(_IRIS_COLUMNS)
    assert iris_spark_df.columns == _IRIS_COLUMNS
