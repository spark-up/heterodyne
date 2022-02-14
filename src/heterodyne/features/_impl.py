from typing import cast

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame

from ._sample import sample_features_from_values, sample_with_select_distinct
from ._simple import simple_features_impl, simple_features_melt


def extract_features(
    df: SparkDataFrame,
    /,
    *,
    use_legacy_names=False,
) -> pd.DataFrame:
    """
    Non-deterministically extract SortingHat features from a Spark DataFrame.

    This routine aims for full bug-for-bug compatibility with the implementation
    used by [Shah21]_. See Warnings for known issues and caveats.

    .. [Shah21] Vraj Shah, Jonathan Lacanlale, Premanand Kumar, Kevin Yang, and Arun Kumar. 2021.
    Towards Benchmarking Feature Type Inference for AutoML Platforms.
    *In Proceedings of the 2021 International Conference on Management of Data (SIGMOD '21),
    June 20-25, 2021, Virtual Event, China.* ACM, New York, NY, USA, 13 pages.
    https://doi.org/10.1145/3448016.3457274

    Currently, this routine's IO is estimated to be two full scans.

    For best results, read files with schema inference.

    Parameters
    ----------
    df: pyspark.sql.DataFrame
        The input Spark DataFrame.
    use_legacy_names: bool, default: False
        If True, return output with column names compatible with train / test
        data from the SortingHat Model Prep Zoo.

    Warnings
    --------
    - `{mean/std}_delimiter_count` will always be equal to `{mean/std}_whitespace count`
    - `{mean/std}_whitespace_count` only accounts for ` `, i.e. U+0020 SPACE.
    - `is_{url/email}` (legacy name `has_{url/email}`) only matches at the start of the string,
        and only match a subset of all URLs/Emails.
    """
    simple_sdf = simple_features_melt(
        simple_features_impl(df, use_legacy_names=use_legacy_names),
        use_legacy_names=use_legacy_names,
    )
    sample_values_sdf = sample_with_select_distinct(df, 5)
    simple_df = cast(pd.DataFrame, simple_sdf.toPandas())
    sample_values_df = cast(pd.DataFrame, sample_values_sdf.toPandas())
    sample_df = sample_features_from_values(
        sample_values_df,
        use_legacy_names=use_legacy_names,
    )
    return pd.concat([simple_df, sample_df], axis='columns')
