from __future__ import annotations

import pandas as pd

from heterodyne.features._sample import sample_features_from_values


def _take_5(s: pd.Series) -> pd.Series:
    u = s.unique()

    if len(u) >= 5:
        return pd.Series(u[:5])
    s = pd.Series(u)
    return s
    # return pd.concat([s, s.sample(5 - len(s))]).reset_index(drop=True)


def _preprocess_df(df: pd.DataFrame):
    sample_df: pd.DataFrame = (
        df.apply(_take_5, axis=0)
        .astype('string')
        .T.reset_index()
        .rename(columns=dict(index='column'))  # type: ignore
    )
    long_df: pd.DataFrame = sample_df.melt(
        id_vars=['column'],
        var_name='sample',
        value_name='value',
    )
    long_df = long_df.drop(columns='sample').dropna()
    return long_df


_EXPECTED_SAMPLE_FEATURES = 'sample_1,sample_2,sample_3,sample_4,sample_5,has_delimiters,has_url,has_email,has_date,mean_word_count,std_dev_word_count,mean_stopword_total,stdev_stopword_total,mean_char_count,stdev_char_count,mean_whitespace_count,stdev_whitespace_count,mean_delim_count,stdev_delim_count,is_list,is_long_sentence'.split(
    ','
)


def test_sample_features_pure(iris_df: pd.DataFrame):
    long_df = _preprocess_df(iris_df)
    assert long_df.columns.tolist() == ['column', 'value']
    print(long_df.to_markdown())

    result = sample_features_from_values(long_df, use_legacy_names=True)
    result_cols = result.columns.tolist()
    assert set(result_cols) == set(_EXPECTED_SAMPLE_FEATURES)
    assert result_cols == _EXPECTED_SAMPLE_FEATURES

    samples_df = result[_EXPECTED_SAMPLE_FEATURES[:5]]
    samples = samples_df.to_dict()
    notna = samples_df.notna().all().all()
    assert samples and notna
