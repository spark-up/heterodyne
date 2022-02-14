import os
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def cache_data_dir(cache: pytest.Cache) -> os.PathLike:
    return cache.makedir('data')


@pytest.fixture
def iris_df(cache_data_dir: Path) -> pd.DataFrame:
    path = cache_data_dir / 'iris.feather'

    if path.exists():
        return pd.read_feather(path)

    iris: pd.DataFrame = pd.read_csv(  # type: ignore
        'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
    )
    iris.to_feather(str(path))
    return iris


@pytest.fixture(autouse=True)
def nltk_data(cache_data_dir: Path):
    from heterodyne.features._data import fetch_data

    if nltk_data := os.environ.get('NLTK_DATA'):
        nltk_data = Path(nltk_data)
    else:
        nltk_data = cache_data_dir / 'nltk'

        os.environ['NLTK_DATA'] = str(nltk_data)
    fetch_data(nltk_data)
