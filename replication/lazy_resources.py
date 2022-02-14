"""
Load read-only resources.

The resources are not guaranteed to be immutable or even independent across
calls. Although the return values may be mutable, doing so violates API safety.
"""


from __future__ import annotations

import pickle
import warnings
from functools import cache, wraps
from os import PathLike
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Any, Callable, Dict, TypeVar

import pandas as pd
from nltk.corpus import stopwords as _stopwords
from nltk.corpus.util import LazyCorpusLoader
from nltk.downloader import Downloader
from nltk.downloader import download as _nltk_download

if TYPE_CHECKING:
    import keras
    import keras.preprocessing.text
    import sklearn.ensemble
    import sklearn.feature_extraction.text
    import sklearn.linear_model
    import sklearn.svm


_T = TypeVar('_T')

_CACHE: Dict[Callable[[], Any], Any] = {}

ROOT = Path(__file__).parent.parent
ZOO_RESOURCES = ROOT / 'data' / 'zoo-resources'
ZOO_DICTIONARIES = ZOO_RESOURCES / 'Dictionary'
ZOO_BENCHMARK_DATA = ROOT / 'data' / 'zoo-benchmark'


warnings.filterwarnings(
    'ignore',
    message='Trying to unpickle estimator',
    category=UserWarning,
    module='sklearn',
)


def load_train():
    return pd.read_csv(ZOO_BENCHMARK_DATA / 'data_train.csv')


def load_test():
    return pd.read_csv(ZOO_BENCHMARK_DATA / 'data_test.csv')


def load_cnn() -> keras.models.Functional:
    import keras

    return keras.models.load_model(ZOO_RESOURCES / 'CNN.h5')


def load_random_forest() -> sklearn.ensemble.RandomForestClassifier:
    with (ZOO_RESOURCES / 'RandomForest.pkl').open('rb') as f:
        return pickle.load(f)


def load_logistic_regression() -> sklearn.linear_model.LogisticRegression:
    with open(ZOO_RESOURCES / 'LogReg.pkl', 'rb') as f:
        return pickle.load(f)


def load_svm() -> sklearn.svm.SVC:
    with open(ZOO_RESOURCES / 'SVM.pkl', 'rb') as f:
        return pickle.load(f)


def load_sklearn_name_vectorizer() -> sklearn.feature_extraction.text.CountVectorizer:
    with open(ZOO_DICTIONARIES / 'dictionaryName.pkl', 'rb') as f:
        return pickle.load(f)


def load_sklearn_sample_vectorizer() -> sklearn.feature_extraction.text.CountVectorizer:
    with open(ZOO_DICTIONARIES / 'dictionarySample.pkl', 'rb') as f:
        return pickle.load(f)


def load_keras_name_tokenizer() -> keras.preprocessing.text.Tokenizer:
    with open(ZOO_DICTIONARIES / 'keras_dictionaryName.pkl', 'rb') as f:
        return pickle.load(f)


def load_keras_sample_tokenizer() -> keras.preprocessing.text.Tokenizer:
    with open(ZOO_DICTIONARIES / 'keras_dictionarySample.pkl', 'rb') as f:
        return pickle.load(f)


def _get_downloader(path: PathLike = None, /) -> Downloader:
    return Downloader(download_dir=str(path))


def fetch_nltk_data(path: PathLike = None) -> None:
    if not path:
        path = mkdtemp()  # type: ignore

    dl = _get_downloader(path)

    if dl.status('stopwords') != 'INSTALLED':
        dl.download('stopwords', quiet=True)
    if dl.status('punkt') != 'INSTALLED':
        dl.download('punkt', quiet=True)


def load_stopwords() -> frozenset[str]:
    fetch_nltk_data()
    return frozenset(_stopwords.words('english'))


def _cache(fn: Callable[[], _T]) -> Callable[[], _T]:
    @wraps(fn)
    def cached() -> _T:
        if fn in _CACHE:
            return _CACHE[fn]
        res = fn()
        _CACHE[fn] = res
        return res

    return cached


for name, fn in dict(globals()).items():
    if name.startswith('load_') or name.startswith('_get_') and callable(fn):
        globals()[name] = _cache(fn)


# Named this way to avoid getting cached, and to trivially eliminate accidental
# infinite recursion.
def force_load():
    """
    Force load all pickled models and dictionaries into memory.
    """
    for name, fn in dict(globals()).items():
        if name.startswith('load_') and callable(fn):
            fn()
