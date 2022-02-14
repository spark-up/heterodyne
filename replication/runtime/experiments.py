from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Tuple

import pandas as pd

from .. import lib as replication_lib
from ..common import abs_limit_10000, extract_features, process_stats
from ..lazy_resources import (
    load_cnn,
    load_keras_name_tokenizer,
    load_keras_sample_tokenizer,
    load_logistic_regression,
    load_random_forest,
    load_sklearn_name_vectorizer,
    load_sklearn_sample_vectorizer,
    load_svm,
    load_test,
)
from ._mojibake import drop_bad_rows
from .measure import Experiment

if TYPE_CHECKING:
    import keras
    import keras.preprocessing.text
    import sklearn.base
    import sklearn.ensemble
    import sklearn.feature_extraction.text
    import sklearn.linear_model
    import sklearn.svm
    from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class _ZooExperiment(Experiment):
    # Placeholder Types
    model: Any = None
    raw_data: pd.DataFrame = field(
        default_factory=lambda: (
            drop_bad_rows(load_test().copy()).reset_index(drop=True)
        )
    )
    prepared_data: Optional[Any] = None

    def __post_init__(self):
        self.iterations = len(self.raw_data.index)

    def run(self):
        assert self.model and self.prepared_data is not None
        return self.model.predict(self.prepared_data)


@dataclass
class Cnn(_ZooExperiment):
    name = 'CNN'

    model: Optional[keras.Model] = None
    prepared_data: Optional[
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ] = None
    name_tokenizer: Optional[keras.preprocessing.text.Tokenizer] = None
    sample_tokenizer: Optional[keras.preprocessing.text.Tokenizer] = None

    def setup(self):
        self.model = load_cnn()
        # self.raw_data = load_test().copy()
        self.name_tokenizer = load_keras_name_tokenizer()
        self.sample_tokenizer = load_keras_sample_tokenizer()

    def prepare(self):
        from keras.preprocessing import sequence as keras_seq

        assert self.name_tokenizer and self.sample_tokenizer
        assert self.raw_data is not None

        # featurized = replication_lib.featurize_file(self.raw_data)
        featurized = self.raw_data
        structured_data = replication_lib.process_statistics(featurized)

        names = featurized['Attribute_name'].values.astype(str)
        samples = featurized['sample_1'].values.astype(str)

        X_names = keras_seq.pad_sequences(
            self.name_tokenizer.texts_to_sequences(names),
            maxlen=256,
        )
        # No, this isn't a bug.
        # The model will throw an error if the sample tokenizer is used
        X_samples = keras_seq.pad_sequences(
            self.name_tokenizer.texts_to_sequences(samples),
            maxlen=256,
        )
        self.prepared_data = X_names, X_samples, structured_data

    def run(self):
        assert self.model and self.prepared_data is not None
        return self.model.predict(self.prepared_data)


@dataclass
class Logistic(_ZooExperiment):
    name = 'Logistic'

    model: Optional[sklearn.linear_model.LogisticRegression] = None
    prepared_data: Optional[pd.DataFrame] = None
    name_vectorizer: Optional[CountVectorizer] = None
    sample_vectorizer: Optional[CountVectorizer] = None

    def setup(self):
        self.model = load_logistic_regression()
        # self.raw_data = load_test().copy()
        self.name_vectorizer = load_sklearn_name_vectorizer()
        self.sample_vectorizer = load_sklearn_sample_vectorizer()

    def prepare(self):
        assert self.raw_data is not None
        assert self.name_vectorizer and self.sample_vectorizer

        stats = process_stats(
            self.raw_data,
            normalize=True,
            abs_limit=abs_limit_10000,
        )
        self.prepared_data = extract_features(
            self.raw_data,
            stats,
            name_vectorizer=self.name_vectorizer,
            sample_vectorizer=self.sample_vectorizer,
            samples=2,
        )


@dataclass
class RandomForest(_ZooExperiment):
    name = 'Random Forest'

    model: Optional[sklearn.ensemble.RandomForestClassifier] = None
    prepared_data: Optional[pd.DataFrame] = None
    name_vectorizer: Optional[CountVectorizer] = None
    sample_vectorizer: Optional[CountVectorizer] = None

    def setup(self):
        self.model = load_random_forest()
        # self.raw_data = load_test().copy()
        self.name_vectorizer = load_sklearn_name_vectorizer()
        self.sample_vectorizer = load_sklearn_sample_vectorizer()

    def prepare(self):
        assert self.raw_data is not None
        assert self.name_vectorizer and self.sample_vectorizer

        stats = process_stats(
            self.raw_data,
            normalize=True,
            abs_limit=abs_limit_10000,
        )
        self.prepared_data = extract_features(
            self.raw_data,
            stats,
            name_vectorizer=self.name_vectorizer,
            # sample_vectorizer=self.sample_vectorizer,
            # samples=1,
        )


@dataclass
class Svm(_ZooExperiment):
    name = 'SVM'

    model: Optional[sklearn.svm.SVC] = None
    prepared_data: Optional[pd.DataFrame] = None
    name_vectorizer: Optional[CountVectorizer] = None
    sample_vectorizer: Optional[CountVectorizer] = None

    def setup(self):
        self.model = load_svm()
        # self.raw_data = load_test().copy()
        self.name_vectorizer = load_sklearn_name_vectorizer()
        self.sample_vectorizer = load_sklearn_sample_vectorizer()

    def prepare(self):
        assert self.raw_data is not None
        assert self.name_vectorizer and self.sample_vectorizer

        stats = process_stats(
            self.raw_data,
            normalize=True,
            abs_limit=abs_limit_10000,
        )
        self.prepared_data = extract_features(
            self.raw_data,
            stats,
            name_vectorizer=self.name_vectorizer,
            sample_vectorizer=self.sample_vectorizer,
            samples=1,
        )
