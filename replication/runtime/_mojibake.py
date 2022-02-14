from __future__ import annotations

import string
from functools import reduce

import numpy as np
import pandas as pd

_str_cols = [
    'Attribute_name',
    'sample_1',
    'sample_2',
    'sample_3',
    'sample_4',
    'sample_5',
]
_TRANS_DELETE_ASCII_PRINTABLE = str.maketrans('', '', string.printable)


def find_bad_rows(df: pd.DataFrame) -> np.ndarray[np.bool_]:
    mask = np.zeros(len(df.index), dtype=np.bool_)
    for col in _str_cols:
        mask |= df[col].fillna('').str.translate(_TRANS_DELETE_ASCII_PRINTABLE).str.len().astype(bool).to_numpy()  # type: ignore
    return mask  # type: ignore


def drop_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = find_bad_rows(df).nonzero()[0]
    return df.drop(index=mask)  # type: ignore
