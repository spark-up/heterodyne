# The iteration order of this map must correspond with the legacy output
from typing import Final

__all__ = ['LEGACY_NAME_MAP']

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
