from functools import cache
from os import PathLike
from tempfile import mkdtemp

from nltk.corpus import stopwords as _stopwords
from nltk.corpus.util import LazyCorpusLoader
from nltk.downloader import Downloader
from nltk.downloader import download as _nltk_download

_downloader: Downloader | None = None


def fetch_data(path: PathLike = None) -> None:
    if not path:
        path = mkdtemp()  # type: ignore

    global _downloader

    if not _downloader:
        _downloader = Downloader(download_dir=str(path))

    if _downloader.status('stopwords') != 'INSTALLED':
        _downloader.download('stopwords', quiet=True)
    if _downloader.status('punkt') != 'INSTALLED':
        _downloader.download('punkt', quiet=True)


@cache
def stopwords() -> frozenset[str]:
    fetch_data()
    return frozenset(_stopwords.words('english'))
