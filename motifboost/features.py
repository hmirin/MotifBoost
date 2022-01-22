import logging
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import List
from typing import TypeVar

from motifboost.repertoire import Repertoire

_logger = logging.getLogger(__name__)

T = TypeVar("T")


class FeatureExtractor(object):
    def fit(self, repertoires: List[Repertoire]):
        pass

    def transform(self, repertoires: List[Repertoire]) -> List[Any]:
        pass


class RepertoireWiseFeatureExtractor(FeatureExtractor, metaclass=ABCMeta):
    @abstractmethod
    def extract(self, r: Repertoire) -> T:
        pass

    @staticmethod
    @abstractmethod
    def cache_key() -> str:
        pass

    def transform(
        self, repertoires: List[Repertoire], use_cache=True, save_cache=True
    ) -> List[T]:
        results = []
        for r in repertoires:
            if not use_cache or self.cache_key() not in r.features:
                # cache is not available
                _logger.debug("cache miss!")
                result = self.extract(r)
                r.features[self.cache_key()] = result
            else:
                result = r.features[self.cache_key()]
            results.append(result)
        return results
