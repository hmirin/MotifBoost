import functools
import logging
import multiprocessing
from multiprocessing import get_context

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
    from typing import Any, Final, List, Literal, Optional, Tuple
except:
    from typing import Any, List, Optional, Tuple
    from typing_extensions import Final, Literal

import lightgbm as lgb
import numba
import numpy as np
import optuna.integration.lightgbm as lgb_optuna
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from tqdm import tqdm

from motifboost.features import FeatureExtractor
from motifboost.repertoire import Repertoire, augment_repertoire
from motifboost.util import basic_void_mark, human_amino_acids

_logger = logging.getLogger(__name__)


class MotifFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        alphabets=None,
        void_mark=None,
        n_processes=None,
        count_weight_mode: bool = True,
        tfidf_mode: bool = False,
        ngram_range: Tuple[int, int] = (3, 4),
    ):
        self.alphabets: Final[Optional[List[str]]] = alphabets
        self.void_mark: Final[Optional[str]] = void_mark
        self.count_weight_mode = count_weight_mode
        self.tfidf_mode = tfidf_mode
        if n_processes is None:
            n_processes = int(multiprocessing.cpu_count() / 2)
        self.n_processes: int = n_processes
        self.idf = None
        self.ngram_range = ngram_range

    def fit(self, repertoires: List[Repertoire], use_cache=True, save_cache=True):
        wrapper = functools.partial(
            self._process_single, use_cache=use_cache, save_cache=save_cache, mode="fit"
        )
        with get_context("fork").Pool(self.n_processes) as pool:
            imap = pool.imap(wrapper, repertoires)
            result = list(
                tqdm(imap, total=len(repertoires), desc="MotifFeatureExtractor")
            )
            self.idf = len(result) / np.sum([arr > 0 for arr in result], axis=0)
            assert len(self.idf.shape) == 1  # TODO: delete me
            assert self.idf.shape[0] == result[0].shape[0]  # TODO: delete me
            return result

    def transform(
        self, repertoires: List[Repertoire], use_cache=True, save_cache=True
    ) -> List[Any]:
        if self.tfidf_mode:
            if self.idf is None:
                raise ValueError("call fit first for tfidf")
        wrapper = functools.partial(
            self._process_single,
            use_cache=use_cache,
            save_cache=save_cache,
            mode="transform",
        )
        with get_context("fork").Pool(self.n_processes) as pool:
            imap = pool.imap(wrapper, repertoires)
            return list(
                tqdm(imap, total=len(repertoires), desc="MotifFeatureExtractor")
            )

    def tfidf_transform(self, arr: np.ndarray) -> np.ndarray:
        arr = np.multiply(self.idf, arr)
        arr[np.isnan(arr)] = 0
        return arr

    def _process_single(
        self,
        repertoire: Repertoire,
        mode: Literal["fit", "transform"],
        use_cache=True,
        save_cache=True,
    ):
        if not use_cache or "motif_feature_ngram" not in repertoire.features:
            # cache is not available
            _logger.error("cache miss!")
            result = ngram_features(
                repertoire.sequences.get_all(),
                alphabets=self.alphabets,
                void_mark=self.void_mark,
                count_weights=repertoire.counts if self.count_weight_mode else None,
                ngram_range=self.ngram_range,
            )
            if save_cache:
                repertoire.features["motif_feature_ngram"] = result
        else:
            result = repertoire.features["motif_feature_ngram"]
        if mode == "transform" and self.tfidf_mode:
            result = self.tfidf_transform(result)
        return result


def trigram_features(
    seqs: List[str],
    alphabets: Optional[List[str]] = None,
    void_mark: Optional[str] = None,
    count_weights: Optional[List[int]] = None,
):  # int8 matrix value range: (0-20), shape :,32
    """
        Returns a feature vector, that each dimension represents a tri-mer.
        if count_weights is given, the feature is weighted by duplicate counts.
        If tfidf_weights is given, the features are converted by tfidf (a seq is seen as a doc)
    :param seqs:
    :param alphabets:
    :param void_mark:
    :param count_weights:
    :return:
    """
    if alphabets is None:
        alphabets = human_amino_acids
    if void_mark is None:
        void_mark = basic_void_mark
    if count_weights is None:
        count_weights = [1] * len(seqs)
    alphabets_with_void = [void_mark] + alphabets
    alphabet_size = len(alphabets_with_void)
    index2aa = dict(enumerate(alphabets_with_void))
    aa2index = {val: key for key, val in index2aa.items()}

    n = len(seqs)
    count_weights = np.array(count_weights, dtype=np.int64)
    seq_arrs = []
    for p in range(n):
        seq = [void_mark] + list(seqs[p]) + [void_mark]
        seq_arrs.append(np.array([aa2index[s] for s in seq]))
    return trigram(seq_arrs, alphabet_size, count_weights)


def ngram_features(
    seqs: List[str],
    alphabets: Optional[List[str]] = None,
    void_mark: Optional[str] = None,
    count_weights: Optional[List[int]] = None,
    ngram_range: Tuple[int, int] = (3, 4),
):  # int8 matrix value range: (0-20), shape :,32
    """
        Returns a feature vector, that each dimension represents a tri-mer.
        if count_weights is given, the feature is weighted by duplicate counts.
        If tfidf_weights is given, the features are converted by tfidf (a seq is seen as a doc)
    :param seqs:
    :param alphabets:
    :param void_mark:
    :param count_weights:
    :return:
    """
    if alphabets is None:
        alphabets = human_amino_acids
    if void_mark is None:
        void_mark = basic_void_mark
    if count_weights is None:
        count_weights = [1] * len(seqs)
    alphabets_with_void = [void_mark] + alphabets
    alphabet_size = len(alphabets_with_void)
    index2aa = dict(enumerate(alphabets_with_void))
    aa2index = {val: key for key, val in index2aa.items()}

    n = len(seqs)
    count_weights = np.array(count_weights, dtype=np.int64)
    seq_arrs = []
    for p in range(n):
        seq = [void_mark] + list(seqs[p]) + [void_mark]
        seq_arrs.append(np.array([aa2index[s] for s in seq]))
    return np.concatenate(
        [
            ngram(seq_arrs, alphabet_size, count_weights, n_gram)
            for n_gram in range(ngram_range[0], ngram_range[1])
        ]
    )


@numba.jit(nopython=True)
def ngram(
    seq_arrs: List[np.array], alphabet_size: int, count_weights: np.array, n_gram: int
):
    n = len(seq_arrs)
    arrays = np.zeros((alphabet_size**n_gram), dtype=np.int64)
    multiplier = np.array([alphabet_size ** (n_gram - 1 - k) for k in range(n_gram)])
    for p in range(n):
        seq_arr = seq_arrs[p]
        for q in range(0, len(seq_arr) - (n_gram - 1)):
            idx = 0
            for k in range(n_gram):
                idx += seq_arr[q + k] * multiplier[k]
            arrays[idx] += count_weights[p]
    # np.ndarray.sum returns int
    arrays = arrays / arrays.sum()
    return arrays


@numba.jit(nopython=True)
def trigram(seq_arrs: List[np.array], alphabet_size: int, count_weights: np.array):
    n = len(seq_arrs)
    arrays = np.zeros((alphabet_size**3), dtype=np.int64)
    for p in range(n):
        seq_arr = seq_arrs[p]
        for q in range(0, len(seq_arr) - 2):
            arrays[
                seq_arr[q] * alphabet_size**2
                + seq_arr[q + 1] * alphabet_size**1
                + seq_arr[q + 2] * alphabet_size**0,
            ] += (
                1 * count_weights[p]
            )
    # np.ndarray.sum returns int
    arrays = arrays / arrays.sum()
    return arrays


class MotifBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        alphabets=None,
        void_mark=None,
        count_weight_mode=True,
        tfidf_mode=False,
        augmentation_times: Optional[int] = 5,
        augmentation_rate: Optional[float] = 0.5,
        ngram_range: Optional[Tuple[int, int]] = (3, 4),
        classifier_method: Literal["optuna-lightgbm"] = "optuna-lightgbm",
        n_jobs: Optional[int] = None,
    ):
        self.classifier_method = classifier_method
        # lightGBM w/ Optuna
        if self.classifier_method in ["optuna-lightgbm", "lightgbm"]:
            self.clf = None
        elif self.classifier_method == "linear_regression":
            self.clf = LogisticRegression()
        elif self.classifier_method == "svm":
            self.clf = SVC(probability=True)
        else:
            raise ValueError("No such clssifiermethod is available:", classifier_method)

        if n_jobs:
            self.n_jobs = n_jobs
        else:
            if platform.machine() == "arm64":
                self.n_jobs = multiprocessing.cpu_count()
            else:
                self.n_jobs = int(multiprocessing.cpu_count() / 2)

        self.feature_extractor = MotifFeatureExtractor(
            alphabets=alphabets,
            void_mark=void_mark,
            n_processes=self.n_jobs,
            count_weight_mode=count_weight_mode,
            tfidf_mode=tfidf_mode,
            ngram_range=ngram_range,
        )
        self.augmentation = augmentation_times
        self.augmentation_rate = augmentation_rate

    def fit(self, repertoires: List[Repertoire], binary_targets: List[bool]):
        print("converting data....")
        if self.augmentation:
            print("augmentating...")
            # caching classification_result
            for r, tgt in zip(repertoires, binary_targets):
                r.info["binary_target"] = tgt
            repertoires = augment_repertoire(
                repertoires,
                n=len(repertoires) * (self.augmentation),
                subsample_size=self.augmentation_rate,
            )
            binary_targets = [r.info["binary_target"] for r in repertoires]
            print("augmted samples length", len(binary_targets))
        trigram_arrays = self.feature_extractor.fit(repertoires)
        print("fitting....")
        if self.classifier_method in ["optuna-lightgbm", "lightgbm"]:
            dtrain = lgb_optuna.Dataset(
                np.array(trigram_arrays), label=np.array(binary_targets, dtype=np.int64)
            )
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "num_threads": self.n_jobs,
            }
            if self.classifier_method == "optuna-lightgbm":
                tuner = lgb_optuna.LightGBMTunerCV(
                    params,
                    dtrain,
                    verbose_eval=100,
                    early_stopping_rounds=100,
                    folds=KFold(n_splits=3),
                    return_cvbooster=True,
                    optuna_seed=0,
                )

                tuner.run()

                print("Best score:", tuner.best_score)
                best_params = tuner.best_params
                print("Best params:", best_params)
                print("  Params: ")
                for key, value in best_params.items():
                    print("    {}: {}".format(key, value))
                self.clf = tuner.get_best_booster()
            else:
                self.clf = lgb.LGBMClassifier(**params)
                self.clf.fit(
                    np.array(trigram_arrays), np.array(binary_targets, dtype=np.int64)
                )
        else:
            self.clf.fit(np.array(trigram_arrays), np.array(binary_targets))

    def predict(self, repertoires: List[Repertoire]) -> List[bool]:
        print("converting data....")
        trigram_arrays = self.feature_extractor.transform(repertoires)
        # pred_class = self.clf.predict(trigram_arrays)
        # return [x == "True" for x in pred_class]
        if self.classifier_method == "optuna-lightgbm":
            # simple averaging
            preds = np.array(
                self.clf.predict(
                    np.array(trigram_arrays), num_iteration=self.clf.best_iteration
                )
            )
            means = np.mean(preds, axis=0)
            return [x > 0.5 for x in means]
        else:
            self.predict(np.array(trigram_arrays))

    def predict_proba(self, repertoires: List[Repertoire]) -> np.ndarray:
        print("converting data....")
        trigram_arrays = self.feature_extractor.transform(repertoires)
        # return self.clf.predict_proba(trigram_arrays)
        if self.classifier_method == "optuna-lightgbm":
            # simple averaging
            preds = np.array(
                self.clf.predict(
                    np.array(trigram_arrays), num_iteration=self.clf.best_iteration
                )
            )
            means = np.mean(preds, axis=0)
            return np.array([1 - means, means]).transpose()
        else:
            return self.clf.predict_proba(np.array(trigram_arrays))
