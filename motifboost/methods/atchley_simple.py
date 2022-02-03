# Taken from https://academic.oup.com/bioinformatics/article/30/22/3181/2390867

import collections
import functools
import multiprocessing
import random
from typing import List, Set

import numba
import numpy as np
from sklearn import svm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from tqdm import tqdm

from motifboost.features import FeatureExtractor
from motifboost.repertoire import Repertoire, repertoire_dataset_loader


aa = collections.defaultdict(int)
aa["A"] = 0
aa["C"] = 1
aa["D"] = 2
aa["E"] = 3
aa["F"] = 4
aa["G"] = 5
aa["H"] = 6
aa["I"] = 7
aa["K"] = 8
aa["L"] = 9
aa["M"] = 10
aa["N"] = 11
aa["P"] = 12
aa["Q"] = 13
aa["R"] = 14
aa["S"] = 15
aa["T"] = 16
aa["V"] = 17
aa["W"] = 18
aa["Y"] = 19

aas = sorted(aa.keys())


@numba.jit()
def atchley_factor(x: str) -> np.ndarray:
    aa = {}
    aa["A"] = 0
    aa["C"] = 1
    aa["D"] = 2
    aa["E"] = 3
    aa["F"] = 4
    aa["G"] = 5
    aa["H"] = 6
    aa["I"] = 7
    aa["K"] = 8
    aa["L"] = 9
    aa["M"] = 10
    aa["N"] = 11
    aa["P"] = 12
    aa["Q"] = 13
    aa["R"] = 14
    aa["S"] = 15
    aa["T"] = 16
    aa["V"] = 17
    aa["W"] = 18
    aa["Y"] = 19
    lookup = np.array(
        [
            [-0.591, -1.302, -0.733, 1.570, -0.146],
            [-1.343, 0.465, -0.862, -1.020, -0.255],
            [1.050, 0.302, -3.656, -0.259, -3.242],
            [1.357, -1.453, 1.477, 0.113, -0.837],
            [-1.006, -0.590, 1.891, -0.397, 0.412],
            [-0.384, 1.652, 1.330, 1.045, 2.064],
            [0.336, -0.417, -1.673, -1.474, -0.078],
            [-1.239, -0.547, 2.131, 0.393, 0.816],
            [1.831, -0.561, 0.533, -0.277, 1.648],
            [-1.019, -0.987, -1.505, 1.266, -0.912],
            [-0.663, -1.524, 2.219, -1.005, 1.212],
            [0.945, 0.828, 1.299, -0.169, 0.933],
            [0.189, 2.081, -1.628, 0.421, -1.392],
            [0.931, -0.179, -3.005, -0.503, -1.853],
            [1.538, -0.055, 1.502, 0.440, 2.897],
            [-0.228, 1.399, -4.760, 0.670, -2.647],
            [-0.032, 0.326, 2.213, 0.908, 1.313],
            [-1.337, -0.279, -0.544, 1.242, -1.262],
            [-0.595, 0.009, 0.672, 2.128, -0.184],
            [0.260, 0.830, 3.097, -0.838, 1.512,],
        ]
    )

    m = len(x)
    xsplit = list(x)
    xfactors = np.zeros(m * 5)
    for i in range(m):
        xfactors[5 * i : 5 * (i + 1)] = lookup[aa[xsplit[i]]]
    return xfactors


def repertoire_to_ngram(repertoire: Repertoire, ngram: int) -> Set[str]:
    ngram_set = set()
    for seq in repertoire.sequences.get_all():
        for q in range(len(seq) - ngram + 1):
            ngram_set.update([seq[q : q + ngram]])
    return ngram_set


# @numba.jit()
def seqs2historgam(
    size: int,
    n_gram: int,
    n_subsample: int,
    codewords_atchely: np.ndarray,
    cluster: np.ndarray,
    seqs: List[str],
):
    histogram = np.zeros(size)
    count = 0
    while count < n_subsample:
        pickseq = random.randint(0, len(seqs) - 1)
        m = len(seqs[pickseq])
        if m > n_gram:
            # start of p-mer must be located p steps from end
            picktriplet = random.randint(0, m - n_gram)
            x = seqs[pickseq][picktriplet : picktriplet + n_gram]
            v = atchley_factor(x)
            dist = (codewords_atchely - v) ** 2
            dist = np.sum(dist, axis=1)
            ind = np.argmin(dist)
            cluster[ind]
            histogram[cluster[ind]] += 1
            count += 1
    return histogram


# caching
atchley_factor("ADC")
seqs2historgam(
    1,
    3,
    100,
    np.zeros((1, 3 * 5), dtype=np.int64),
    np.zeros(1, dtype=np.int64),
    ["ADCA"],
)


def seqs2historgam_wrapper(
    something,
    size: int,
    n_gram: int,
    n_subsample: int,
    codewords_atchely: np.ndarray,
    cluster: np.ndarray,
    seqs: List[str],
):
    return seqs2historgam(size, n_gram, n_subsample, codewords_atchely, cluster, seqs)


def repertoire2vector(
    repertoire: Repertoire,
    codewords_atchely: np.ndarray,
    cluster: np.ndarray,
    n_gram:int,
    n_subsample: int,
    n_codewords: int,
    n_augmentation=100,
):
    wrapper = functools.partial(
        seqs2historgam_wrapper,
        size=n_codewords,
        n_gram=n_gram,
        n_subsample=n_subsample,
        codewords_atchely=codewords_atchely,
        cluster=cluster,
        seqs=repertoire.sequences.get_all(),
    )
    repertoire.sequences.get_all()
    with multiprocessing.Pool(10) as pool:
        imap = pool.imap(wrapper, range(n_augmentation))
        result = list(tqdm(imap, total=n_augmentation, desc="Augmentation"))
        return result


class AtchleySimpleEncoder(FeatureExtractor):
    def __init__(self, n_gram: int, n_subsample: int, n_codewords: int, n_augmentation: int = 100):
        self.codeword2cluster = None
        self.codewords_atchely = None
        self.n_gram = n_gram
        self.n_subsample = n_subsample
        self.n_codewords = n_codewords
        self.n_augmentation = n_augmentation

    def fit(self, repertoires: List[Repertoire]):
        # check codewords
        wrapper = functools.partial(repertoire_to_ngram, ngram=self.n_gram)
        with multiprocessing.Pool(10) as pool:
            imap = pool.imap(wrapper, repertoires)
            result = list(tqdm(imap, total=len(repertoires), desc="Ngram"))
        # result = [wrapper(repertoire) for repertoire in tqdm(repertoires,desc="Ngram")]
        codewords = set.union(*result, set())
        codewords_atchely = [atchley_factor(x) for x in codewords]
        self.codewords = codewords
        self.codewords_atchely = np.array(codewords_atchely)

        # codewords to cluster
        print("training KMeans...")
        kmeans = KMeans(n_clusters=self.n_codewords).fit(
            codewords_atchely
        )
        codeword2cluster = {w: l for w, l in zip(codewords, kmeans.labels_)}
        self.cluster = kmeans.labels_
        self.codeword2cluster = codeword2cluster

    def transform(self, repertoires: List[Repertoire]) -> List[List[np.ndarray]]:
        return [
            repertoire2vector(
                r, self.codewords_atchely, self.cluster, self.n_gram, self.n_subsample, self.n_codewords
            )
            for r in tqdm(repertoires, desc="Transforming")
        ]


class AtchleySimpleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,n_gram:int,n_subsample:int,n_codewords:int=100,n_augmentation:int=100):
        self.n_gram = n_gram
        self.n_subsample = n_subsample
        self.n_codewords = n_codewords
        self.n_augmentation = n_augmentation
        self.feature_extractor = AtchleySimpleEncoder(n_gram,n_subsample,n_codewords,n_augmentation)
        self.clf = svm.SVC(probability=True)

    def fit(self, repertoires: List[Repertoire], binary_targets: List[bool]):
        print("training extractor....")
        self.feature_extractor.fit(repertoires)
        augmented_features = self.feature_extractor.transform(repertoires)
        X = []
        y = []
        for feats, sig in zip(augmented_features, binary_targets):
            for feat in feats:
                X.append(feat)
                y.append(sig)
        print("training classifier....")
        self.clf.fit(np.array(X), np.array(y, dtype=np.int64))

    def predict(self, repertoires: List[Repertoire]) -> List[bool]:
        features = [
            np.mean(x, axis=0) for x in self.feature_extractor.transform(repertoires)
        ]
        pred_class = self.clf.predict_proba(np.array(features))[:, 1] > 0.5
        return pred_class

    def predict_proba(self, repertoires: List[Repertoire]) -> np.ndarray:
        print("converting data....")
        features = [
            np.mean(x, axis=0) for x in self.feature_extractor.transform(repertoires)
        ]
        pred_proba = self.clf.predict_proba(np.array(features))
        return pred_proba


# from dataclasses import dataclass
# from typing import Callable, Literal, Optional

# from motifboost.repertoire import Repertoire


# @dataclass()
# class DatasetSettings:
#     experiment_id: str
#     get_class: Callable[[Repertoire], bool]
#     filter_by_sample_id: Optional[Callable[[str], bool]]
#     filter_by_repertoire: Optional[Callable[[Repertoire], bool]]
#     get_split: Optional[
#         Callable[[Repertoire], Literal["train", "test", "other"]]
#     ] = None


# def huth_get_class(r: Repertoire) -> bool:
#     return bool(r.info["cmv"])


# def huth_filter_by_sample_id(sample_id: str) -> bool:
#     return "all" in sample_id


# huth_classification = DatasetSettings(
#     "Huth", huth_get_class, huth_filter_by_sample_id, None
# )


# def heather_get_class(r: Repertoire) -> bool:
#     return bool(r.info["HIV"])


# def heather_filter_by_sample_id_alpha(sample_id: str) -> bool:
#     return "alpha" in sample_id


# def heather_filter_by_sample_id_beta(sample_id: str) -> bool:
#     return "beta" in sample_id


# def heather_filter_by_repertoire(r: Repertoire) -> bool:
#     return not r.info["treated"]


# heather_classification_alpha = DatasetSettings(
#     "Heather",
#     heather_get_class,
#     heather_filter_by_sample_id_alpha,
#     heather_filter_by_repertoire,
# )

# heather_classification_beta = DatasetSettings(
#     "Heather",
#     heather_get_class,
#     heather_filter_by_sample_id_beta,
#     heather_filter_by_repertoire,
# )


# def emerson_get_class(r: Repertoire) -> bool:
#     return bool(r.info["CMV"])


# def emerson_cohort_get_split(r: Repertoire) -> Literal["train", "test", "other"]:
#     if "HIP" in r.sample_id:
#         return "train"
#     elif "Keck" in r.sample_id:
#         return "test"
#     else:
#         print("unknown sample_id:", r.sample_id)
#         return "other"


# emerson_classification_cohort_split = DatasetSettings(
#     "Emerson", emerson_get_class, None, None, emerson_cohort_get_split,
# )


# repertoires = repertoire_dataset_loader(
#     "./data/preprocessed/",
#     "Huth",
#     huth_classification.filter_by_sample_id,
#     huth_classification.filter_by_repertoire,
#     multiprocess_mode=True,
#     save_memory=True,  # for emerson full
# )

# sac = AtchleySimpleClassifier(n_gram=3,n_subsample=10000,n_codewords=100,n_augmentation=100)
# sac.fit(repertoires, [huth_classification.get_class(r) for r in repertoires])
# from IPython import embed

# embed()
