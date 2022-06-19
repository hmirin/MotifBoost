# Taken from https://academic.oup.com/bioinformatics/article/30/22/3181/2390867

import functools
import multiprocessing
from multiprocessing import get_context
import random
from typing import List, Set

import numba
import numpy as np
from sklearn import svm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from tqdm import tqdm

from motifboost.features import FeatureExtractor
from motifboost.repertoire import Repertoire

aa = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
}
aas = sorted(aa.keys())


@numba.jit()
def atchley_factor(x: str) -> np.ndarray:
    aa = {
        "A": 0,
        "C": 1,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 5,
        "H": 6,
        "I": 7,
        "K": 8,
        "L": 9,
        "M": 10,
        "N": 11,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "V": 17,
        "W": 18,
        "Y": 19,
    }
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
            [
                0.260,
                0.830,
                3.097,
                -0.838,
                1.512,
            ],
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
    _,
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
    n_gram: int,
    n_subsample: int,
    n_codewords: int,
    n_augmentation=100,
    n_jobs=1,
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
    if n_jobs > 1:
        with get_context("fork").Pool(n_jobs) as pool:
            imap = pool.imap(wrapper, range(n_augmentation))
            result = list(tqdm(imap, total=n_augmentation, desc="Augmentation"))
    else:
        result = [wrapper(x) for x in range(n_augmentation)]
    return result


class AtchleySimpleEncoder(FeatureExtractor):
    def __init__(
        self,
        n_gram: int,
        n_subsample: int,
        n_codewords: int,
        n_augmentation: int = 100,
        n_jobs=1,
    ):
        self.codeword2cluster = None
        self.codewords_atchely = None
        self.n_gram = n_gram
        self.n_subsample = n_subsample
        self.n_codewords = n_codewords
        self.n_augmentation = n_augmentation
        self.n_jobs = n_jobs

    def fit(self, repertoires: List[Repertoire]):
        # check codewords
        wrapper = functools.partial(repertoire_to_ngram, ngram=self.n_gram)
        if self.n_jobs > 1:
            with get_context("fork").Pool(self.n_jobs) as pool:
                imap = pool.imap(wrapper, repertoires)
                result = list(tqdm(imap, total=len(repertoires), desc="Ngram"))
        else:
            result = [wrapper(x) for x in tqdm(repertoires, desc="Ngram")]
        # result = [wrapper(repertoire) for repertoire in tqdm(repertoires,desc="Ngram")]
        codewords = set.union(*result, set())
        codewords_atchely = [atchley_factor(x) for x in codewords]
        self.codewords = codewords
        self.codewords_atchely = np.array(codewords_atchely)

        # codewords to cluster
        print("training KMeans...")
        kmeans = KMeans(n_clusters=self.n_codewords).fit(codewords_atchely)
        codeword2cluster = {w: l for w, l in zip(codewords, kmeans.labels_)}
        self.cluster = kmeans.labels_
        self.codeword2cluster = codeword2cluster

    def transform(self, repertoires: List[Repertoire]) -> List[List[np.ndarray]]:
        wrapper = functools.partial(
            repertoire2vector,
            codewords_atchely=self.codewords_atchely,
            cluster=self.cluster,
            n_gram=self.n_gram,
            n_subsample=self.n_subsample,
            n_codewords=self.n_codewords,
            n_augmentation=self.n_augmentation,
            n_jobs=1,
        )
        if self.n_jobs > 1:
            with get_context("fork").Pool(self.n_jobs) as pool:
                imap = pool.imap(wrapper, repertoires)
                result = list(tqdm(imap, total=len(repertoires), desc="Transforming"))
        else:
            result = [wrapper(r) for r in tqdm(repertoires, desc="Transforming")]
        return result


class AtchleySimpleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_gram: int,
        n_subsample: int,
        n_codewords: int = 100,
        n_augmentation: int = 100,
        n_jobs: int = 1,
    ):
        self.n_gram = n_gram
        self.n_subsample = n_subsample
        self.n_codewords = n_codewords
        self.n_augmentation = n_augmentation
        self.n_jobs = n_jobs
        self.feature_extractor = AtchleySimpleEncoder(
            n_gram, n_subsample, n_codewords, n_augmentation, n_jobs
        )
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
