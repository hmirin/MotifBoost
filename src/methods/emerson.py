import logging
import math
import multiprocessing
import os
import sys
from collections import defaultdict
from functools import lru_cache
from multiprocessing import current_process, shared_memory
from typing import Callable, DefaultDict, Dict, List, Literal, Optional, Set

import catboost as ctb
import numpy as np
import xgboost as xgb
from immuneML.data_model.encoded_data.EncodedData import EncodedData
from immuneML.ml_methods.ProbabilisticBinaryClassifier import \
    ProbabilisticBinaryClassifier
from joblib import Parallel, delayed, parallel_backend
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.features import FeatureExtractor
from src.fisher_exact import fisher_exact
from src.repertoire import Repertoire
from src.sequences import SequenceContainer

_logger = logging.Logger(__name__)

fisher_exact_function = fisher_exact
fisher_exact(1, 1, 1, 1)


class EmersonFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        get_class: Callable[[Repertoire], bool],
        alphabets: List[str],
        th: float,
        save_fisher_results=False,
        save_memory=False,
    ):
        self.feature_sequences: Optional[List[str]] = None
        self.feature_dict: Optional[Dict[str, int]] = None
        self.get_class = get_class
        self.alphabets = alphabets
        self.th = th
        self.save_fisher_results = save_fisher_results
        self.save_memory = save_memory

    def fit(self, repertoires: List[Repertoire]):
        pos_d: DefaultDict[str, int] = defaultdict(lambda: 0)
        neg_d: DefaultDict[str, int] = defaultdict(lambda: 0)
        pos_count = 0
        neg_count = 0
        for r in tqdm(repertoires, desc="collecting..."):
            pos = self.get_class(r)
            if pos:
                pos_count += 1
                for b in set(r.sequences.get_all()):
                    pos_d[b] += 1
            else:
                neg_count += 1
                for b in set(r.sequences.get_all()):
                    neg_d[b] += 1
        keys = list(pos_d.keys() | neg_d.keys())
        features = []

        def fisher_exact_wrapper(key) -> float:
            pos_exist = pos_d[key]
            pos_missing = pos_count - pos_exist
            neg_exist = neg_d[key]
            neg_missing = neg_count - neg_exist
            return fisher_exact_function(
                pos_exist, pos_missing, neg_exist, neg_missing)

        ps = [fisher_exact_wrapper(x)
              for x in tqdm(keys, desc="exact test...")]
        features = [key for key, p in zip(keys, ps) if p < self.th]

        self.feature_sequences = features
        self.feature_dict = {
            f: idx for idx, f in enumerate(
                self.feature_sequences)}
        if self.save_fisher_results:
            self.keys = SequenceContainer(
                self.alphabets, save_memory=self.save_memory)
            self.keys.bulk_append(keys)
            self.saved_ps = np.array(ps)

    def fit_by_th(self, th):
        features = []
        for key, p in tqdm(zip(self.keys, self.saved_ps)):
            if p < self.th:
                features.append(key)
        self.feature_sequences = features
        self.feature_dict = {
            f: idx for idx, f in enumerate(
                self.feature_sequences)}

    def transform(self, repertoires: List[Repertoire]) -> List[List[int]]:
        lst = []
        for r in repertoires:
            arr = [0] * len(self.feature_sequences)
            for seq, cnt in zip(r.sequences.get_all(), r.counts):
                if seq in self.feature_dict:
                    arr[self.feature_dict[seq]] += cnt
            lst.append(arr)
        return lst

    def transform_for_immuneml(
            self, repertoires: List[Repertoire]) -> EncodedData:
        lst = []
        for r in repertoires:
            pos_counts = 0
            for seq, cnt in zip(r.sequences.get_all(), r.counts):
                if seq in self.feature_dict:
                    pos_counts += cnt
            lst.append([pos_counts, sum(r.counts)])
        return EncodedData(
            examples=np.array(lst),
            labels={"feature": [self.get_class(r) for r in repertoires]},
        )


class EmersonClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        get_class: Callable[[Repertoire], bool],
        alphabets: List[str],
        th: float,
        max_iterations: int,
        update_rate: float,
        likelihood_threshold: float = None,
        classifier_mode: Literal["beta", "catboost", "xgboost"] = "beta",
    ):
        self.classifier_mode = classifier_mode
        # xgboost
        if self.classifier_mode == "beta":
            self.clf = ProbabilisticBinaryClassifier(
                max_iterations, update_rate, likelihood_threshold
            )
        elif self.classifier_mode == "catboost":
            task_type = os.environ.get("TASK_TYPE", "GPU")
            self.clf = ctb.CatBoostClassifier(
                task_type=task_type, random_state=0)
        else:
            self.clf = xgb.XGBClassifier(
                eval_metric="auc",
                tree_method="gpu_hist",
                random_state=0,
            )
        self.feature_extractor = EmersonFeatureExtractor(
            th=th, get_class=get_class, alphabets=alphabets
        )

    def fit(self, repertoires: List[Repertoire], binary_targets: List[bool]):
        print("converting data....")
        self.feature_extractor.fit(repertoires)
        print("fitting....")
        if isinstance(self.clf, ProbabilisticBinaryClassifier):
            enc = self.feature_extractor.transform_for_immuneml(repertoires)
            self.clf.fit(encoded_data=enc, label_name="feature")
        else:
            # self.clf.fit(trigram_arrays, binary_targets)
            features = self.feature_extractor.transform(repertoires)
            self.clf.fit(
                np.array(features),
                np.array(
                    binary_targets,
                    dtype=np.int64))

    def predict(self, repertoires: List[Repertoire]) -> List[bool]:
        print("converting data....")
        if isinstance(self.clf, ProbabilisticBinaryClassifier):
            enc = self.feature_extractor.transform_for_immuneml(repertoires)
            pred_class = self.clf.predict(enc, "feature")
        else:
            features = self.feature_extractor.transform(repertoires)
            pred_class = self.clf.predict(np.array(features))
            # pred_class = self.clf.predict(trigram_arrays)
            # return [x == "True" for x in pred_class]
        return pred_class

    def predict_proba(self, repertoires: List[Repertoire]) -> np.ndarray:
        print("converting data....")
        features = self.feature_extractor.transform(repertoires)
        if isinstance(self.clf, ProbabilisticBinaryClassifier):
            enc = self.feature_extractor.transform_for_immuneml(repertoires)
            pred_proba = self.clf.predict_proba(enc, "feature")
        else:
            pred_proba = self.clf.predict_proba(np.array(features))
        # return self.clf.predict_proba(trigram_arrays)
        return pred_proba["feature"]


class EmersonClassifierWithParameterSearch(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        get_class: Callable[[Repertoire], bool],
        alphabets: List[str],
        fisher_test_ths=[0.00005, 0.0005, 0.005, 0.05],
        max_iterations=[1000, 10000],
        update_rates=[0.0001, 0.001, 0.01, 0.1],
        # fisher_test_ths = [0.00005], #CV=0.88 / Cohort2=0.83
        # max_iterations = [10000],
        # update_rates = [0.0001],
        likelihood_thresholds=[None],  # This is optional
        cross_validation_n_split=5,
        multi_process: Optional[int] = None,
    ):
        self.clfs = []
        for mi in max_iterations:
            for ur in update_rates:
                for lt in likelihood_thresholds:
                    self.clfs.append(ProbabilisticBinaryClassifier(mi, ur, lt))
        self.feature_extractors = [
            EmersonFeatureExtractor(th=th, get_class=get_class, alphabets=alphabets)
            for th in fisher_test_ths
        ]
        self.get_class = get_class
        self.alphabets = alphabets
        self.kfold = StratifiedKFold(
            n_splits=cross_validation_n_split, shuffle=True, random_state=0
        )
        self.multi_process = multi_process

    # メモリを食うのでこれは分ける
    def concurrent_wrapper(self, exp):
        fe = exp[0]
        clf = exp[1]
        split_data = exp[2]
        cv_predicted = []
        cv_ground_truth = []
        try:
            for (
                fitted_fe,
                train_repertoires,
                train_targets,
                validation_repertoires,
                validation_targets,
            ) in tqdm(split_data, desc="cv"):
                fitted_fe.fit_by_th(th=fe.th)
                enc = fitted_fe.transform_for_immuneml(train_repertoires)
                clf.fit(encoded_data=enc, label_name="feature")
                enc = fitted_fe.transform_for_immuneml(validation_repertoires)
                proba = clf.predict_proba(enc, label_name="feature")["feature"]
                cv_predicted = cv_predicted + list(proba[:, 1])
                cv_ground_truth = cv_ground_truth + validation_targets
            fpr, tpr, _ = roc_curve(cv_ground_truth, cv_predicted)
            roc_auc = auc(fpr, tpr)
            print(
                "roc_auc",
                roc_auc,
                exp[0].th,
                exp[1].max_iterations,
                exp[1].update_rate,
                exp[1].likelihood_threshold,
            )
            return roc_auc
        except Exception as e:
            print(split_file, "is error.")
            print(e)
            sys.exit(1)

    def fit(self, repertoires: List[Repertoire], binary_targets: List[bool]):
        get_class = self.get_class
        alphabets = self.alphabets
        multi_process = self.multi_process
        # create feature extractor

        def concurrent_fe_wrapper(x):
            train_indices = x[0]
            validation_indices = x[1]
            train_repertoires = []
            train_targets = []
            validation_repertoires = []
            validation_targets = []
            for i in train_indices:
                train_repertoires.append(repertoires[i])
                train_targets.append(binary_targets[i])
            for i in validation_indices:
                validation_repertoires.append(repertoires[i])
                validation_targets.append(binary_targets[i])
            fe = EmersonFeatureExtractor(
                get_class=get_class,
                alphabets=alphabets,
                th=1,
                save_fisher_results=True,
                save_memory=multi_process is not None,
            )  # 1 is meaningless here
            fe.fit(train_repertoires)
            return (
                fe,
                train_repertoires,
                train_targets,
                validation_repertoires,
                validation_targets,
            )

        if multi_process:
            with parallel_backend("loky", n_jobs=multi_process):
                split_data = list(
                    Parallel()(
                        delayed(concurrent_fe_wrapper)(x)
                        for x in self.kfold.split(repertoires, binary_targets)
                    )
                )
        else:
            split_data = [
                concurrent_fe_wrapper(x)
                for x in self.kfold.split(repertoires, binary_targets)
            ]
        print("trained feature extractors")
        exps = []
        for fe in self.feature_extractors:
            for clf in self.clfs:
                exps.append((fe, clf, split_data))
        print("training classifiers...")
        if multi_process:
            with parallel_backend("loky", n_jobs=multi_process):
                roc_aucs = Parallel()(
                    delayed(self.concurrent_wrapper)(exp)
                    for exp in tqdm(exps, desc="ParameterSearch")
                )
        else:
            roc_aucs = [
                self.concurrent_wrapper(exp)
                for exp in tqdm(exps, desc="ParameterSearch")
            ]
        # roc_aucs = list(map(concurrent_wrapper, tqdm(exps, desc="ParameterSearch")))
        roc_auc_max = 0
        best_exp = None
        for roc_auc, exp in zip(roc_aucs, exps):
            if roc_auc > roc_auc_max:
                roc_auc_max = roc_auc
                best_exp = exp
        print(
            "maximum roc_auc",
            roc_auc_max,
            best_exp[0].th,
            best_exp[1].max_iterations,
            best_exp[1].update_rate,
            best_exp[1].likelihood_threshold,
        )
        fe = best_exp[0]
        clf = best_exp[1]
        best_classifier = ProbabilisticBinaryClassifier(
            clf.max_iterations, clf.update_rate, clf.likelihood_threshold
        )
        best_feature_extractor = EmersonFeatureExtractor(
            th=fe.th, get_class=fe.get_class, alphabets=fe.alphabets
        )
        best_feature_extractor.fit(repertoires)
        enc = best_feature_extractor.transform_for_immuneml(repertoires)
        best_classifier.fit(encoded_data=enc, label_name="feature")
        self.clf = best_classifier
        self.feature_extractor = best_feature_extractor

    def predict(self, repertoires: List[Repertoire]) -> List[bool]:
        enc = self.feature_extractor.transform_for_immuneml(repertoires)
        pred_class = self.clf.predict(enc, "feature")
        return pred_class

    def predict_proba(self, repertoires: List[Repertoire]) -> np.ndarray:
        features = self.feature_extractor.transform(repertoires)
        enc = self.feature_extractor.transform_for_immuneml(repertoires)
        pred_proba = self.clf.predict_proba(enc, "feature")
        return pred_proba["feature"]
