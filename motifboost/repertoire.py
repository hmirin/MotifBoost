import copy
import functools
import glob
import multiprocessing
import os
import pathlib
import random
from typing import Any, Callable, Dict, List, Optional, Union

import cloudpickle
import numba
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from motifboost.sequences import SequenceContainer
from motifboost.util import human_amino_acids


class Repertoire(object):
    def __init__(
        self,
        experiment_id: str,
        sample_id: str,
        info: Dict[str, Union[str, int, float]],
        sequences: List[str],
        counts: List[int],
        alphabets: List[str] = human_amino_acids,
        save_memory=False,
    ):
        self.experiment_id = experiment_id
        self.sample_id = sample_id
        self.info = info
        self.alphabets = alphabets
        self.save_memory = save_memory
        self.sequences = SequenceContainer(alphabets, save_memory)
        self.sequences.bulk_append(sequences)
        self.counts = np.array(counts, dtype=np.uint16)  # max count is 65535
        self.features: Dict[str, Any] = {}

    def save(self, save_dir: str):
        path = pathlib.Path(save_dir + "/" + self.experiment_id + "/" + self.sample_id)
        path.mkdir(parents=True, exist_ok=True)
        seq_path = path / "sequence_data.feather"
        pd.DataFrame.from_dict(
            {"sequences": self.sequences.get_all(), "counts": self.counts}
        ).to_feather(str(seq_path))
        info_path = path / "info.pkl"
        with open(info_path, "wb") as f:
            cloudpickle.dump(self.info, f)

    @staticmethod
    def load(
        save_dir: str, experiment_id: str, sample_id: str, save_memory=False
    ) -> "Repertoire":
        path = pathlib.Path(save_dir + "/" + experiment_id + "/" + sample_id)
        df = pd.read_feather(path / "sequence_data.feather")
        sequences = list(df["sequences"])
        counts = list(df["counts"])
        with open(path / "info.pkl", "rb") as f:
            info = cloudpickle.load(f)
        return Repertoire(
            experiment_id, sample_id, info, sequences, counts, save_memory=save_memory
        )


def load_repertoire_wrapper(
    sample_id: str, save_dir: str, experiment_id: str, save_memory: bool
) -> Repertoire:
    return Repertoire.load(save_dir, experiment_id, sample_id, save_memory)


@functools.lru_cache(maxsize=10)
def repertoire_dataset_loader(
    save_dir: str,
    experiment_id: str,
    filter_by_sample_id: Optional[Callable[[str], bool]] = None,
    filter_by_repertoire: Optional[Callable[[Repertoire], bool]] = None,
    skip_after: Optional[int] = None,
    n_processes: Optional[int] = None,
    multiprocess_mode: bool = True,
    save_memory: bool = False,
) -> List[Repertoire]:

    wrapper = functools.partial(
        load_repertoire_wrapper,
        save_dir=save_dir,
        experiment_id=experiment_id,
        save_memory=save_memory,
    )
    if n_processes is None:
        n_processes = multiprocessing.cpu_count()
    path = save_dir + "/" + experiment_id + "/"
    sample_ids = list([os.path.basename(p) for p in glob.glob(path + "*")])
    if filter_by_sample_id is not None:
        sample_ids = [x for x in sample_ids if filter_by_sample_id(x)]
    if skip_after:
        random.shuffle(sample_ids)
        sample_ids = sample_ids[:skip_after]
    if multiprocess_mode:
        with multiprocessing.Pool(n_processes) as pool:
            imap = pool.imap(wrapper, sample_ids)
            results = list(
                tqdm(imap, total=len(sample_ids), desc="RepertoireLoader_Multi")
            )
    else:
        results = list(map(wrapper, tqdm(sample_ids, desc="RepertoireLoader_Single")))
    if filter_by_repertoire is not None:
        results = [x for x in results if filter_by_repertoire(x)]
    # for r in results:
    #     try:
    #         assert len(r.sequences.get_all()) == len(set(r.sequences.get_all()))
    #     except AssertionError as e:
    #         print("experiment_id: ", experiment_id, "has duplicate sequences. quit.")
    #     except Exception as e:
    #         raise e

    return results


def augment_repertoire(
    repertoires: List[Repertoire], n: int, subsample_size=0.25
) -> List[Repertoire]:
    data = []
    idx = 0
    while True:
        if idx >= len(repertoires):
            idx = 0
        if len(data) >= n:
            break
        data.append(repertoires[idx])
        idx += 1
    augment_simple_wrapper = functools.partial(augment_simple, subsample_size)

    return list(
        Parallel(n_jobs=-1)(
            delayed(augment_simple_wrapper)(r=x) for x in tqdm(data, desc="Augmenter")
        )
    )


@numba.njit()
def augment_numba(counts: np.array, subsample_size: float, len_seqs: int):
    sum_counts = np.sum(counts)
    n_seqs_subsampled = int(sum_counts * subsample_size)
    weights = counts / sum_counts
    # p is not supported by numba
    # indices = np.random.choice(len_seqs,size=n_seqs_subsampled, replace=True,p=weights)
    # https://github.com/numba/numba/issues/2539
    cumsum = np.cumsum(weights)
    indices = [
        np.searchsorted(cumsum, np.random.random(), side="right")
        for _ in range(n_seqs_subsampled)
    ]
    ret_arr = np.zeros((len_seqs), dtype=numba.int64)
    for i in indices:
        ret_arr[i] += 1
    ret_indices = np.ravel(np.argwhere(ret_arr > 0))
    ret_values = ret_arr[ret_indices]
    return ret_indices, ret_values


def augment_simple(subsample_size: float, r: Repertoire) -> Repertoire:
    # sample
    seqs = r.sequences.get_all()
    counts = r.counts
    key, cnt = augment_numba(counts, subsample_size, len(seqs))
    sub_seqs = [seqs[k] for k in key]
    return Repertoire(
        experiment_id=r.experiment_id,
        sample_id=r.sample_id,
        info=copy.copy(r.info),
        sequences=sub_seqs,
        counts=cnt,
        alphabets=r.alphabets,
        save_memory=r.save_memory,
    )
