from dataclasses import dataclass
from typing import Callable, Literal, Optional

from src.repertoire import Repertoire


@dataclass()
class DatasetSettings:
    experiment_id: str
    get_class: Callable[[Repertoire], bool]
    filter_by_sample_id: Optional[Callable[[str], bool]]
    filter_by_repertoire: Optional[Callable[[Repertoire], bool]]
    get_split: Optional[
        Callable[[Repertoire], Literal["train", "test", "other"]]
    ] = None


def huth_get_class(r: Repertoire) -> bool:
    return bool(r.info["cmv"])


def huth_filter_by_sample_id(sample_id: str) -> bool:
    return "all" in sample_id


huth_classification = DatasetSettings(
    "Huth", huth_get_class, huth_filter_by_sample_id, None
)


def heather_get_class(r: Repertoire) -> bool:
    return bool(r.info["HIV"])


def heather_filter_by_sample_id_alpha(sample_id: str) -> bool:
    return "alpha" in sample_id


def heather_filter_by_sample_id_beta(sample_id: str) -> bool:
    return "beta" in sample_id


def heather_filter_by_repertoire(r: Repertoire) -> bool:
    return not r.info["treated"]


heather_classification_alpha = DatasetSettings(
    "Heather",
    heather_get_class,
    heather_filter_by_sample_id_alpha,
    heather_filter_by_repertoire,
)

heather_classification_beta = DatasetSettings(
    "Heather",
    heather_get_class,
    heather_filter_by_sample_id_beta,
    heather_filter_by_repertoire,
)


def emerson_get_class(r: Repertoire) -> bool:
    return bool(r.info["CMV"])


def emerson_cohort_get_split(
        r: Repertoire) -> Literal["train", "test", "other"]:
    if "HIP" in r.sample_id:
        return "train"
    elif "Keck" in r.sample_id:
        return "test"
    else:
        print("unknown sample_id:", r.sample_id)
        return "other"


emerson_classification_cohort_split = DatasetSettings(
    "Emerson",
    emerson_get_class,
    None,
    None,
    emerson_cohort_get_split,
)
