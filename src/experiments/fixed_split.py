from typing import Callable, List, Literal, Optional

from src.repertoire import Repertoire, repertoire_dataset_loader
from src.util import summarize_metrics


def main(
    mlflow_experiment_id: str,
    save_dir: str,
    fig_save_dir: str,
    experiment_id: str,
    get_split: Callable[[Repertoire], Literal["train", "test", "other"]],
    filter_by_sample_id: Optional[Callable[[str], bool]],
    filter_by_repertoire: Optional[Callable[[Repertoire], bool]],
    get_class: Callable[[Repertoire], bool],
    classifier,
    prefix: str,
    multiprocess_mode=True,
    repertoires: Optional[List[Repertoire]] = None,
):
    if not repertoires:
        repertoires = repertoire_dataset_loader(
            save_dir,
            experiment_id,
            filter_by_sample_id,
            filter_by_repertoire,
            multiprocess_mode=multiprocess_mode,
            save_memory=True,  # for emerson full
        )
    print("load samples:", len(repertoires))
    positive_person_ids = [r.sample_id for r in repertoires if get_class(r)]
    negative_person_ids = [
        r.sample_id for r in repertoires if not get_class(r)]
    person_ids = positive_person_ids + negative_person_ids

    assert len(person_ids) == len(set(person_ids))
    print("positive samples:", len(positive_person_ids))
    print("negative samples:", len(negative_person_ids))

    # Fixed Split CV
    train_repertoires = [r for r in repertoires if get_split(r) == "train"]
    val_repertoires = [r for r in repertoires if get_split(r) == "test"]
    data_labels = [r.info["person_id"] for r in val_repertoires]
    classifier.fit(train_repertoires, [get_class(x)
                   for x in train_repertoires])
    pred_probas = classifier.predict_proba(val_repertoires)[:, 1]
    pred_classes = pred_probas > 0.5
    ground_truths = [get_class(r) for r in val_repertoires]
    summarize_metrics(
        data_labels,
        ground_truths,
        pred_classes,
        pred_probas,
        fig_save_dir,
        experiment_id + "_" + prefix,
        mlflow_experiment_id,
    )


# main(
#     save_dir="./data/interim/repertoires/",
#     fig_save_dir="./data/",
#     experiment_id="Emerson",
#     get_split=lambda x: "train"
#     if "HIP" in x.sample_id
#     else "test"
#     if "Keck" in x.sample_id
#     else "other",
#     filter_by_sample_id=None,
#     filter_by_repertoire=None,
#     get_class=lambda x: x.info["CMV"],
#     classifier=ShannonEntropyClassifier(),
#     prefix="emerson_shannon_cohort",
# )

# main(
#     save_dir="./data/interim/repertoires/",
#     fig_save_dir="./data/",
#     experiment_id="Emerson",
#     filter_by_sample_id=None,
#     filter_by_repertoire=None,
#     get_class=lambda x: x.info["CMV"],
#     classifier=MotifClassifier(count_weight_mode=True),
#     prefix="emerson_shannon_cohort",
# )
#
