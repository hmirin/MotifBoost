import multiprocessing

import mlflow
from tqdm import tqdm

from src.dataset.settings import emerson_classification_cohort_split
from src.experiments.fixed_split import main
from src.methods.emerson import EmersonClassifierWithParameterSearch
from src.methods.motif import MotifBoostClassifier
from src.util import human_amino_acids

mlflow.set_experiment("emerson_cohort_split_full")


classifier_dict = {
    "motifboost": MotifBoostClassifier(),
    "emerson": EmersonClassifierWithParameterSearch(),
}

settings_dict = {
    "all": emerson_classification_cohort_split,
}

for setting_prefix, setting in settings_dict.items():
    for classifier_prefix, classifier in tqdm(
        classifier_dict.items(), desc="classifiers"
    ):
        main(
            save_dir="./data/interim/repertoires/",
            fig_save_dir="./data/",
            experiment_id=setting.experiment_id,
            filter_by_sample_id=setting.filter_by_sample_id,
            filter_by_repertoire=setting.filter_by_repertoire,
            get_class=setting.get_class,
            get_split=setting.get_split,
            classifier=classifier,
            prefix="_".join([setting.experiment_id, setting_prefix, classifier_prefix]),
            mlflow_experiment_id="emerson_cohort_split_full",
            multiprocess_mode=True,
        )
