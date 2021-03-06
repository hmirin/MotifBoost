# https://cancerres.aacrjournals.org/content/79/7/1671.long
import atexit
import datetime
import functools
import tempfile
from multiprocessing import get_context
from pathlib import Path
from typing import List

import pandas as pd
from immuneML.data_model.dataset import RepertoireDataset
from immuneML.encodings.atchley_kmer_encoding.AtchleyKmerEncoder import \
    AtchleyKmerEncoder as AtchleyKmerEncoderImmuneML
from immuneML.encodings.EncoderParams import EncoderParams
from immuneML.environment.Label import Label
from immuneML.environment.LabelConfiguration import LabelConfiguration
from immuneML.IO.dataset_import import AIRRImport
from immuneML.ml_methods.AtchleyKmerMILClassifier import \
    AtchleyKmerMILClassifier as AtchleyKmerMILClassifierImmuneML
from tqdm import tqdm

from motifboost.repertoire import Repertoire

# from logging import getLogger
# logger = getLogger(__name__)
# logger.setLevel(logging.INFO)


class TemporaryDirectoryFactory:
    def __init__(
        self,
    ):
        self.dirs: List[Path] = []

    def new(self) -> Path:
        d = Path(tempfile.mkdtemp())
        print("new temp dir created:", d)
        self.dirs.append(d)
        atexit.register(self.reset)
        return d

    def reset(self):
        import shutil

        print("Deleting temporary directory...")
        """delete all directory in dirs"""
        for d in self.dirs:
            print("deleting...", d)
            shutil.rmtree(d)
        self.dirs = []

    def __del__(self):
        self.reset()


def save_repertoire_by_immuneml_format(repertoire: Repertoire, dir: Path) -> Path:
    seqs = repertoire.sequences.get_all()
    seqs2 = ["A" * len(s) for s in seqs]
    counts = list(repertoire.counts)
    idxs = list(range(len(seqs)))
    name = repertoire.sample_id
    df = pd.DataFrame.from_dict(
        {
            "sequence_id": idxs,
            "sequence_aas": seqs,
            "sequences": seqs2,
            "counts": counts,
        }
    )
    pth = dir / f"{name}.tsv"
    df.to_csv(pth, index=False, sep="\t")
    return pth


def save_repertoires_by_immuneml_format(
    repertoires: List[Repertoire], directory: TemporaryDirectoryFactory, n_jobs=12
) -> Path:
    d = directory.new()
    wrapper = functools.partial(save_repertoire_by_immuneml_format, dir=d)
    if n_jobs > 1:
        with get_context("fork").Pool(n_jobs) as pool:
            imap = pool.imap(wrapper, repertoires)
            paths = list(tqdm(imap, total=len(repertoires), desc="Converting"))
    else:
        paths = [wrapper(x) for x in repertoires]
    subject_ids = [r.sample_id for r in repertoires]
    dic = {}
    for key in repertoires[0].info.keys():
        values = [r.info[key] for r in repertoires]
        dic.update({key: values})
    dic["filename"] = paths
    dic["subject_id"] = subject_ids
    df = pd.DataFrame.from_dict(dic)
    pth = d / "metadata.csv"
    df.to_csv(pth, index=False)
    return d


class Repertoire2ImmuneMLDataset:
    def __init__(self, n_jobs=1):
        self.temp_dir_factory = TemporaryDirectoryFactory()
        self.n_jobs = n_jobs

    def transform(self, repertoires: List[Repertoire]) -> RepertoireDataset:
        saved_path = save_repertoires_by_immuneml_format(
            repertoires, self.temp_dir_factory
        )
        print(datetime.datetime.now(), "Loading to ImmumemlRepertoire")
        datasets = AIRRImport.AIRRImport.import_dataset(
            {
                "number_of_processes": self.n_jobs,
                "path": saved_path,
                "metadata_file": saved_path / "metadata.csv",
                "result_path": saved_path,
                "region_type": "IMGT_CDR3",
                "column_mapping": {},
            },
            repertoires[0].experiment_id,
        )
        return saved_path, datasets


class AtchleyKmerMILClassifier:
    def __init__(
        self,
        target_label,
        iteration_count,
        threshold,
        evaluate_at,
        use_early_stopping=True,
        random_seed=0,
        learning_rate=0.01,
        zero_abundance_weight_init=True,
        abundance_type="relative_abundance",
        n_jobs=8,
    ):
        self.feature_extractor = None
        self.classifier = AtchleyKmerMILClassifierImmuneML(
            iteration_count,
            threshold,
            evaluate_at,
            use_early_stopping,
            random_seed,
            learning_rate,
            zero_abundance_weight_init,
            n_jobs,
        )
        self.rep2repdataset = Repertoire2ImmuneMLDataset(n_jobs)
        self.encoder_path = None
        self.target_label = Label(target_label)
        self.abundance_type = abundance_type
        self.n_jobs = n_jobs

    def fit(self, repertoires: List[Repertoire], _: List[bool]):
        print(datetime.datetime.now(), "Converting to immuneML format...")
        saved_path, datasets = self.rep2repdataset.transform(repertoires)
        print(datetime.datetime.now(), "Encoding to k-mer...")
        self.feature_extractor = AtchleyKmerEncoderImmuneML.build_object(
            datasets,
            **{
                "k": 4,
                "skip_first_n_aa": 0,
                "skip_last_n_aa": 0,
                "abundance": self.abundance_type,
                "normalize_all_features": False,
            },
        )
        self.encoder_params_fit = EncoderParams(
            saved_path / "result",
            LabelConfiguration(labels=[self.target_label]),
            pool_size=self.n_jobs,
            learn_model=True,
        )
        self.encoder_params_predict = EncoderParams(
            saved_path / "result",
            LabelConfiguration(labels=[self.target_label]),
            pool_size=self.n_jobs,
            learn_model=False,
        )
        # The next line takes long time.
        enc_dataset = self.feature_extractor.encode(datasets, self.encoder_params_fit)
        print(datetime.datetime.now(), "Training classifier...")
        self.classifier.fit(enc_dataset.encoded_data, self.target_label.name)

    def predict(self, repertoires: List[Repertoire]):
        global cached_prediction_encoded_data_key
        global cached_prediction_encoded_data_value

        def to_key(repertoires, target_label):
            return "_".join([r.sample_id for r in repertoires] + [target_label])

        use_cache = False
        if cached_prediction_encoded_data_key is None:
            pass
        else:
            if cached_prediction_encoded_data_key == to_key(
                repertoires, self.target_label.name
            ):
                enc_dataset = cached_prediction_encoded_data_value
                use_cache = True
            else:
                pass

        if use_cache:
            print("Cache hit!")
        else:
            print(datetime.datetime.now(), "Converting to immuneML format...")
            saved_path, datasets = self.rep2repdataset.transform(repertoires)
            print(datetime.datetime.now(), "Encoding to k-mer...")
            # special code for paper
            enc_dataset = self.feature_extractor.encode(
                datasets, self.encoder_params_predict
            )
            cached_prediction_encoded_data_key = to_key(
                repertoires, self.target_label.name
            )
            cached_prediction_encoded_data_value = enc_dataset
        return self.classifier.predict(
            enc_dataset.encoded_data, self.target_label.name
        )[self.target_label.name]

    def predict_proba(self, repertoires: List[Repertoire]):
        global cached_prediction_encoded_data_key
        global cached_prediction_encoded_data_value
        # special code for paper
        def to_key(repertoires, target_label):
            return "_".join([r.sample_id for r in repertoires] + [target_label])

        use_cache = False
        if cached_prediction_encoded_data_key is None:
            pass
        else:
            if cached_prediction_encoded_data_key == to_key(
                repertoires, self.target_label.name
            ):
                enc_dataset = cached_prediction_encoded_data_value
                use_cache = True
            else:
                pass

        if use_cache:
            print("Cache hit!")
        else:
            print(datetime.datetime.now(), "Converting to immuneML format...")
            saved_path, datasets = self.rep2repdataset.transform(repertoires)
            print(datetime.datetime.now(), "Encoding to k-mer...")
            enc_dataset = self.feature_extractor.encode(
                datasets, self.encoder_params_predict
            )
            cached_prediction_encoded_data_key = to_key(
                repertoires, self.target_label.name
            )
            cached_prediction_encoded_data_value = enc_dataset
        return self.classifier.predict_proba(
            enc_dataset.encoded_data, self.target_label.name
        )[self.target_label.name]


cached_prediction_encoded_data_key = None
cached_prediction_encoded_data_value = None
