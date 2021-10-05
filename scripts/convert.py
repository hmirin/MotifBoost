import glob
import json
import logging
import os
import random
import re

import click
import pandas as pd
from tqdm import tqdm

from src.repertoire import Repertoire

_logger = logging.getLogger(__name__)

# Usage:
# Place unzipped tsvs under ./data/Emerson_NatGenet_2017/downloads/
# run this script from root like: python -m scripts.convert


def convert_Emerson_to_repertoire(
    data_folder: str, processed_data_folder: str, save_dir: str
):
    # sample informations
    cohort_1_df = pd.read_excel(
        processed_data_folder +
        "/sample_information.xlsx",
        sheet_name="Cohort 1")
    cohort_2_df = pd.read_excel(
        processed_data_folder +
        "/sample_information.xlsx",
        sheet_name="Cohort 2")
    cohort_1_df = cohort_1_df.dropna(axis=1, how="all")
    cohort_2_df = cohort_2_df.dropna(axis=1, how="all")
    sample_info_df = pd.concat([cohort_1_df, cohort_2_df])
    sample_info_df = sample_info_df[
        [c for c in sample_info_df.columns if "Inferred" not in c]
    ]
    files = set(glob.glob(data_folder + "/*.tsv"))

    for _, row in tqdm(sample_info_df.iterrows(), total=len(sample_info_df)):
        print(row)
        cmv_status = row["Known CMV status"]
        subject_id = row["Subject ID"]
        if cmv_status not in ["+", "-"]:
            print(
                "This subject has unknown CMV status:",
                row["Subject ID"],
                "Skipping...")
            continue

        matched_file = None
        for f in files:
            if subject_id in f:
                matched_file = f
                files = files - set(f)

        if matched_file is None:
            print(
                "This subject has no data available:",
                row["Subject ID"],
                "Skipping...")
            continue

        df = pd.read_csv(matched_file, sep="\t")
        df = df.dropna(axis=1, how="all")
        df = df[df["frame_type"] == "In"]

        # TODO: referring the conversion method of XXX
        count_column = "reads"
        if count_column not in df.columns:
            count_column = "templates"
        count_df = df[["amino_acid", count_column]]
        count_df = count_df.groupby("amino_acid").agg(sum).reset_index()
        r = Repertoire(
            experiment_id="Emerson",
            sample_id=subject_id,
            info={
                "CMV": cmv_status == "+",
                "person_id": subject_id,
                "sex": row["Sex"],
                "Age": row["Age"],
                "Race and Ethnicity": row[
                    "Race and ethnicity "
                ],  # trailing space is necessary
                "HLA alleles": row["Known HLA alleles"],
            },
            sequences=list(count_df["amino_acid"]),
            counts=list(count_df[count_column]),
        )
        r.save(save_dir)


def make_test_dataset(save_dir: str):
    # make 10 people dataset
    # 6 positive / 4 negative
    for i in range(10):
        positive = i < 6
        if positive:
            aas = ["A", "W", "K"]
        else:
            aas = ["A", "K"]
        sequences = [
            "".join(
                random.choices(
                    aas,
                    k=random.randint(
                        5,
                        10))) for x in range(100)]
        counts = [random.randint(1, 10) for x in range(100)]
        r = Repertoire(
            experiment_id="Test",
            sample_id="TestPatient_" + str(i),
            info={
                "positive": positive,
                "person_id": "TestPatient_" + str(i),
            },
            sequences=sequences,
            counts=counts,
        )
        r.save(save_dir)


@click.command()
@click.option(
    "--save_dir", default="./data/interim/repertoires", help="Path to save dir"
)
@click.option("--test", is_flag=True, help="Path to save dir")
def main(save_dir: str, test: bool):
    if test:
        print("generating fake classification dataset")
        make_test_dataset(save_dir)
    else:
        convert_Emerson_to_repertoire(
            "./data/Emerson_NatGenet_2017/downloads/",
            "./data/Emerson_NatGenet_2017/processed/",
            save_dir,
        )


if __name__ == "__main__":
    main()
