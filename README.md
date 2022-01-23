<img width="400" alt="XCRMotifBoost" src="https://user-images.githubusercontent.com/1284876/136069583-fa9d8217-be30-4f24-9447-eafc4b473278.png">

A library for robust and data-efficient classification of RepSeq data

## Install
- Requirements
  - Python >= 3.8.5

```
git clone https://github.com/hmirin/MotifBoost
cd MotifBoost
pip install .
```

## Usage

- This library provides an easy interface for repertoire classification problem: 
  - ```Repertoire``` provides the repertoire dataset class
  - ```MotifBoostClassifier``` provies a scikit-learn like interface for our robust and data-efficient repertoire classification method.

## Example 

- See `examples` for detail.
  - Emerson, et al. Nat. Genet. 2017
  - You need to download the zip from [immuneaccess](https://clients.adaptivebiotech.com/pub/emerson-2017-natgen) and convert the data using `scripts/convert.py`.
    - Follow the instruction in the script.


```python
from motifboost.repertoire import repertoire_dataset_loader
from motifboost.dataset.settings import emerson_classification_cohort_split as settings
from motifboost.methods.motif import MotifBoostClassifier

# Get array of Repertoire instances
repertoires = repertoire_dataset_loader(
            save_dir,
            experiment_id,
            filter_by_sample_id,
            filter_by_repertoire,
            multiprocess_mode=multiprocess_mode,
            save_memory=False
        )

# Create train / val datasets from Repertoire metadata 
# Get target variable from Repertoire metadata
# Helper functions are already defined for this dataset
train_repertoires = [r for r in repertoires if settings.get_split(r) == "train"]
val_repertoires = [r for r in repertoires if settings.get_split(r) == "test"]
data_labels = [r.info["person_id"] for r in val_repertoires]

# MotifBoostClassifier implements a sklearn like API.
MotifBoostClassifier.fit(train_repertoires, [get_class(x) for x in train_repertoires])
MotifBoostClassifier.predict_proba(val_repertoires, [get_class(x) for x in val_repertoires])
```

## Citation

```
@article {MotifBoost,
	author = {Katayama, Yotaro and Kobayashi, Tetsuya J.},
	title = {MotifBoost: k-mer based data-efficient immune repertoire classification method},
	elocation-id = {2021.09.28.462258},
	year = {2021},
	doi = {10.1101/2021.09.28.462258},
	URL = {https://www.biorxiv.org/content/early/2021/10/01/2021.09.28.462258},
	eprint = {https://www.biorxiv.org/content/early/2021/10/01/2021.09.28.462258.full.pdf},
	journal = {bioRxiv}
}
```

## Reproduction of Results in the Paper

- Figure. 1, 2 and 4
  - For N=640, you can run `src/examples/emerson-dataset-comparison.py`
  - For other N, the same sampling procedure can be done with `scripts/emerson_subsampler.py`.
    - Scripts will be further added for easier reproduction
  - DeepRC must be pateched to split the data by Cohort
    - Patches will be added to this repository 
- Figure. 3
  - See `papers/k-mer-feature-space-visualization.ipynb`