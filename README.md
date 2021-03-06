<img width="400" alt="XCRMotifBoost" src="https://user-images.githubusercontent.com/1284876/136069583-fa9d8217-be30-4f24-9447-eafc4b473278.png">

A library for robust and data-efficient classification of RepSeq data

## Install
- Requirements
  - Python >= 3.8

```
git clone https://github.com/hmirin/MotifBoost
cd MotifBoost
pip install .
```

- To use with python3.7, set ``LOKY_PICKLER=pickle5``

- To run on M1 Mac (2022 Jun.)
  - Remove ``immuneML==2.1.1`` and ``numba-scipy`` ``scipy==1.6.2`` from setup.py
  - Install ``immuneML==2.1.1`` from source
    - Remove ``h5py`` from setup.py
  - Install ``numba-scipy`` from git repository
    - Remove ``scipy==1.6.2`` from setup.py in this repository
    - ``scipy==1.7.3`` will be used

## Usage

- This library provides an easy interface for repertoire classification problem: 
  - ```Repertoire``` provides the repertoire dataset class
  - ```MotifBoostClassifier``` provies a scikit-learn like interface for our robust and data-efficient repertoire classification method.

## Example x

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
clf = MotifBoostClassifier()
clf.fit(train_repertoires, [get_class(x) for x in train_repertoires])
clf.predict_proba(val_repertoires, [get_class(x) for x in val_repertoires])
```

## Citation

See [our paper](https://www.frontiersin.org/articles/10.3389/fimmu.2022.797640/full)

```
@article{10.3389/fimmu.2022.797640,
author={Katayama, Yotaro and Kobayashi, Tetsuya J.},   	 
title={Comparative Study of Repertoire Classification Methods Reveals Data Efficiency of k-mer Feature Extraction},      
journal={Frontiers in Immunology},      	
volume={13},          
year={2022}, 	  
doi={10.3389/fimmu.2022.797640},      
}
```

## Examples / Reproducing the results of the paper

- See [https://github.com/hmirin/motifboost-paper](https://github.com/hmirin/motifboost-paper)
