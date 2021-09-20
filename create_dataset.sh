#!/bin/bash
python src/data/get_external_data.py
python src/data/create_dataset.py
pytest test/data/test_raw_data.py
dvc add data/raw
dvc add data/raw_features
dvc add data/raw_labelled
dvc add data/selected_features
dvc add data/train_data
dvc commit
git add data/raw.dvc
git add data/raw_labelled.dvc
git add data/raw_features.dvc
git add data/selected_features.dvc
git add data/train_data.dvc
git add data/.gitignore
dvc push
# git commit --no-verify -m "`date` updated dataset"
