#!/bin/bash
# python src/data/get_external_data.py
python src/data/merge_raw_data.py
python src/data/labelling.py
python src/data/preprocess.py
pytest test/data/test_raw_data.py
dvc add data/processed
dvc add data/raw
dvc commit
git add data/processed.dvc data/raw.dvc
dvc push
git commit --no-verify -m "`date` updated dataset"
