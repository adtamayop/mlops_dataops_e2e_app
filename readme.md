# MLOps and DataOps integration


This repository contains an application that integrates mlops and dataops methodologies


TODO list:

1. Change features names
2. Add DVC config
3. Add pre commit config
4. Add pre commit of DVC
5. Add pytest coverage
6. LOAD api_key of alphavetage from environment
7. Guardar los datos con fechas de creación
8. Add poetry
9. Pendientes todas las formas de labelling


SABADO:


TODO: Crear sh co

1. Descargar los datos ejecutando python src/data/get_external_data.py
2. Crear dataset ejecutando python src/data/merge_raw_data.py
3. Labelling.py
4. Preprocess and split preprocess.py













CLOUD CONFIGURATION

1. Crear proyecto: mlops-dataops-project   id: 	mlops-dataops-project
2. Crear bucket: mlops-dataops-bucket
3. Crear notebooks: mlops-dataops-notebook
4. Crear pipeline de kubeflow: kubeflow-mlops-dataops-pipeline
5.





tfx==1.0.0
# tensorflow==2.5.0
dvc
dvc[gdrive]
great_expectations
pre-commit
munch
pytest
pytest-cov
python-dotenv
requests
