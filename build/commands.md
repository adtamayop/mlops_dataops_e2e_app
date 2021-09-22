## DOCKER

docker build --tag mlops_dataops_image build/.

docker run \
    --name mlops_dataops_image \
    --cpus="9.0" \
    --memory="10g" \
    --memory-reservation="8g" \
    -v $(pwd):/app \
    -d mlops_dataops_image tail -f /dev/null



docker exec -it mlops_dataops_image bash

docker rm -f mlops_dataops_image

borrar todas las imagenes: docker system prune -a

---

## DVC

1. `dvc init`
2.  `dvc add file_name or folder_name`
3. Va a salir un commando que debo ingresar
    `git add data/.gitignore data.csv.dvc`
4.  `git commit -m "....."`
5. en la carpeta que quiero en google drive hay un id en la url
    `dvc remote add -d storage gdrive://URL_ID`
    `dvc remote add -d storage gdrive://11R7ZKkU1NF-5le3S5Z0Z23St9IlDcNa2`
    dvc remote add --default myremote gdrive://11R7ZKkU1NF-5le3S5Z0Z23St9IlDcNa2
6. `git commit ./dvc/config -m "Configure remote storage"`
7. `dvc push` (puede que nos pida verificación)
8. si elimino los archivos, con dvc pull los puedo traer

`dvc destroy`

Movernos entre versiones de datasets en el tiempo:

1. para ver los commits y tener el id o  podemos por ejm ir al último cambio de un archivo especfico así:
    `git log --oneline`
2. `git checkout HEAD^1 data/data.csv.dvc`
3. `dvc checkout`
y ahí ya me devuelvo de versión

Obtener solo los archivos
`dvc get link_repo`
`dvc list --dvc-only link_repo`

dvc remote modify myremote --local \
dvc remote modify storage gdrive_user_credentials_file
      gdrive_user_credentials_file


dvc remote modify storage gdrive_user_credentials_file build/gdrive-user-credentials.json

pre-commit run --all-files
pytest


dvc remote add --default myremote \
                           gdrive://0AIac4JZqHhKmUk9PDA/dvcstore


                           dvc remote add -d storage gdrive://1IAd7Gtf0YElFL3N5d76yxFt42kQwQXfc


dvc remote modify storage --local gdrive_user_credentials_file build/gdrive-user-credentials.json

dvc remote modify myremote --local gdrive_service_account_json_file_path build/gdrive-user-credentials.json


Manejo de versiones de datasets:

1.  Dupliqué el contenido del dataset 1 para modificarlo
2.  dvc add data/year1month1.csv
3.  git add data/year1month1.csv.dvc
4.  git commit -m "update datset"
5.  dvc push

En este punto ya tenemos 2 versiones diferentes de un mismo archivo, así que podríamos devolvernos con git

git checkout HEAD^1 data/year1month1.csv.dvc
dvc checkout
git commit -m "revert dataset y1m1 to his original state"

y ya hemos restaurado el dataset de year1month1 a su estado original sin duplicados





Para volver el script ejecutable de creación de dataset

chmod +x ./updatescript



tensorboard --logdir /app/tfx_pipeline_output/pipelines/local_pipeline/Trainer/model_run/

mlflow ui

https://www.mlflow.org/docs/latest/tracking.html
