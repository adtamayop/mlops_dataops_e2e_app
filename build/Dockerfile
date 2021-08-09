# syntax=docker/dockerfile:1

FROM python:3.7.11

COPY . /app

RUN python3 -m pip install --upgrade pip

RUN pip install -r app/requirements_local_run.txt

CMD ["bash"]