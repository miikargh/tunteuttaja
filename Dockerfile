FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
LABEL maintainer="Miika Moilanen <mamoilanen@gmail.com>" \
    name="emojify" \
    description="API to turn text into emojis" \
    version="0.0.1"

ENV LANG=C.UTF-8
ENV PYTHONIOENCODING=UTF-8

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./tunteuttaja ./tunteuttaja
COPY ./app.py .
COPY ./models/demo-model ./model

ENV MODEL_DIR=./model
ENV LABELS_PATH=./model/labels.txt

CMD uvicorn app:app --host=0.0.0.0 --port=9090
