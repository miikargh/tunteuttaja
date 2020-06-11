FROM clojure:openjdk-8-lein-2.9.3-buster as build

WORKDIR /usr/opt/

COPY ./app ./app

RUN apt-get update && apt-get install nodejs npm -y
RUN cd app && lein prod

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
LABEL maintainer="Miika Moilanen <mamoilanen@gmail.com>" \
    name="emojify" \
    description="API to turn text into emojis" \
    version="0.0.1"

ENV LANG=C.UTF-8
ENV PYTHONIOENCODING=UTF-8

WORKDIR /usr/www/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./tunteuttaja ./tunteuttaja
COPY ./app.py .
COPY ./models/demo-model ./model
COPY --from=build /usr/opt/app/resources/public ./public

ENV MODEL_DIR=./model
ENV LABELS_PATH=./model/labels.txt
ENV STATIC_PATH=/usr/www/app/public
ENV API_PORT=9090

EXPOSE $API_PORT
CMD uvicorn app:app --host=0.0.0.0 --port=$API_PORT
