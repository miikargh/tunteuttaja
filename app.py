import json
import argparse
import os
from dataclasses import dataclass

import emoji

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from tunteuttaja import TunteuttajaPipeline


@dataclass
class Config:
    MODEL_DIR: str = os.environ.get("MODEL_DIR", "./models/demo-model")
    LABELS_PATH: str = os.environ.get("LABELS_PATH", "./models/demo-model/labels.txt")
    STATIC_PATH: str = os.environ.get(
        "STATIC_PATH", os.path.abspath("./app/resources/public")
    )


cfg = Config()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_credentials=True,
    allow_headers=["*"],
)
app.mount("/app", StaticFiles(directory=cfg.STATIC_PATH, html=True), name="app")


model = AutoModelForSequenceClassification.from_pretrained(cfg.MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_DIR)
pipeline = TunteuttajaPipeline(model=model, tokenizer=tokenizer)

with open(cfg.LABELS_PATH, "r") as f:
    label_index = f.read().split(",")


class Request(BaseModel):
    text: str
    num: int = 5


@app.post("/tunteuta")
async def predict(request: Request):
    predictions = pipeline(request.text)[: request.num]

    for pred in predictions:
        emo = label_index[pred["label"]]
        pred["emoji"] = emo
        pred["unicode"] = "{:X}".format(ord(emo))
        pred["alt"] = emoji.demojize(emo)

    return {"original_text": request.text, "predictions": predictions}
