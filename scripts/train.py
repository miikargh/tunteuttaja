import csv
import sys
import os
from pathlib import Path
from filelock import FileLock
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
)

from pprint import pprint

PRETRAINED_MODEL = "TurkuNLP/bert-base-finnish-cased-v1"
MODEL_OUT = "./models/emoji-turku-bert-downsampled-to-10000-2epochs"
DATASET_FOLDER = "./data/emojiset-downsampled-to-10000"
LABELS_FILE = "./data/emojiset-downsampled-to-10000/labels.txt"
MAX_LEN = 128
BATCH_SIZE = 16
SAVE_STEPS = 1000
SAVE_TOTAL_LIMIT = 5
EPOCHS = 2


@dataclass(frozen=True)
class InputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None


def convert_tsv_to_features(tsv_file, tokenizer, max_len):

    with open(tsv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    texts, labels = zip(*[line.strip().split("\t") for line in lines])
    labels = [int(label) for label in labels]
    batch_encoding = tokenizer.batch_encode_plus(
        texts, max_length=max_len, pad_to_max_length=True,
    )
    features = []
    for i in range(len(texts)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        # from IPython.core.debugger import set_trace; set_trace()
        features.append(feature)

    return features


class EmojiDataset(Dataset):
    def __init__(
        self, tokenizer, data_dir, label_list_path, mode, max_len, overwrite_cache=False
    ):
        cached_features_file = (
            Path(data_dir)
            / f"cached_{mode}_{tokenizer,__class__.__name__, str(max_len)}"
        )
        lock_path = str(cached_features_file) + ".lock"

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                print("Cached features found, loading...")
                self.features = torch.load(cached_features_file)
            else:
                with open(label_list_path, "r") as f:
                    self.label_list = [i for i in range(len(f.read().split(",")))]

                self.features = convert_tsv_to_features(
                    Path(data_dir) / f"{mode}.tsv", tokenizer, max_len
                )
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def get_labels(self):
        return self.label_list


tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

train_dataset = EmojiDataset(tokenizer, DATASET_FOLDER, LABELS_FILE, "train", MAX_LEN)
eval_dataset = EmojiDataset(tokenizer, DATASET_FOLDER, LABELS_FILE, "dev", MAX_LEN)
test_dataset = EmojiDataset(tokenizer, DATASET_FOLDER, LABELS_FILE, "test", MAX_LEN)

label_list = train_dataset.get_labels()

config = AutoConfig.from_pretrained(PRETRAINED_MODEL, num_labels=len(label_list))
model = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL, config=config
)


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return (preds == p.label_ids).mean()


train_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    evaluate_during_training=True,
    per_gpu_train_batch_size=BATCH_SIZE,
    per_gpu_eval_batch_size=4,
    output_dir=MODEL_OUT,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    fp16=True,
    fp16_opt_level="O2",
    num_train_epochs=EPOCHS,
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train(model_path=MODEL_OUT if os.path.isdir(MODEL_OUT) else None,)
trainer.save_model()
print("yeaj")
