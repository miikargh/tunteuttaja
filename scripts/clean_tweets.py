import re

import emoji
from nltk import word_tokenize
import pandas as pd
from whatlangid import WhatLangId
from tqdm import tqdm

from pprint import pprint

INPUT = "./data/all.tsv"
OUTPUT = "./data/clean.tsv"
EMOJI_INDEX_PATH = "./emoji_index.txt"
FREQ_TRESHOLD=1000 # How many examples per emoji to have at least

NEW_EMOJI_INDEX_PATH = "./new_emoji_index.txt"

wtl = WhatLangId()
df = pd.read_csv(INPUT, delimiter="\t", names=["id", "emoji", "alpha", "text"])

with open(EMOJI_INDEX_PATH, "r") as f:
    emoji_list = f.read().split(",")

emoji_to_idx = {emoji: idx for idx, emoji in enumerate(emoji_list)}


def gen_examples(df):
    for _, row in tqdm(df.iterrows()):

        text = row["text"]

        try:
            if wtl.predict_lang(text) != "fi":
                continue
        except ValueError:
            continue

        # Remove RT from start
        text = re.sub("^RT ", "", text)

        # Remove @user links
        text = re.sub(r"@[A-Za-z0-9_]+[:]*", "", text)

        # Remove urls starting with http or https
        text = re.sub(r"https?://\S+", "", text)

        # # Remove non alphanumerics
        # text = re.sub(r"([^\s\w]|_)", "", text)

        # Extract spillover emojis
        spillover_emojis = []
        non_emoji_chars = []

        text = " ".join(word_tokenize(text))

        for char in text:
            if char in emoji.UNICODE_EMOJI:
                if char in emoji_to_idx:
                    spillover_emojis.append(emoji_to_idx[char])
                else:
                    idx = len(emoji_to_idx)
                    emoji_to_idx[char] = idx
                    emoji_list.append(char)
                    spillover_emojis.append(idx)

            else:
                non_emoji_chars.append(char)

        emojiless = "".join(non_emoji_chars)
        examples = [(emojiless, row["emoji"])]

        for spillover_emoji in spillover_emojis:
            examples.append((emojiless, spillover_emoji))

        for text, emoji_idx in examples:
            yield text, emoji_idx



new_df = pd.DataFrame(gen_examples(df))
new_df = new_df.drop_duplicates()
new_df =

new_df.to_csv(OUTPUT, header=False, index=False, sep="\t")

with open(NEW_EMOJI_INDEX_PATH, "w+") as f:
    f.write(",".join(emoji_list))
