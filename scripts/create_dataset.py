from pathlib import Path
from argparse import ArgumentParser
from typing import List, Tuple, Set

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# pd.options.mode.chained_assignment = "raise"

parser = ArgumentParser()
parser.add_argument(
    "--input", help="Input .tsv file", type=str, default="./data/clean.tsv"
)
parser.add_argument(
    "--output",
    help="Output folder",
    type=str,
    default="./data/emojiset-downsampled-to-10000",
)
parser.add_argument(
    "--label_list",
    help="Path to label list",
    type=str,
    default="./data/new_emoji_index.txt",
)
parser.add_argument(
    "--drop_freq",
    help="How many examples to have for each emoji",
    type=int,
    default=1000,
)
parser.add_argument(
    "--dev_test_size",
    help="Combined size of dev and test size. Half goes to dev and half to test.",
    type=float,
    default=2000,
)
parser.add_argument(
    "--downsample_max_freq",
    help=(
        "Downsample examples with labels that have frequency higher than this value to this value. "
        + "Set -1 for no downsampling"
    ),
    type=int,
    default=10000,
)
parser.add_argument(
    "--random_state", help="Random seed.", type=int, default=42,
)
parser.add_argument(
    "--blocklist",
    help="Path to a blocklist containing emoji unicodes without the 'U+' separated by commas.",
    type=str,
    default="./blocklist.txt",
)


def load_blocklist(path: str, label_list: List[str]) -> Set[int]:
    """ Loads the blocklist from given path. """
    with open(path, "r") as f:
        blocklist_unicodes = [e.strip() for e in f.read().split(",")]
    blocklist_emojis = [chr(int(f"0x{e}", 0)) for e in blocklist_unicodes]
    return set([label_list.index(e) for e in blocklist_emojis])


def print_emoji_freqs(df: pd.DataFrame, label_index: List[int]) -> None:
    value_counts = df["label"].value_counts()
    value_list = value_counts.index

    for i, count in enumerate(value_counts):
        try:
            emoji = label_index[value_list[i]]
        except IndexError:
            print(i, value_list[i], len(label_index))

        print(emoji, count)


def drop_infrequents(
    df: pd.DataFrame, label_index: List[int], freq: int
) -> Tuple[pd.DataFrame, List[int]]:
    value_counts = df["label"].value_counts()
    too_infrequents = set(value_counts[value_counts < freq].index.values.tolist())
    drops = df[df["label"].isin(too_infrequents)].index
    df.drop(drops, inplace=True)

    # Create new label index without the dropped labels
    label_index = [l for i, l in enumerate(label_index) if i not in too_infrequents]

    return df, label_index


def downsample(df: pd.DataFrame, label_index: List[int], freq: int) -> pd.DataFrame:
    value_counts = df["label"].value_counts()
    too_frequents = set(value_counts[value_counts > freq].index.values.tolist())
    by_label = []
    for idx in range(len(label_index)):
        label_df = df[df["label"] == idx]
        if idx in too_frequents:
            downsampled = resample(
                label_df, replace=False, n_samples=freq, random_state=args.random_state,
            )
            by_label.append(downsampled)
        else:
            by_label.append(label_df)

    return pd.concat(by_label).sample(frac=1.0, random_state=args.random_state)


def remove_blocklisted(
    df: pd.DataFrame, blocklist: List[str], label_index=List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    df = df[~df["label"].isin(blocklist)]
    new_label_index = [l for i, l in enumerate(label_index) if i not in blocklist]
    old_to_new_label = {
        i: new_label_index.index(l)
        for i, l in enumerate(label_index)
        if l in new_label_index
    }
    df.loc[:, "label"] = df.loc[:, "label"].apply(lambda l: old_to_new_label[l])
    return df.reset_index(drop=True), new_label_index


if __name__ == "__main__":

    args = parser.parse_args("")  # Empty string for debug. Remove when done!
    with open(args.label_list, "r") as f:
        label_index = f.read().split(",")

    df = pd.read_csv(args.input, delimiter="\t", header=None, low_memory=False)
    df.columns = ["text", "label"]

    if args.blocklist is not None:
        blocklist = load_blocklist(args.blocklist, label_index)
        df, label_index = remove_blocklisted(df, blocklist, label_index)

    if args.drop_freq > 0:
        df, label_index = drop_infrequents(df, label_index, args.drop_freq)

    # Downsample too frequent values
    if args.downsample_max_freq > 0:
        df = downsample(df, label_index, args.downsample_max_freq)

    split_ratio = args.dev_test_size / (len(df) * 2)
    train, dev = train_test_split(
        df, test_size=split_ratio, random_state=args.random_state, stratify=df["label"]
    )
    train, test = train_test_split(
        train,
        test_size=split_ratio,
        random_state=args.random_state,
        stratify=train["label"],
    )

    OUT_PATH = Path(args.output)
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    with (OUT_PATH / "labels.txt").open("w+") as f:
        f.write(",".join(label_index))

    train.to_csv(OUT_PATH / "train.tsv", header=False, index=False, sep="\t")
    dev.to_csv(OUT_PATH / "dev.tsv", header=False, index=False, sep="\t")
    test.to_csv(OUT_PATH / "test.tsv", header=False, index=False, sep="\t")

    print("done yo")
