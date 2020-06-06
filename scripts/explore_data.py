import pandas as pd

with open("./data/new_emoji_index.txt", "r") as f:
    emoji_index = f.read().split(",")

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

df = pd.read_csv("./data/clean.tsv", delimiter="\t", header=None)
df.columns = ["text", "label"]
# print(df.head())
# print(df["label"].value_counts())
# for kek in df["label"].value_counts():
#     print(kek, kak)

value_counts = df["label"].value_counts()
value_list = value_counts.index

for i, count in enumerate(value_counts):
    try:
        emoji = emoji_index[value_list[i]]
    except IndexError:
        print(i, value_list[i], len(emoji_index))

    print(emoji, count)
