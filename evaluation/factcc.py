import os
import uuid
import ujson

import tensorflow_datasets as tfds

ds = tfds.load('cnn_dailymail', split="test", shuffle_files=False)
print(len(ds))
target = "/Users/ashapovalov/Projects/summarization-research/data/evaluation"
partition = os.path.join(target, "dataset")
if not os.path.exists(partition):
    os.makedirs(partition)
path = os.path.join(partition, "data-dev.jsonl")

with open(path, "w+") as f:
    for example in ds:
        T = example['article'].numpy().decode('utf-8')
        G = example['highlights'].numpy().decode('utf-8')
        for claim in G.split("\n"):
            print(ujson.dumps({"id": str(uuid.uuid4()), "text": T, "claim": claim, "label": "CORRECT"}), file=f)
