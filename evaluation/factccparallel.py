import tensorflow_datasets as tfds
import concurrent
import os
import uuid
import ujson

max_workers = 5
batch_size = 10

ds = tfds.load('cnn_dailymail', split='test', shuffle_files=False).take(60)

def write(start):
    id = str(uuid.uuid4())
    path = os.path.join(partition, "data-dev-{}.jsonl".format(id))
    with open(path, "w+") as batch:
        for example in ds.skip(start).take(batch_size):
            T = example['article'].numpy().decode('utf-8')
            G = example['highlights'].numpy().decode('utf-8')
            for claim in G.split("\n"):
                print(ujson.dumps({"id": str(uuid.uuid4()), "text": T, "claim": claim}), file=batch)

    return path


if __name__ == '__main__':
    target = "/Users/ashapovalov/Projects/summarization-research/data/evaluation"
    partition = os.path.join(target, "dataset")
    starts = [b for b in range(0, len(ds), batch_size)]
    batches = [starts[i:i+max_workers] for i in range(0, len(starts), max_workers)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch in batches:
            for path in executor.map(write, batch):
                print("Batch {} successful.".format(path))