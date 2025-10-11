import os
from tqdm import tqdm

import tiktoken
from datasets import load_dataset
import numpy as np

# Number of parallel processes for tokenizing the dataset
num_proc = 8

# Number of workers for load_dataset() call
num_workers = 8

enc = tiktoken.get_encoding("gpt2")

if __name__ == "__main__":
    dataset = load_dataset("roneneldan/TinyStories", num_proc=num_workers)

    split_dataset = dataset["train"].train_test_split(test_size=0.01, seed=42)
    split_dataset["finetune"] = split_dataset.pop("test") # rename 'test' to 'finetune'
    split_dataset["validation"] = dataset["validation"]

    print(split_dataset)

    # DatasetDict({
    #    train: Dataset({
    #        features: ['text'],
    #        num_rows: 2098521
    #    })
    #    validation: Dataset({
    #        features: ['text'],
    #        num_rows: 21198
    #    })
    # })

    assert split_dataset["train"] 
    assert split_dataset["validation"]
    assert split_dataset["finetune"]

    def process(example):
        """
        Tokenize the text
        """
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token) # append the end of text token
        out = {"ids": ids, "len": len(ids)}
        return out
    
    # Tokenize the dataset in parallel
    tokenized_dataset = split_dataset.map(
        process,
        remove_columns=["text"],
        num_proc=num_proc,
        desc="Tokenizing the dataset",
    )

    # Save the tokenized dataset to disk for training
    for split, dset in tokenized_dataset.items():
        arr_len = np.sum(dset["len"], dtype=np.int64)
        filename = os.path.join(os.path.dirname(__file__), f"tinystories_{split}.bin")
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples to write to disk
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
    
            # Write into memmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()