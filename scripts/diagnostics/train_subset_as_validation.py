"""Write a copy of a DatasetDict whose validation split is a random train subset.

    python scripts/diagnostics/train_subset_as_validation.py SOURCE DESTINATION SEED

The subset has the validation split's size. Evaluating the SFT model on it
with the validation protocol shows whether the real validation questions
score like training questions or like held-out ones.
"""

import sys

from datasets import DatasetDict, load_from_disk

source, destination, seed = sys.argv[1:]
data = load_from_disk(source)
subset = data["train"].shuffle(seed=int(seed)).select(range(len(data["validation"])))
overlap = set(subset["question"]) & set(data["validation"]["question"])
assert not overlap, f"{len(overlap)} train questions are also validation questions"
DatasetDict(train=data["train"], validation=subset, test=data["test"]).save_to_disk(
    destination
)
print(f"{destination}: validation is {len(subset)} train rows, seed {seed}")
