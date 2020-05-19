import random
import numpy as np
from keras.utils import to_categorical
from pprint import pprint
from typing import Sequence, List

_seed = 5489
random.seed(_seed)
np.random.seed(_seed)

num_samples = 23
num_event_classes = 33
max_labels = 5


def make_random_labels(
    num_samples: int, num_classes: int, max_labels_per_sample: int
) -> List[List[int]]:
    all_sample_labels = list()
    for _ in range(num_samples):
        num_labels = np.random.randint(low=1, high=max_labels + 1)
        # 50% chance that the only label is "not an event"
        if num_labels == 1 and np.random.randint(2):
            sample_labels = [0]
        else:
            sample_labels = random.sample(range(1, num_classes), k=num_labels)
        all_sample_labels.append(sample_labels)
    return all_sample_labels


def make_categorical_labels(
    all_sample_labels: Sequence[Sequence[int]], *, num_classes: int
) -> np.ndarray:
    return np.stack(
        [
            to_categorical(sample_labels, num_classes=num_classes).sum(axis=0)
            for sample_labels in all_sample_labels
        ]
    )


all_sample_labels = make_random_labels(num_samples, num_event_classes, max_labels)
pprint(all_sample_labels)
print(make_categorical_labels(all_sample_labels, num_classes=num_event_classes))
