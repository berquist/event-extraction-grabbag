import random
import numpy as np
import pytest
from keras.backend.numpy_backend import binary_crossentropy
from keras.utils import to_categorical
from pprint import pprint
from typing import List


def make_random_labels(
    *, num_samples: int, num_classes: int, max_labels_per_sample: int
) -> List[List[int]]:
    """
    Make `num_samples` worth of random labels. Each sample will be of a length between 1 and
    `max_labels_per_sample`, with values between 0 and `num_classes` + 1. It is assumed that
    `num_classes` does _not_ include the category for negative examples, which is zero.
    """
    all_sample_labels = list()
    for _ in range(num_samples):
        num_labels = np.random.randint(low=1, high=max_labels_per_sample + 1)
        # 50% chance that the only label is "not an event"
        if num_labels == 1 and np.random.randint(2):
            sample_labels = [0]
        else:
            sample_labels = random.sample(range(1, num_classes + 1), k=num_labels)
        all_sample_labels.append(sample_labels)
    return all_sample_labels


def make_categorical_labels_including_negative_example_slot(
    all_sample_labels: List[List[int]], *, num_classes: int
) -> np.ndarray:
    """
    The negative example label, zero, is assumed to *not* be one of the `num_classes`, but is
    included in the output as an explicit class.

    The output has shape `[len(all_sample_labels), num_classes + 1]`.
    """
    return np.stack(
        [
            to_categorical(sample_labels, num_classes=num_classes + 1).sum(axis=0)
            for sample_labels in all_sample_labels
        ]
    )


def make_categorical_labels_excluding_negative_example_slot(
    all_sample_labels: List[List[int]], *, num_classes: int
) -> np.ndarray:
    """
    The negative example label, zero, is assumed to *not* be one of the `num_classes` and is
    excluded from the output, represented by a vector of all zeros.

    Worded differently, for a categorical mapping, if the mapping includes the "none" category,
    `num_classes` should be `len(mapping) - 1`.

    The output has shape `[len(all_sample_labels), num_classes]`. Since the negative example label
    slot has been removed, all the categories will be shifted down by one in the result compared to
    their original category.
    """
    all_categorical_labels = list()
    for sample_labels in all_sample_labels:
        if sample_labels != [0]:
            # this "shift" doesn't work when `not_a_positive_example_value` is anything other than
            # zero
            categorical_labels = to_categorical(
                np.asarray(sample_labels) - 1, num_classes=num_classes
            ).sum(axis=0)
        else:
            categorical_labels = to_categorical([], num_classes=num_classes).sum(axis=0)
        all_categorical_labels.append(categorical_labels)
    return np.stack(all_categorical_labels)


def test_make_categorical_labels_including_negative_example_slot() -> None:
    all_sample_labels = [[1, 3, 4], [2, 4], [0], [1], [1, 5]]
    num_classes = 5
    all_categorical_labels = np.asarray(
        [
            [0, 1, 0, 1, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    np.testing.assert_equal(
        make_categorical_labels_including_negative_example_slot(
            all_sample_labels, num_classes=num_classes
        ),
        all_categorical_labels,
    )


def test_make_categorical_labels_excluding_negative_example_slot() -> None:
    all_sample_labels = [[1, 3, 4], [2, 4], [0], [1], [1, 5]]
    # For a categorical mapping, if the mapping includes the "none" category, this would be
    # `len(mapping) - 1`.
    num_classes = 5
    all_categorical_labels = np.asarray(
        [
            [1, 0, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    np.testing.assert_equal(
        make_categorical_labels_excluding_negative_example_slot(
            all_sample_labels, num_classes=num_classes
        ),
        all_categorical_labels,
    )

    # What happens when you specify one too few classes?
    with pytest.raises(IndexError) as excinfo:
        make_categorical_labels_excluding_negative_example_slot(
            all_sample_labels, num_classes=num_classes - 1
        )
    assert str(excinfo.value) == "index 4 is out of bounds for axis 1 with size 4"


def test_category_mapping_to_binary_matrix_including_negative_example_slot() -> None:
    categories = {"not a fruit": 0, "apple": 1, "pear": 2, "banana": 3, "plum": 4}
    text_labels = [
        ["not a fruit"],
        ["not a fruit"],
        ["plum"],
        ["apple"],
        # this one is a mutant
        ["apple", "plum", "pear", "banana"],
        ["not a fruit"],
    ]
    categorical_labels = [
        [categories[label] for label in sample] for sample in text_labels
    ]
    binary_matrix_labels = make_categorical_labels_including_negative_example_slot(
        categorical_labels, num_classes=len(categories) - 1
    )
    np.testing.assert_equal(
        binary_matrix_labels,
        np.asarray(
            [
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0],
                [0, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
            ]
        ),
    )


def test_category_mapping_to_binary_matrix_excluding_negative_example_slot() -> None:
    categories = {"not a fruit": 0, "apple": 1, "pear": 2, "banana": 3, "plum": 4}
    text_labels = [
        ["not a fruit"],
        ["not a fruit"],
        ["plum"],
        ["apple"],
        # this one is a mutant
        ["apple", "plum", "pear", "banana"],
        ["not a fruit"],
    ]
    categorical_labels = [
        [categories[label] for label in sample] for sample in text_labels
    ]
    binary_matrix_labels = make_categorical_labels_excluding_negative_example_slot(
        categorical_labels, num_classes=len(categories) - 1
    )
    np.testing.assert_equal(
        binary_matrix_labels,
        np.asarray(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
            ]
        ),
    )

    # What happens when you specify one too few classes?
    with pytest.raises(IndexError) as excinfo:
        make_categorical_labels_excluding_negative_example_slot(
            categorical_labels, num_classes=len(categories) - 2
        )
    assert str(excinfo.value) == "index 3 is out of bounds for axis 1 with size 3"


def event_binary_crossentropy_np(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(np.mean(binary_crossentropy(y_true, y_pred), axis=0))


def biased_event_binary_crossentropy_np(
    y_true: np.ndarray, y_pred: np.ndarray, *, beta: float = 0.0
):
    return np.mean(
        np.mean(binary_crossentropy(y_true, y_pred) * (1 + (y_true * beta)), axis=0)
    )


def round_traditional(val, digits=None):
    """https://stackoverflow.com/a/38239574"""
    return round(val + 10 ** (-len(str(val)) - 1), digits)


round_traditional_np = np.vectorize(round_traditional)


def make_logits(p):
    return np.log(p / (1 - p))


if __name__ == "__main__":
    _seed = 5489
    random.seed(_seed)
    np.random.seed(_seed)

    num_samples = 27
    num_event_classes = 10
    max_labels = 3

    all_sample_labels = make_random_labels(
        num_samples=num_samples,
        num_classes=num_event_classes,
        max_labels_per_sample=max_labels,
    )
    pprint(all_sample_labels)
    including = make_categorical_labels_including_negative_example_slot(
        all_sample_labels, num_classes=num_event_classes
    )
    excluding = make_categorical_labels_excluding_negative_example_slot(
        all_sample_labels, num_classes=num_event_classes
    )
    assert including.shape == (num_samples, num_event_classes + 1)
    assert excluding.shape == (num_samples, num_event_classes)
    print(including)
    print(excluding)
