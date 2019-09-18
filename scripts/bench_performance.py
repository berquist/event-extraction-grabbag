from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Iterable, MutableMapping, Sequence, Tuple, TypeVar
import argparse
import logging
import pickle

from vistautils.collection_utils import get_only

from vistanlp_sandbox.utils.classification_tools import Predictor, TrainTestPair


T = TypeVar("T")  # pylint:disable=invalid-name


def _flatten(seq: Sequence[Sequence[T]]) -> Sequence[T]:
    return tuple(chain.from_iterable(seq))


def _compute_performance(
    predictors_and_data: Iterable[Tuple[Predictor, TrainTestPair]]
) -> None:
    # mapping from label -> 'tp'/'fp'/'fn' -> count
    performance: MutableMapping[str, MutableMapping[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for predictor, train_test_pair in predictors_and_data:
        # flattened because test data is still grouped by input object
        test_data = _flatten(train_test_pair.test_data)
        # for labelled_example in test_data:
        for labelled_example in test_data[:100]:
            # FIXME this is the problem: prediction is extremly expensive
            predicted_label = predictor.predict(labelled_example.item)
            if predicted_label == labelled_example.label:
                performance[labelled_example.label]["tp"] += 1
            else:
                performance[labelled_example.label]["fn"] += 1
                performance[predicted_label]["fp"] += 1
    for (label, scores) in sorted(performance.items()):
        precision_denom = scores["tp"] + scores["fp"]
        if precision_denom > 0:
            precision = 100.0 * scores["tp"] / precision_denom
        else:
            precision = 0.0
        recall_denom = scores["tp"] + scores["fn"]
        if recall_denom > 0:
            recall = 100.0 * scores["tp"] / recall_denom
        else:
            recall = 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        logging.info(f"{label} = {scores} = P {precision}/R {recall}/F1 {f1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_pickle_path", type=Path)
    args = parser.parse_args()

    data_and_predictors = get_only(pickle.loads(args.model_pickle_path.read_bytes()))
    train_test_pair, predictor = data_and_predictors
    predictors_and_data = [(predictor, train_test_pair)]
    _compute_performance(predictors_and_data)
