from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from flexnlp import Pipeline

from gaia_event_extraction.integrations.bert import BERTWordEmbeddingAnnotator


def main(params: Parameters) -> None:
    bert_model_dir = params.existing_directory("bert_model_dir")

    pipeline = (
        Pipeline.builder()
        .add(
            BERTWordEmbeddingAnnotator.from_model_dir(
                model_dir=bert_model_dir,
                do_wordpiece=True,
                do_lower_case=True,
                layer_indices=(-1, -2, -3, -4),
            )
        )
        .build()
    )

    return None


if __name__ == "__main__":
    parameters_only_entry_point(main)
