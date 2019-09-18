import argparse
import logging
from itertools import chain
from pathlib import Path

from vistautils.parameters import YAMLParametersLoader
from flexnlp import Document
from gaia_event_extraction.model_data_utils import SERIALIZED_INPUT_DOCUMENT_LIST

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parameter_filename", type=Path)
    args = parser.parse_args()
    params = YAMLParametersLoader().load(args.parameter_filename)

    sns.set()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    documents = []
    for filename in params.path_list_from_file(SERIALIZED_INPUT_DOCUMENT_LIST):
        with open(filename, "rb") as file:
            logger.info("Unpickling %s", filename)
            documents.append(Document.from_pickle_file(file))

    tokens_grouped_by_sentence = [
        [
            [token for token in doc.tokens() if sentence.contains_span(token)]
            # This is actually slower?
            # doc.tokens().tokens_enclosed_by(sentence.span)
            for sentence in doc.sentences()
        ]
        for doc in documents
    ]
    sentence_lengths = list(
        chain.from_iterable(
            [
                [len(sentence) for sentence in document]
                for document in tokens_grouped_by_sentence
            ]
        )
    )

    fig, ax = plt.subplots()
    ax = sns.distplot(sentence_lengths, ax=ax)
    fig.savefig(
        params.existing_directory("analysis_output_dir") / "distplot.pdf",
        bbox_inches="tight",
    )
    fig, ax = plt.subplots()
    ax = sns.distplot(
        sentence_lengths,
        ax=ax,
        hist_kws={"cumulative": True},
        kde_kws={"cumulative": True},
    )
    fig.savefig(
        params.existing_directory("analysis_output_dir") / "distplot_cumulative.pdf",
        bbox_inches="tight",
    )
