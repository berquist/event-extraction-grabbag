import logging
from itertools import chain

from flexnlp import Document
from gaia_event_extraction.model_data_utils import SERIALIZED_INPUT_DOCUMENT_LIST

import matplotlib.pyplot as plt
import seaborn as sns

from common import get_parameters, tokens_grouped_by_sentence_fast


if __name__ == "__main__":

    sns.set()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    params = get_parameters()
    documents = load_serialized_documents(
        params.path_list_from_file(SERIALIZED_INPUT_DOCUMENT_LIST)
    )

    tokens_grouped_by_sentence = tokens_grouped_by_sentence_fast(documents)
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
