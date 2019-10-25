import argparse
import logging
from pathlib import Path
from typing import List, Optional, Sequence

from immutablecollections import immutableset, ImmutableSet
from vistautils.parameters import YAMLParametersLoader
from flexnlp import Document, Token


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("parameter_filename", type=Path)
    args = parser.parse_args()
    return YAMLParametersLoader().load(args.parameter_filename)


def load_serialized_documents(
    filelist: Sequence[Path], logger: Optional[logging.Logger] = None
) -> List[Document]:
    documents = []
    for filename in filelist:
        with open(filename, "rb") as handle:
            if logger:
                logger.info("Unpickling %s", filename)
            documents.append(Document.from_pickle_file(handle))
    return documents


def doc_tokens_grouped_by_sentence_slow(doc: Document) -> List[ImmutableSet[Token]]:
    return [
        immutableset(token for token in doc.tokens() if sentence.contains_span(token))
        for sentence in doc.sentences()
    ]


def doc_tokens_grouped_by_sentence_fast(doc: Document) -> List[ImmutableSet[Token]]:
    return [
        doc.tokens().tokens_enclosed_by(sentence.span) for sentence in doc.sentences()
    ]


def tokens_grouped_by_sentence_slow(documents: Sequence[Document]):
    return [doc_tokens_grouped_by_sentence_slow(doc) for doc in documents]


def tokens_grouped_by_sentence_fast(documents: Sequence[Document]):
    return [doc_tokens_grouped_by_sentence_fast(doc) for doc in documents]
