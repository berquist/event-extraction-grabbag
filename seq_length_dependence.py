from pathlib import Path

from flexnlp import TokenTheory
from flexnlp.model.doc_test_utils import TestDoc
from flexnlp.model.selectors import algorithm
from flexnlp.embeddings import TokenEmbeddingTheory
from gaia_event_extraction.integrations.bert import BERTWordEmbeddingAnnotator
from gaia_event_extraction.integrations.wordpiece import WordPieceTokenizationAnnotator

import numpy as np
np.set_printoptions(linewidth=200)


if __name__ == "__main__":

    model_dir = "/Users/berquist/projects/aida/event-extraction/models/bert/uncased_L-12_H-768_A-12"

    paragraph = """A paragraph is a self-contained unit of a discourse in writing dealing with a particular point or idea. A paragraph consists of one or more sentences. Though not required by the syntax of any language, paragraphs are usually an expected part of formal writing, used to organize longer prose."""
    test_doc_2 = TestDoc(paragraph).to_doc()

    b1 = BERTWordEmbeddingAnnotator.from_model_dir(
        model_dir=Path(model_dir),
        do_wordpiece=True,
        do_lower_case=True,
        layer_indices=(-1,),
        max_seq_length=24,
    )
    b2 = BERTWordEmbeddingAnnotator.from_model_dir(
        model_dir=Path(model_dir),
        do_wordpiece=True,
        do_lower_case=True,
        layer_indices=(-1,),
        max_seq_length=48,
    )
    b3 = BERTWordEmbeddingAnnotator.from_model_dir(
        model_dir=Path(model_dir),
        do_wordpiece=True,
        do_lower_case=True,
        layer_indices=(-1,),
        max_seq_length=128,
    )
    b4 = BERTWordEmbeddingAnnotator.from_model_dir(
        model_dir=Path(model_dir),
        do_wordpiece=True,
        do_lower_case=True,
        layer_indices=(-1,),
        max_seq_length=256,
    )
    b5 = BERTWordEmbeddingAnnotator.from_model_dir(
        model_dir=Path(model_dir),
        do_wordpiece=True,
        do_lower_case=True,
        layer_indices=(-1,),
        max_seq_length=512,
    )
    a1 = b1.annotate(test_doc_2)
    a2 = b2.annotate(test_doc_2)
    a3 = b3.annotate(test_doc_2)
    a4 = b4.annotate(test_doc_2)
    a5 = b5.annotate(test_doc_2)

    wordpiece_tt = a1.theory(TokenTheory, theory_selector=algorithm(WordPieceTokenizationAnnotator))

    e5 = a5.theory(TokenEmbeddingTheory).embeddings
    e4 = a4.theory(TokenEmbeddingTheory).embeddings
    e3 = a3.theory(TokenEmbeddingTheory).embeddings
    e2 = a2.theory(TokenEmbeddingTheory).embeddings
    e1 = a1.theory(TokenEmbeddingTheory).embeddings

    s1 = slice(27, 31)
    s2 = slice(0, 1)
    s3 = slice(0, 3)

    # print(e5[s1, s2, s3])
    print(e5[s1, s2, s3] - e4[s1, s2, s3])
    print(e5[s1, s2, s3] - e3[s1, s2, s3])
    print(e5[s1, s2, s3] - e2[s1, s2, s3])
    print(e5[s1, s2, s3] - e1[s1, s2, s3])
