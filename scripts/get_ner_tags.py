from typing import Collection, AbstractSet, Union, Mapping, Optional
from flexnlp import (
    Document,
    Language,
    LanguageTheory,
    MentionTheory,
    Pipeline,
    TokenTheory,
    Mention,
    Token,
    EntityType,
)
from flexnlp.annotators.apf import ApfIngester
from flexnlp.integrations.corenlp import CoreNLPNameFinder
from flexnlp.integrations.spacy import SpacyAnnotator
from flexnlp.model.algorithm import GOLD
from flexnlp.model.selectors import algorithm
from gaia_event_extraction.integrations.wordpiece import WordPieceTokenizationAnnotator
from vistautils.span import HasSpan, HasSpanIndex, Span
from immutablecollections import (
    immutabledict,
    immutableset,
    immutablesetmultidict,
    ImmutableSetMultiDict,
)

TAG = "extra-mentions"


def get_mention_pipeline() -> Pipeline:
    _mention_pipeline = (
        Pipeline.builder()
        .add(
            SpacyAnnotator.create_for_language(
                SpacyAnnotator.ENGLISH,
                use_regions=True,
                respect_existing_sentences=True,
                output_theories=(MentionTheory,),
            )
        )
        .add(CoreNLPNameFinder(), using_view={TokenTheory: algorithm(SpacyAnnotator)})
        .build()
    )
    return Pipeline.builder().add(_mention_pipeline, tag=TAG).build()


def get_items_overlapping_with_this_one(
    given: Union[Span, HasSpan], items: Collection[HasSpan]
) -> AbstractSet[HasSpan]:
    if isinstance(given, Span):
        candidate = given
    elif isinstance(given, HasSpan):
        candidate = given.span
    return HasSpanIndex.index(items).get_overlapping(candidate)


# def find_overlapping_mentions(mentions: MentionTheory):
#     for mention in mentions:


def tokens_to_mentions(
    tokens: TokenTheory, mentions: MentionTheory
) -> ImmutableSetMultiDict[Token, Mention]:
    mention_index = HasSpanIndex.index(mentions)
    ret = []
    for token in tokens:
        for mention in mention_index.get_containing(token):
            ret.append((token, mention))
    return immutablesetmultidict(ret)


# head_span_to_tokens = []
# for head_span in head_span_to_mention:
#     for token in token_index.get_contained(head_span):
#         head_span_to_tokens.append((head_span, token))
# head_span_to_tokens = immutablesetmultidict(head_span_to_tokens)

# mention_to_primary_type = immutabledict(
#     ((mention, mention.entity_type.get_primary_type()) for mention in mentions),
#     forbid_duplicate_keys=True,
# )


def tokens_to_entity_types_via_head_spans(
    tokens: TokenTheory, mentions: MentionTheory, not_an_entity_tag: str = "NA"
) -> Mapping[Token, EntityType]:
    token_index = HasSpanIndex.index_disjoint(tokens)
    head_span_to_entity_type = immutabledict(
        ((mention.head_span, mention.entity_type) for mention in mentions),
        forbid_duplicate_keys=True,
    )
    _token_to_head_span = []
    for head_span in head_span_to_entity_type:
        # Assume that there is no overlap
        for token in token_index.get_contained(head_span):
            _token_to_head_span.append((token, head_span))
    token_to_head_span = immutabledict(_token_to_head_span, forbid_duplicate_keys=True)
    _token_to_entity_type = []
    for token in tokens:
        if token not in token_to_head_span:
            et = EntityType(not_an_entity_tag)
        else:
            et = head_span_to_entity_type[token_to_head_span[token]]
        _token_to_entity_type.append((token, et))
    return immutabledict(_token_to_entity_type)


if __name__ == "__main__":

    filename = "/Users/berquist/projects/aida/event-extraction/output/bert-annotated-ace-documents-train-with-predictions/AFP_ENG_20030323.0020.flexnlp"
    with open(filename, "rb") as handle:
        doc = Document.from_pickle_file(handle)
    doc_with_lots_of_mention_algos = get_mention_pipeline().process(
        Document.builder_from_document(doc)
        .add_theory(LanguageTheory.builder().add(Language("eng", 1.0)).build(), GOLD)
        .build()
    )

    # tokens_from_spacy = doc.tokens(algorithm(SpacyAnnotator))
    tokens_from_wordpiece = doc.tokens(algorithm(WordPieceTokenizationAnnotator))
    metadata_from_wordpiece = doc.metadata_for(tokens_from_wordpiece)
    tokens_from_spacy = metadata_from_wordpiece[
        WordPieceTokenizationAnnotator.EXISTING_TOKEN_THEORY_USED_FOR_WORDPIECE
    ]
    map_spacy_to_wordpiece_indices = metadata_from_wordpiece[
        WordPieceTokenizationAnnotator.MULTIMAP_FROM_EXISTING_TO_WORDPIECE_TOKENIZATION
    ]
    mentions_from_apf = doc_with_lots_of_mention_algos.mentions(algorithm(ApfIngester))
    mentions_from_corenlp = doc_with_lots_of_mention_algos.mentions(
        algorithm(CoreNLPNameFinder)
    )
    mentions_from_spacy = doc_with_lots_of_mention_algos.mentions(
        algorithm(SpacyAnnotator)
    )

    s = Span.from_inclusive_to_exclusive(2778, 2915)
    print("== ACE ==")
    print(
        "\n".join(
            str(mention)
            for mention in get_items_overlapping_with_this_one(s, mentions_from_apf)
        )
    )
    print("== CoreNLP ==")
    print(
        "\n".join(
            str(mention)
            for mention in get_items_overlapping_with_this_one(s, mentions_from_corenlp)
        )
    )
    print("== spaCy ==")
    print(
        "\n".join(
            str(mention)
            for mention in get_items_overlapping_with_this_one(s, mentions_from_spacy)
        )
    )

    spacy_tokens_to_ace_mention_entity_types = tokens_to_entity_types_via_head_spans(
        tokens_from_spacy, mentions_from_apf
    )
    wordpiece_tokens_to_ace_mention_entity_types = immutabledict(
        (
            (
                tokens_from_wordpiece[wordpiece_token_id],
                spacy_tokens_to_ace_mention_entity_types[
                    tokens_from_spacy[spacy_token_id]
                ].get_primary_type(),
            )
            for spacy_token_id, wordpiece_token_id in map_spacy_to_wordpiece_indices.items()
        ),
        forbid_duplicate_keys=True,
    )

    entity_types_from_spacy = immutableset(m.entity_type for m in mentions_from_spacy)
    entity_types_from_corenlp = immutableset(
        m.entity_type for m in mentions_from_corenlp
    )
    entity_types_from_apf = immutableset(m.entity_type for m in mentions_from_apf)
