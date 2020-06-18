from pathlib import Path
from typing import AbstractSet
from immutablecollections import immutableset
from flexnlp import Document
from flexnlp.model.selectors import tag
from vistautils.collection_utils import get_only
from gaia_event_extraction.pipelines import TAG_ACE_GOLD, TAG_ACE_ANNOTATED
from vistanlp_sandbox.models.events import EventMentionTheory, EVENT_TRIGGER_ROLE, EventMention

def get_triggers(emt: EventMentionTheory) -> AbstractSet[EventMention.Argument]:
    return immutableset(
        get_only(
            [
                argument
                for argument in event_mention.arguments
                if argument.event_role == EVENT_TRIGGER_ROLE
            ]
        )
        for event_mention in emt
    )

if __name__ == "__main__":
    basedir = "/nas/gaia/users/berquist/event-extraction/output"
    dirs = ("bert-annotated-ace-documents-train-with-predictions", )#"bert-annotated-ace-documents-train-with-predictions-overfit")
    filename = "AFP_ENG_20030323.0020.flexnlp"

    for d in dirs:
        p = Path(basedir) / d / filename
        with open(str(p), "rb") as handle:
            doc = Document.from_pickle_file(handle)
        emt_gold = doc.theory(EventMentionTheory, tag(TAG_ACE_GOLD))
        emt_system = doc.theory(EventMentionTheory, tag(TAG_ACE_ANNOTATED))
