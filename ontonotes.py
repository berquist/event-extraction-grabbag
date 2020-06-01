from flexnlp.model.ontology import Ontology
from flexnlp.model.entity import EntityType

from attr import attrs


@attrs(frozen=True, slots=True)
class OntonotesEntityTypes(Ontology):
    """
    Event types found in the OntoNotes 5.0 annotations, used in pretrained spaCy models.
    """

    @property
    def name(self) -> str:
        return "ontonotes-entity-types"

    PERSON = EntityType("PERSON")
    NORP = EntityType("NORP")
    FAC = EntityType("FAC")
    ORG = EntityType("ORG")
    GPE = EntityType("GPE")
    LOC = EntityType("LOC")
    PRODUCT = EntityType("PRODUCT")
    EVENT = EntityType("EVENT")
    WORK_OF_ART = EntityType("WORK_OF_ART")
    LAW = EntityType("LAW")
    LANGUAGE = EntityType("LANGUAGE")
    DATE = EntityType("DATE")
    TIME = EntityType("TIME")
    PERCENT = EntityType("PERCENT")
    MONEY = EntityType("MONEY")
    QUANTITY = EntityType("QUANTITY")
    ORDINAL = EntityType("ORDINAL")
    CARDINAL = EntityType("CARDINAL")
