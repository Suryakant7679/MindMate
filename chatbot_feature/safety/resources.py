"""Reviewed directory. Country is supplied explicitly, never inferred from language."""
from pydantic import Field
from .schemas import StrictModel

INDIA_MENTAL_HEALTH_SUPPORT = {"tele_manas": "14416", "tele_manas_alternate": "1800-891-4416", "emergency": "112"}


class Resources(StrictModel):
    support_name: str = Field(max_length=100)
    support_number: str = Field(max_length=30, pattern=r"^[0-9+ -]+$")
    alternate: str | None = Field(default=None, max_length=30, pattern=r"^[0-9+ -]+$")
    emergency: str = Field(max_length=30, pattern=r"^[0-9+ -]+$")


DIRECTORY = {"IN": Resources(support_name="Tele-MANAS", support_number="14416", alternate="1800-891-4416", emergency="112")}
