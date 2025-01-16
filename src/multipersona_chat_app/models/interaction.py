from pydantic import BaseModel
from typing import Optional, List

class Interaction(BaseModel):
    """
    Represents the immediate interaction from a character:
    - outward emotional state
    - internal thoughts
    - visible or audible action
    - spoken dialogue
    - two booleans indicating if there's an expected location change or appearance change
    """
    emotion: str
    thoughts: str
    action: str
    dialogue: str
    location_change_expected: bool
    appearance_change_expected: bool


class AppearanceSegments(BaseModel):
    """
    Holds the detailed subfields describing the appearance.
    Each field is optional; empty means no changes.
    """
    hair: Optional[str] = ""
    clothing: Optional[str] = ""
    accessories_and_held_items: Optional[str] = ""
    posture_and_body_language: Optional[str] = ""
    facial_expression: Optional[str] = ""
    other_relevant_details: Optional[str] = ""


class LocationUpdate(BaseModel):
    location: str
    transition_action: str = ""
    rationale: str
    other_characters: List[str] = []


class AppearanceUpdate(BaseModel):
    new_appearance: AppearanceSegments
    transition_action: str = ""
    rationale: str
    other_characters: List[str] = []


class LocationFromScratch(BaseModel):
    location: str


class AppearanceFromScratch(BaseModel):
    hair: str = ""
    clothing: str = ""
    accessories_and_held_items: str = ""
    posture_and_body_language: str = ""
    facial_expression: str = ""
    other_relevant_details: str = ""
