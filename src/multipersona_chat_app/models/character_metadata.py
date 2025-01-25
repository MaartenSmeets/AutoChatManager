from pydantic import BaseModel

class CharacterMetadata(BaseModel):
    """Holds extra data for a character, including NPC-specific role/purpose."""
    is_npc: bool = False
    role: str = ""  # Purpose/role for NPCs; can be empty for PCs