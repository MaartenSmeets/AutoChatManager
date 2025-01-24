# File: /home/maarten/AutoChatManager/src/multipersona_chat_app/npc_manager.py

import logging
import asyncio
import json
from typing import List, Optional
from pydantic import BaseModel

from db.db_manager import DBManager
import utils
from llm.ollama_client import OllamaClient
from models.character import Character
from models.character_metadata import CharacterMetadata
from npc_prompts import (
    NPC_CREATION_SYSTEM_PROMPT,
    NPC_CREATION_USER_PROMPT,
    NPC_SYSTEM_PROMPT_TEMPLATE,
    NPC_DYNAMIC_PROMPT_TEMPLATE
)

logger = logging.getLogger(__name__)

class NPCCreationOutput(BaseModel):
    should_create_npc: bool
    npc_name: str
    npc_role: str
    npc_appearance: str
    npc_location: str

class SameLocationCheck(BaseModel):
    same_location: bool

class NPCManager:
    """
    The NPCManager's role is now:
    1) Determine if a new NPC is needed and create it as a normal character (with is_npc=True in metadata).
    2) Decide if any existing NPCs should get a turn (location-based logic).
    """

    def __init__(self, session_id: str, db: DBManager, llm_client: OllamaClient):
        self.session_id = session_id
        self.db = db
        self.llm_client = llm_client

    async def maybe_create_npc(
        self,
        recent_lines: List[str],
        setting_desc: str
    ) -> Optional[str]:
        """
        Uses LLM to decide if a new NPC is needed. If yes, create it in 'session_characters'
        plus a 'character_metadata' entry with is_npc=True and the new role.
        Returns the newly created NPC name or None if no new NPC was created.
        """

        known_chars = self.db.get_character_names(self.session_id)
        npc_characters = []
        for char_name in known_chars:
            meta = self.db.get_character_metadata(self.session_id, char_name)
            if meta and meta.is_npc:
                role = meta.role.strip() if meta.role else "(No Role)"
                npc_characters.append(f"{char_name} ({role})")
            else:
                npc_characters.append(char_name)

        known_str = ", ".join(npc_characters) if npc_characters else "(none)"
        lines_for_prompt = "\n".join(recent_lines)

        user_prompt = NPC_CREATION_USER_PROMPT.format(
            recent_lines=lines_for_prompt,
            known_characters=known_str,
            setting_description=setting_desc
        )
        creation_client = OllamaClient(
            'src/multipersona_chat_app/config/llm_config.yaml',
            output_model=NPCCreationOutput
        )
        if self.llm_client.user_selected_model:
            creation_client.set_user_selected_model(self.llm_client.user_selected_model)

        system_prompt = NPC_CREATION_SYSTEM_PROMPT

        logger.info("Asking LLM if a new NPC is needed.")
        result = await asyncio.to_thread(
            creation_client.generate,
            prompt=user_prompt,
            system=system_prompt,
            use_cache=False
        )
        if not result or not isinstance(result, NPCCreationOutput):
            logger.debug("No valid NPC creation output.")
            return None

        if not result.should_create_npc:
            logger.debug("NPC creation not indicated by LLM.")
            return None

        new_npc_name = result.npc_name.strip()
        if not new_npc_name:
            logger.debug("LLM returned an empty NPC name, skipping creation.")
            return None

        if new_npc_name in known_chars:
            logger.debug(f"NPC name '{new_npc_name}' already in session; skipping.")
            return None

        # Create a new Character object
        npc_char = Character(
            name=new_npc_name,
            character_system_prompt="",     # Will fill if needed
            dynamic_prompt_template="",     # Will fill if needed
            appearance=result.npc_appearance.strip(),
            character_description="",
            fixed_traits=""
        )
        # Add them to session_characters
        self.db.add_character_to_session(
            self.session_id,
            new_npc_name,
            initial_location=result.npc_location.strip(),
            initial_appearance=npc_char.appearance
        )

        # Store in the new metadata table: is_npc=True, role=whatever
        npc_meta = CharacterMetadata(is_npc=True, role=result.npc_role.strip())
        self.db.save_character_metadata(self.session_id, new_npc_name, npc_meta)

        """
        Generates system and dynamic prompts for a new NPC and saves them to the database.
        """
        system_prompt = NPC_SYSTEM_PROMPT_TEMPLATE.replace("{npc_name}", new_npc_name) \
        .replace("{npc_role}", npc_meta.role) \
        .replace("{npc_appearance}", result.npc_appearance.strip()) \
        .replace("{npc_location}", result.npc_location.strip()) \
        .replace("{npc_goal}", "To fulfill their role as a " + result.npc_role.strip())

        dynamic_prompt_template = NPC_DYNAMIC_PROMPT_TEMPLATE.replace('{character_name}',new_npc_name)

        self.db.save_character_prompts(
                self.session_id,
                new_npc_name,
                system_prompt,
                dynamic_prompt_template
            )

        logger.info(f"New NPC '{new_npc_name}' created with role '{npc_meta.role}'.")
        return new_npc_name

    async def is_same_location_llm(self, loc1: str, loc2: str, setting_desc: str) -> bool:
        system_prompt = (
            "You are an assistant checking if two location descriptions could refer to the same or nearly the same place,\n"
            "given a certain setting context. If they are possibly describing the same or adjacent area, respond true.\n"
            "Respond in valid JSON only, with a single key 'same_location' set to true or false.\n"
            "No extra text.\n"
            f"Setting context: {setting_desc}\n"
        )
        user_prompt = (
            f"Location A:\n{loc1}\n\n"
            f"Location B:\n{loc2}\n\n"
            "Do they describe essentially the same or nearly the same place in that setting?\n"
            "Return JSON like {\"same_location\": true} or {\"same_location\": false}."
        )

        checker_client = OllamaClient(
            'src/multipersona_chat_app/config/llm_config.yaml',
            output_model=SameLocationCheck
        )
        if self.llm_client.user_selected_model:
            checker_client.set_user_selected_model(self.llm_client.user_selected_model)

        response_text = await asyncio.to_thread(
            checker_client.generate,
            prompt=user_prompt,
            system=system_prompt,
            use_cache=False
        )
        if not response_text or not isinstance(response_text, SameLocationCheck):
            return False

        return response_text.same_location

    async def npc_should_speak(self, npc_name: str, setting_desc: str) -> bool:
        """
        Decide if a given NPC 'npc_name' should get a turn (e.g. if it shares location
        with any PC, or some other logic). Return True if it should speak, else False.
        """

        # Check location overlap with any other character
        npc_loc = self.db.get_character_location(self.session_id, npc_name) or ""
        if not npc_loc.strip():
            return False

        all_chars = self.db.get_session_characters(self.session_id)
        # Filter out the NPC itself
        others = [c for c in all_chars if c != npc_name]

        # If it finds at least one overlap, returns True
        for other_name in others:
            other_loc = self.db.get_character_location(self.session_id, other_name) or ""
            # We do a naive check: if the loc strings are identical, or do something more advanced
            is_same_loc = await self.is_same_location_llm(npc_loc.strip().lower(), other_loc.strip().lower(), setting_desc)
            if is_same_loc:
                return True
        return False