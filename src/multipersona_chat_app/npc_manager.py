import logging
import asyncio
import json
from typing import Dict, List, Optional, Set
from datetime import datetime
from pydantic import BaseModel
import yaml

from db.db_manager import DBManager
from llm.ollama_client import OllamaClient
from models.interaction import Interaction
from models.interaction import AppearanceSegments
from npc_prompts import (
    NPC_CREATION_SYSTEM_PROMPT,
    NPC_CREATION_USER_PROMPT,
    NPC_REPLY_SYSTEM_PROMPT,
    NPC_REPLY_USER_PROMPT
)

logger = logging.getLogger(__name__)


class NPCCreationOutput(BaseModel):
    should_create_npc: bool
    npc_name: str
    npc_purpose: str
    npc_appearance: str
    npc_location: str


class NPCInteractionOutput(BaseModel):
    dialogue: str
    action: str
    emotion: str
    thoughts: str
    location_change_expected: bool
    appearance_change_expected: bool


class NPCMemorySummary(BaseModel):
    id: int
    summary: str
    covered_up_to_message_id: int


class NPCManager:
    def __init__(self, session_id: str, db: DBManager, llm_client: OllamaClient, config_path: str):
        self.session_id = session_id
        self.db = db
        self.llm_client = llm_client
        self.config = self.load_config(config_path)

        self.summarization_threshold = self.config.get('summarization_threshold', 10)
        self.recent_dialogue_lines = self.config.get('recent_dialogue_lines', 3)
        self.summary_of_summaries_count = self.config.get('summary_of_summaries_count', 3)
        self.npc_similarity_threshold = self.config.get('npc_similarity_threshold', 0.8)
        self.max_similarity_retries = self.config.get('max_similarity_retries', 2)

    def load_config(self, path: str) -> dict:
        try:
            with open(path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading NPC manager config: {e}")
            return {}

    async def is_same_location_llm(self, loc1: str, loc2: str) -> bool:
        """
        Determine via an LLM call if loc1 and loc2 refer to the same location.
        Return True or False based on a JSON response { "same_location": true/false }.
        """
        # Prepare a system prompt that simply instructs the LLM to answer in strict JSON
        system_prompt = (
            "You are an assistant checking if two location descriptions refer to the exact same place.\n"
            "Respond in valid JSON only, with a single key same_location set to true or false.\n"
            "No extra text.\n"
        )
        # The user prompt includes the two location strings
        user_prompt = (
            f"Location A:\n{loc1}\n\n"
            f"Location B:\n{loc2}\n\n"
            "Do they describe the same place?"
        )

        # Create a new client for this check, or reuse config
        checker_client = OllamaClient(
            'src/multipersona_chat_app/config/llm_config.yaml'
        )
        if self.llm_client.user_selected_model:
            checker_client.set_user_selected_model(self.llm_client.user_selected_model)

        response_text = await asyncio.to_thread(
            checker_client.generate,
            prompt=user_prompt,
            system=system_prompt,
            use_cache=False
        )
        if not response_text or not isinstance(response_text, str):
            return False

        # Attempt to parse the JSON {"same_location": true/false}
        try:
            data = json.loads(response_text)
            return bool(data.get("same_location", False))
        except Exception as e:
            logger.warning(f"Failed to parse location check LLM output: {response_text} (Error: {e})")
            return False

    async def check_npc_interactions(
        self,
        current_setting_description: str,
        last_checked_msg_id: int
    ) -> int:
        """
        Main entry point to check new messages (after last_checked_msg_id) to:
         1) Determine if a new NPC is needed
         2) Generate that NPC if needed and store
         3) Check if any existing NPC wants to respond
         4) Summarize NPC memory if needed
        Returns the new 'last_checked_msg_id' after processing all relevant messages.
        """
        all_msgs = self.db.get_messages(self.session_id)
        new_messages = [m for m in all_msgs if m['id'] > last_checked_msg_id and m['message_type'] in ['user', 'character']]
        if not new_messages:
            return last_checked_msg_id

        new_messages.sort(key=lambda x: x['id'])

        # We'll iterate message by message
        for msg in new_messages:
            relevant_lines = await self.get_relevant_recent_lines(msg['id'], self.recent_dialogue_lines)
            # Step 1: Check if new NPC needed
            await self.handle_npc_creation_if_needed(relevant_lines, current_setting_description)

            # Step 2: For each NPC, see if they need to respond
            all_npcs = self.db.get_all_npcs_in_session(self.session_id)
            for npc_name in all_npcs:
                await self.process_npc_reply(npc_name)

        if new_messages:
            last_checked_msg_id = new_messages[-1]['id']

        # Summaries or merges if needed, per NPC
        await self.maybe_summarize_npc_memory()

        return last_checked_msg_id

    async def handle_npc_creation_if_needed(self, recent_lines: List[str], setting_desc: str):
        """
        Calls the LLM to check if we should create a new NPC based on recent lines.
        If so, create it, store in DB, and store the introduction message as a normal NPC line.
        We've removed the strict filter that required direct mention of the NPC name/purpose.
        """
        known_npcs = self.db.get_all_npcs_in_session(self.session_id)
        lines_for_prompt = "\n".join(recent_lines)
        known_str = ", ".join(known_npcs) if known_npcs else "(none)"

        system_prompt = NPC_CREATION_SYSTEM_PROMPT
        user_prompt = NPC_CREATION_USER_PROMPT.format(
            recent_lines=lines_for_prompt,
            known_npcs=known_str,
            setting_description=setting_desc
        )

        creation_client = OllamaClient(
            'src/multipersona_chat_app/config/llm_config.yaml',
            output_model=NPCCreationOutput
        )
        if self.llm_client.user_selected_model:
            creation_client.set_user_selected_model(self.llm_client.user_selected_model)

        creation_result = await asyncio.to_thread(
            creation_client.generate,
            prompt=user_prompt,
            system=system_prompt,
            use_cache=False
        )
        if not creation_result or not isinstance(creation_result, NPCCreationOutput):
            return

        if creation_result.should_create_npc and creation_result.npc_name.strip():
            npc_name = creation_result.npc_name.strip()
            if npc_name in known_npcs:
                logger.info(f"NPC '{npc_name}' already exists. Skipping creation.")
                return

            self.db.add_npc_to_session(
                self.session_id,
                npc_name,
                creation_result.npc_purpose,
                creation_result.npc_appearance,
                creation_result.npc_location
            )
            logger.info(f"New NPC created: {npc_name} | Purpose='{creation_result.npc_purpose}'")

            # Store introduction as if the NPC performed an entrance action + a short line of dialogue
            introduction_msg = (
                f"*Soft footsteps announce a newcomer.*\n"
                f"\"Greetings. I am {npc_name}.\""
            )
            self.db.save_message(
                session_id=self.session_id,
                sender=f"NPC: {npc_name}",
                message=introduction_msg,
                visible=1,
                message_type="character",
                emotion=None,
                thoughts=None
            )
        else:
            logger.debug("No new NPC creation indicated by LLM or no valid NPC name.")

    async def process_npc_reply(self, npc_name: str):
        """
        Check if the NPC should respond. 
        We now use an LLM-based check for location equivalence 
        instead of exact string matching to see which characters are present.
        Also, we provide an overview of *all* NPCs to the LLM so it can decide.
        """
        npc_data = self.db.get_npc_data(self.session_id, npc_name)
        if not npc_data:
            return
        npc_location = npc_data['location'] or ""
        if not npc_location.strip():
            return

        # Gather all session characters and check location equivalence via LLM
        session_chars = self.db.get_session_characters(self.session_id)
        matching_chars = []
        for c_name in session_chars:
            c_loc = self.db.get_character_location(self.session_id, c_name) or ""
            same_loc = await self.is_same_location_llm(c_loc, npc_location)
            if same_loc:
                matching_chars.append(c_name)

        if not matching_chars:
            # No character at same place => NPC idle
            return

        # Provide an overview of NPCs to the LLM
        all_npcs = self.db.get_all_npcs_in_session(self.session_id)
        all_npcs_str = ", ".join(all_npcs) if all_npcs else "(none)"

        relevant_lines = await self.get_relevant_recent_lines(None, self.recent_dialogue_lines)
        npc_summaries = self.db.get_all_npc_summaries(self.session_id, npc_name)
        joined_summaries = "\n".join([s for s in npc_summaries])

        system_prompt = NPC_REPLY_SYSTEM_PROMPT.format(
            npc_name=npc_name,
            npc_purpose=npc_data['purpose'],
            npc_appearance=npc_data['appearance'],
            npc_location=npc_data['location'],
            npc_summaries=joined_summaries
        )
        user_prompt = NPC_REPLY_USER_PROMPT.format(
            recent_lines="\n".join(relevant_lines),
            npc_location=npc_data['location'],
            npc_summaries=joined_summaries,
            all_npcs=all_npcs_str
        )

        reply_client = OllamaClient(
            'src/multipersona_chat_app/config/llm_config.yaml',
            output_model=NPCInteractionOutput
        )
        if self.llm_client.user_selected_model:
            reply_client.set_user_selected_model(self.llm_client.user_selected_model)

        result = await asyncio.to_thread(
            reply_client.generate,
            prompt=user_prompt,
            system=system_prompt,
            use_cache=False
        )
        if not result or not isinstance(result, NPCInteractionOutput):
            return

        if result.dialogue.strip() or result.action.strip():
            final_message = ""
            if result.action.strip():
                final_message += f"*{result.action}*\n"
            if result.dialogue.strip():
                final_message += result.dialogue.strip()

            self.db.save_message(
                session_id=self.session_id,
                sender=f"NPC: {npc_name}",
                message=final_message,
                visible=1,
                message_type="character",
                emotion=result.emotion.strip() if result.emotion else None,
                thoughts=result.thoughts.strip() if result.thoughts else None
            )
            logger.info(f"{npc_name} responded with: {final_message}")

            if result.location_change_expected:
                logger.debug(f"{npc_name} indicated location change. Implement if needed.")

            if result.appearance_change_expected:
                logger.debug(f"{npc_name} indicated appearance change. Implement if needed.")


    async def maybe_summarize_npc_memory(self):
        """
        Summarize each NPC's conversation lines if they exceed threshold.
        Then combine old summaries if needed.
        """
        all_npcs = self.db.get_all_npcs_in_session(self.session_id)
        for npc in all_npcs:
            await self.summarize_npc_history_if_needed(npc)
            await self.combine_summaries_if_needed(npc)

    async def summarize_npc_history_if_needed(self, npc_name: str):
        """
        Summarize the NPC's visible lines if threshold is exceeded. 
        Hide them once summarized.
        """
        npc_msgs = self.db.get_visible_messages_for_npc(self.session_id, npc_name)
        if len(npc_msgs) < self.summarization_threshold:
            return

        npc_summarize_client = OllamaClient('src/multipersona_chat_app/config/llm_config.yaml')
        if self.llm_client.user_selected_model:
            npc_summarize_client.set_user_selected_model(self.llm_client.user_selected_model)

        while True:
            npc_msgs = self.db.get_visible_messages_for_npc(self.session_id, npc_name)
            if len(npc_msgs) < self.summarization_threshold:
                break
            chunk = npc_msgs[: (self.summarization_threshold - self.recent_dialogue_lines)]
            chunk_ids = [m['id'] for m in chunk]

            lines_for_summary = []
            max_message_id_in_chunk = 0
            for m in chunk:
                if m["id"] > max_message_id_in_chunk:
                    max_message_id_in_chunk = m["id"]
                lines_for_summary.append(f"{m['sender']}: {m['message']}")

            summary_prompt = f"""
You are {npc_name}. Summarize these lines from your perspective:
{chr(10).join(lines_for_summary)}
"""
            summary = await asyncio.to_thread(
                npc_summarize_client.generate,
                prompt=summary_prompt,
                use_cache=False
            )
            if not summary:
                summary = "No significant summary."

            self.db.save_new_npc_summary(self.session_id, npc_name, summary, max_message_id_in_chunk)
            self.db.hide_messages_for_npc(self.session_id, npc_name, chunk_ids)

    async def combine_summaries_if_needed(self, npc_name: str):
        """
        Combine older NPC summaries if we exceed summary_of_summaries_count.
        """
        all_summary_records = self.db.get_all_npc_summaries_records(self.session_id, npc_name)
        if len(all_summary_records) < self.summary_of_summaries_count:
            return

        combine_client = OllamaClient('src/multipersona_chat_app/config/llm_config.yaml')
        if self.llm_client.user_selected_model:
            combine_client.set_user_selected_model(self.llm_client.user_selected_model)

        while True:
            all_summary_records = self.db.get_all_npc_summaries_records(self.session_id, npc_name)
            if len(all_summary_records) < self.summary_of_summaries_count:
                break

            chunk = all_summary_records[: self.summary_of_summaries_count]
            chunk_ids = [r["id"] for r in chunk]
            chunk_summaries = [r["summary"] for r in chunk]
            covered_up_to = max(r["covered_up_to_message_id"] for r in chunk)

            combine_prompt = f"""
You are {npc_name}. Combine these summaries into one cohesive summary:

{chr(10).join(chunk_summaries)}
"""

            combined = await asyncio.to_thread(
                combine_client.generate,
                prompt=combine_prompt,
                use_cache=False
            )
            if not combined:
                combined = "No combined summary."

            self.db.save_new_npc_summary(self.session_id, npc_name, combined, covered_up_to)
            self.db.delete_npc_summaries_by_ids(self.session_id, npc_name, chunk_ids)

    async def get_relevant_recent_lines(self, up_to_message_id: Optional[int], limit: int) -> List[str]:
        """
        Helper to gather the last 'limit' lines up to 'up_to_message_id' 
        from the main messages (user or character). 
        If up_to_message_id is None, use the entire message list's last 'limit' lines.
        """
        all_msgs = self.db.get_messages(self.session_id)
        if up_to_message_id:
            truncated = [m for m in all_msgs if m['id'] <= up_to_message_id]
        else:
            truncated = all_msgs
        truncated = [m for m in truncated if m['message_type'] in ['user', 'character']]

        if len(truncated) <= limit:
            lines = truncated
        else:
            lines = truncated[-limit:]
        lines_str = [f"{m['sender']}: {m['message']}" for m in lines]
        return lines_str
