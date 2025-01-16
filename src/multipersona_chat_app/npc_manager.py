import logging
import asyncio
import json
from typing import Dict, List, Optional, Set
from datetime import datetime
from pydantic import BaseModel
import yaml

from db.db_manager import DBManager
from llm.ollama_client import OllamaClient
from models.interaction import Interaction, AppearanceSegments, LocationUpdate, AppearanceUpdate
from npc_prompts import (
    NPC_CREATION_SYSTEM_PROMPT,
    NPC_CREATION_USER_PROMPT,
    NPC_INTRO_SYSTEM_PROMPT,
    NPC_INTRO_USER_PROMPT,
    NPC_REPLY_SYSTEM_PROMPT,
    NPC_REPLY_USER_PROMPT
)
from templates import (
    LOCATION_UPDATE_SYSTEM_PROMPT,
    LOCATION_UPDATE_CONTEXT_TEMPLATE,
    APPEARANCE_UPDATE_SYSTEM_PROMPT,
    APPEARANCE_UPDATE_CONTEXT_TEMPLATE,
)
# --- ADDED IMPORT FOR MARKDOWN REMOVAL ---
from utils import remove_markdown

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


#
# NEW: Pydantic model to parse location check results { "same_location": bool }
#
class SameLocationCheck(BaseModel):
    same_location: bool


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

        # NEW/UPDATED: for real-time LLM status updates
        self.llm_status_callback = None

    def load_config(self, path: str) -> dict:
        try:
            with open(path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading NPC manager config: {e}")
            return {}

    # NEW/UPDATED: method to accept a callback for pushing LLM status messages
    def set_llm_status_callback(self, callback):
        self.llm_status_callback = callback

    async def is_same_location_llm(self, loc1: str, loc2: str, setting_desc: str) -> bool:
        """
        Determine via an LLM call if loc1 and loc2 refer to the same (or nearly the same) location,
        given the overall setting context. We parse structured JSON { "same_location": true|false }.
        Return the boolean from "same_location".
        """
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

        # Use an OllamaClient with the SameLocationCheck output model
        checker_client = OllamaClient(
            'src/multipersona_chat_app/config/llm_config.yaml',
            output_model=SameLocationCheck
        )
        if self.llm_client.user_selected_model:
            checker_client.set_user_selected_model(self.llm_client.user_selected_model)

        if self.llm_status_callback:
            await self.llm_status_callback(
                "Comparing two location descriptions via LLM to see if they match or overlap."
            )

        response_text = await asyncio.to_thread(
            checker_client.generate,
            prompt=user_prompt,
            system=system_prompt,
            use_cache=False
        )
        if not response_text or not isinstance(response_text, SameLocationCheck):
            return False

        if self.llm_status_callback:
            await self.llm_status_callback("Done comparing those locations.")

        return response_text.same_location

    async def check_npc_interactions(
        self,
        current_setting_description: str,
        last_checked_msg_id: int
    ) -> int:
        """
        Main entry point to check new messages (after last_checked_msg_id) to:
         1) Determine if a new NPC is needed
         2) Generate that NPC if needed and store
         3) *After processing all new messages*, check if any existing NPC wants to respond (only once)
         4) Summarize NPC memory if needed

        Returns the new 'last_checked_msg_id' after processing all relevant messages.
        """
        all_msgs = self.db.get_messages(self.session_id)
        new_messages = [
            m for m in all_msgs
            if m['id'] > last_checked_msg_id and m['message_type'] in ['user', 'character']
        ]
        if not new_messages:
            return last_checked_msg_id

        new_messages.sort(key=lambda x: x['id'])

        # Keep track of newly created NPCs so we skip them for an immediate second reply
        just_created_npcs = []

        # --- STEP 1 & 2: For each new message, handle potential NPC creation ---
        for msg in new_messages:
            relevant_lines = await self.get_relevant_recent_lines(msg['id'], self.recent_dialogue_lines)
            newly_created_npc = await self.handle_npc_creation_if_needed(relevant_lines, current_setting_description)
            if newly_created_npc:
                just_created_npcs.append(newly_created_npc)

        # --- STEP 3: Let each NPC respond only once for this batch of new messages ---
        all_npcs = self.db.get_all_npcs_in_session(self.session_id)
        for npc_name in all_npcs:
            # Skip a fresh introduction from immediately replying again
            if npc_name in just_created_npcs:
                logger.info(f"Skipping immediate reply for newly introduced NPC: {npc_name}")
                continue
            await self.process_npc_reply(npc_name, current_setting_description)

        last_checked_msg_id = new_messages[-1]['id']

        # --- STEP 4: Summaries if needed ---
        await self.maybe_summarize_npc_memory()
        return last_checked_msg_id

    async def handle_npc_creation_if_needed(self, recent_lines: List[str], setting_desc: str) -> Optional[str]:
        """
        Calls the LLM to check if we should create a new NPC based on recent lines.
        If so, create it, store in DB, then immediately generate a short LLM-based introduction
        with action and dialogue, stored as a normal NPC line.

        Returns the newly created NPC's name if created, otherwise None.
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

        if self.llm_status_callback:
            await self.llm_status_callback(
                "Checking if a new NPC is needed based on the latest lines of dialogue or context."
            )

        creation_result = await asyncio.to_thread(
            creation_client.generate,
            prompt=user_prompt,
            system=system_prompt,
            use_cache=False
        )
        if not creation_result or not isinstance(creation_result, NPCCreationOutput):
            return None

        if creation_result.should_create_npc and creation_result.npc_name.strip():
            npc_name = creation_result.npc_name.strip()
            if npc_name in known_npcs:
                logger.info(f"NPC '{npc_name}' already exists. Skipping creation.")
                return None

            self.db.add_npc_to_session(
                self.session_id,
                npc_name,
                creation_result.npc_purpose,
                creation_result.npc_appearance,
                creation_result.npc_location
            )
            logger.info(f"New NPC created: {npc_name} | Purpose='{creation_result.npc_purpose}'")

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"New NPC '{npc_name}' was created because the LLM indicated it's needed."
                )

            # Immediately generate an LLM-based introduction (action + dialogue)
            intro_system_prompt = NPC_INTRO_SYSTEM_PROMPT.format(
                npc_name=npc_name,
                npc_purpose=creation_result.npc_purpose,
                npc_appearance=creation_result.npc_appearance,
                npc_location=creation_result.npc_location
            )
            intro_user_prompt = NPC_INTRO_USER_PROMPT.format(
                recent_lines="\n".join(recent_lines),
                npc_name=npc_name
            )

            intro_client = OllamaClient(
                'src/multipersona_chat_app/config/llm_config.yaml',
                output_model=NPCInteractionOutput
            )
            if self.llm_client.user_selected_model:
                intro_client.set_user_selected_model(self.llm_client.user_selected_model)

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Generating an introduction for the newly created NPC '{npc_name}'."
                )

            intro_output = await asyncio.to_thread(
                intro_client.generate,
                prompt=intro_user_prompt,
                system=intro_system_prompt,
                use_cache=False
            )
            if intro_output and isinstance(intro_output, NPCInteractionOutput):
                # Remove markdown from raw outputs
                intro_output.action = remove_markdown(intro_output.action)
                intro_output.dialogue = remove_markdown(intro_output.dialogue)

                final_msg = ""
                if intro_output.action.strip():
                    final_msg += f"*{intro_output.action.strip()}*\n"
                if intro_output.dialogue.strip():
                    final_msg += intro_output.dialogue.strip()

                self.db.save_message(
                    session_id=self.session_id,
                    sender=f"NPC: {npc_name}",
                    message=final_msg,
                    visible=1,
                    message_type="character",
                    emotion=intro_output.emotion.strip() if intro_output.emotion else None,
                    thoughts=intro_output.thoughts.strip() if intro_output.thoughts else None
                )
                logger.info(f"NPC introduction for '{npc_name}': {final_msg}")

                if self.llm_status_callback:
                    await self.llm_status_callback(f"Introduction completed for NPC '{npc_name}'.")
            else:
                logger.debug("NPC introduction step returned no result or invalid format.")

            return npc_name

        else:
            logger.debug("No new NPC creation indicated or no valid NPC name.")
            return None

    async def process_npc_reply(self, npc_name: str, setting_desc: str):
        """
        Check if the NPC should respond to recent lines, but only if it shares location with characters.
        After generating the NPC's new message, handle possible location or appearance changes.
        """
        npc_data = self.db.get_npc_data(self.session_id, npc_name)
        if not npc_data:
            return
        npc_location = npc_data['location'] or ""
        if not npc_location.strip():
            return

        # Check if any player characters share (or nearly share) location with this NPC
        session_chars = self.db.get_session_characters(self.session_id)
        matching_chars = []
        for c_name in session_chars:
            c_loc = self.db.get_character_location(self.session_id, c_name) or ""
            same_loc = await self.is_same_location_llm(c_loc, npc_location, setting_desc)
            if same_loc:
                matching_chars.append(c_name)

        # If no one is around, NPC has no reason to speak
        if not matching_chars:
            return

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

        if self.llm_status_callback:
            await self.llm_status_callback(
                f"Generating a reply for NPC '{npc_name}' because they share location with at least one character."
            )

        result = await asyncio.to_thread(
            reply_client.generate,
            prompt=user_prompt,
            system=system_prompt,
            use_cache=False
        )
        if not result or not isinstance(result, NPCInteractionOutput):
            return

        # Strip/clean markdown from raw LLM outputs:
        result.action = remove_markdown(result.action)
        result.dialogue = remove_markdown(result.dialogue)

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

            # Handle location/appearance changes
            if result.location_change_expected:
                logger.debug(f"{npc_name} indicated location change. Now evaluating update.")
                await self.evaluate_npc_location_update(npc_name)

            if result.appearance_change_expected:
                logger.debug(f"{npc_name} indicated appearance change. Now evaluating update.")
                await self.evaluate_npc_appearance_update(npc_name)

    #
    # NEW: Evaluate NPC location change similarly to how it's done for characters
    #
    async def evaluate_npc_location_update(self, npc_name: str, visited: Optional[Set[str]] = None):
        if visited is None:
            visited = set()
        if npc_name in visited:
            return
        visited.add(npc_name)

        try:
            location_llm = OllamaClient(
                'src/multipersona_chat_app/config/llm_config.yaml',
                output_model=LocationUpdate
            )
            if self.llm_client:
                location_llm.set_user_selected_model(self.llm_client.user_selected_model)

            system_prompt = LOCATION_UPDATE_SYSTEM_PROMPT.format(
                character_name=npc_name,
                moral_guidelines=""  # Not used or can be loaded if needed
            )
            dynamic_context = self.build_npc_location_update_context(npc_name)

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Evaluating location update for NPC '{npc_name}' because location_change_expected is True."
                )

            update_response = await asyncio.to_thread(
                location_llm.generate,
                prompt=dynamic_context,
                system=system_prompt,
                use_cache=False
            )

            if not update_response or not isinstance(update_response, LocationUpdate):
                logger.info(f"No location update or invalid data for NPC '{npc_name}'.")
                return

            new_loc = (update_response.location or "").strip()
            transition_action = remove_markdown((update_response.transition_action or "").strip())
            current_location = self.db.get_npc_data(self.session_id, npc_name)['location'] or ""

            # If the new location is the same as old, treat as no change
            if new_loc == current_location:
                new_loc = ""
                transition_action = ""

            # Post a minimal transition action if non-empty
            if transition_action:
                self.db.save_message(
                    session_id=self.session_id,
                    sender=f"NPC: {npc_name}",
                    message=f"*{transition_action}*",
                    visible=1,
                    message_type="character"
                )

            if new_loc:
                # Update DB location for this NPC
                self.db.add_npc_to_session(
                    self.session_id,
                    npc_name,
                    self.db.get_npc_data(self.session_id, npc_name)['purpose'],
                    self.db.get_npc_data(self.session_id, npc_name)['appearance'],
                    new_loc
                )
                logger.info(f"NPC '{npc_name}' location updated to '{new_loc}'. Rationale: {update_response.rationale}")
            else:
                logger.info(f"No location change for NPC '{npc_name}'. Rationale: {update_response.rationale}")

            # If other_characters are indicated, evaluate them as well
            for other_npc in update_response.other_characters:
                if not other_npc:
                    continue
                if other_npc not in self.db.get_all_npcs_in_session(self.session_id):
                    logger.info(f"Ignoring 'other_characters' entry: {other_npc} not an NPC in current session.")
                    continue
                if other_npc not in visited:
                    await self.evaluate_npc_location_update(other_npc, visited=visited)

        except Exception as e:
            logger.error(f"Error in evaluate_npc_location_update for NPC '{npc_name}': {e}", exc_info=True)

    def build_npc_location_update_context(self, npc_name: str) -> str:
        # Gather a few recent lines for context
        visible_history = self.db.get_visible_messages_for_npc(self.session_id, npc_name)
        relevant_lines = []
        for m in visible_history[-5:]:
            relevant_lines.append(f"{m['sender']}: {m['message']}")
        location_so_far = self.db.get_npc_data(self.session_id, npc_name)['location'] or "(Unknown)"

        # Minimal plan text for an NPC—no dedicated plan for them, so we use a placeholder
        plan_text = "No formal plan is tracked for NPC."

        return LOCATION_UPDATE_CONTEXT_TEMPLATE.format(
            character_name=npc_name,
            location_so_far=location_so_far,
            plan_text=plan_text,
            relevant_lines=json.dumps(relevant_lines, ensure_ascii=False, indent=2)
        )

    #
    # NEW: Evaluate NPC appearance change similarly to how it's done for characters
    #
    async def evaluate_npc_appearance_update(self, npc_name: str, visited: Optional[Set[str]] = None):
        if visited is None:
            visited = set()
        if npc_name in visited:
            return
        visited.add(npc_name)

        try:
            appearance_llm = OllamaClient(
                'src/multipersona_chat_app/config/llm_config.yaml',
                output_model=AppearanceUpdate
            )
            if self.llm_client:
                appearance_llm.set_user_selected_model(self.llm_client.user_selected_model)

            system_prompt = APPEARANCE_UPDATE_SYSTEM_PROMPT.format(
                character_name=npc_name,
                moral_guidelines=""  # Not used or can be loaded if needed
            )
            dynamic_context = self.build_npc_appearance_update_context(npc_name)

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Evaluating appearance update for NPC '{npc_name}' because appearance_change_expected is True."
                )

            update_response = await asyncio.to_thread(
                appearance_llm.generate,
                prompt=dynamic_context,
                system=system_prompt,
                use_cache=False
            )

            if not update_response or not isinstance(update_response, AppearanceUpdate):
                logger.info(f"No appearance update or invalid data for NPC '{npc_name}'.")
                return

            new_segments = update_response.new_appearance
            transition_action = remove_markdown((update_response.transition_action or "").strip())

            # Retrieve old segments for NPC
            old_npc = self.db.get_npc_data(self.session_id, npc_name)
            if not old_npc:
                logger.info(f"No existing NPC data for '{npc_name}' to compare.")
                return

            old_app = old_npc['appearance']
            # We'll do a naive "only update if any field is non-empty" approach
            combined_app_fields = []

            def append_if_filled(label, new_text):
                t = new_text.strip()
                if t:
                    combined_app_fields.append(f"{label}: {t}")

            append_if_filled("Hair", new_segments.hair)
            append_if_filled("Clothing", new_segments.clothing)
            append_if_filled("Accessories/Held Items", new_segments.accessories_and_held_items)
            append_if_filled("Posture/Body Language", new_segments.posture_and_body_language)
            append_if_filled("Facial Expression", new_segments.facial_expression)
            append_if_filled("Other", new_segments.other_relevant_details)

            new_app_str = ""
            if combined_app_fields:
                new_app_str = " | ".join(combined_app_fields)
            else:
                # If the LLM didn't specify any changes, we treat that as no update
                new_app_str = old_app.strip()

            # Post the minimal transition action if non-empty
            if transition_action:
                self.db.save_message(
                    session_id=self.session_id,
                    sender=f"NPC: {npc_name}",
                    message=f"*{transition_action}*",
                    visible=1,
                    message_type="character"
                )

            if new_app_str and new_app_str != old_app.strip():
                # Update the NPC in DB with new appearance
                self.db.add_npc_to_session(
                    self.session_id,
                    npc_name,
                    old_npc['purpose'],
                    new_app_str,
                    old_npc['location']
                )
                logger.info(
                    f"Appearance updated for NPC '{npc_name}'. Rationale: {update_response.rationale} | "
                    f"New appearance: {new_app_str}"
                )
            else:
                logger.info(f"No actual appearance change for NPC '{npc_name}'. Rationale: {update_response.rationale}")

            for other_npc in update_response.other_characters:
                if not other_npc:
                    continue
                if other_npc not in self.db.get_all_npcs_in_session(self.session_id):
                    logger.info(f"Ignoring 'other_characters' entry for NPC appearance: {other_npc} not in current session.")
                    continue
                if other_npc not in visited:
                    await self.evaluate_npc_appearance_update(other_npc, visited=visited)

        except Exception as e:
            logger.error(f"Error in evaluate_npc_appearance_update for NPC '{npc_name}': {e}", exc_info=True)

    def build_npc_appearance_update_context(self, npc_name: str) -> str:
        visible_history = self.db.get_visible_messages_for_npc(self.session_id, npc_name)
        relevant_lines = [f"{m['sender']}: {m['message']}" for m in visible_history[-7:]]
        old_npc = self.db.get_npc_data(self.session_id, npc_name)
        old_appearance = old_npc['appearance'] if old_npc else "(Unknown)"

        # No plan for NPC, so just a placeholder
        plan_text = "NPC has no formal plan."

        return APPEARANCE_UPDATE_CONTEXT_TEMPLATE.format(
            character_name=npc_name,
            current_location=old_npc['location'] if old_npc else "(Unknown)",
            old_appearance=old_appearance,
            plan_text=plan_text,
            relevant_lines=json.dumps(relevant_lines, ensure_ascii=False, indent=2)
        )

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
        Gather the last 'limit' lines up to 'up_to_message_id' from user or character messages.
        If up_to_message_id is None, get the last 'limit' lines from the entire session so far.
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
