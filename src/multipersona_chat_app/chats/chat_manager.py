# File: /home/maarten/AutoChatManager/src/multipersona_chat_app/chats/chat_manager.py

import os
import logging
from typing import List, Dict, Tuple, Optional, Set
from models.character import Character
from db.db_manager import DBManager
from llm.ollama_client import OllamaClient
from datetime import datetime
from npc_manager import NPCManager
import yaml
from image_manager import ImageManager
from templates import (
    INTRODUCTION_TEMPLATE,
    CHARACTER_INTRODUCTION_SYSTEM_PROMPT_TEMPLATE,
    CharacterIntroductionOutput,
    SUMMARIZE_PROMPT,
    COMBINE_SUMMARIES_PROMPT,
    LOCATION_UPDATE_SYSTEM_PROMPT,
    LOCATION_UPDATE_CONTEXT_TEMPLATE,
    APPEARANCE_UPDATE_SYSTEM_PROMPT,
    APPEARANCE_UPDATE_CONTEXT_TEMPLATE,
    PLAN_UPDATE_SYSTEM_PROMPT,
    PLAN_UPDATE_USER_PROMPT,
    LOCATION_FROM_SCRATCH_SYSTEM_PROMPT,
    LOCATION_FROM_SCRATCH_USER_PROMPT,
    APPEARANCE_FROM_SCRATCH_SYSTEM_PROMPT,
    APPEARANCE_FROM_SCRATCH_USER_PROMPT
)
from models.interaction import (
    Interaction,
    AppearanceSegments,
    LocationUpdate,
    AppearanceUpdate,
    LocationFromScratch,
    AppearanceFromScratch
)
from models.character_metadata import CharacterMetadata
from pydantic import BaseModel
import utils
import json
import asyncio
import re

logger = logging.getLogger(__name__)

from typing import List as PyList

class CharacterPlan(BaseModel):
    goal: str = ""
    steps: PyList[str] = []
    why_new_plan_goal: str = ""

class ChatManager:
    def __init__(self, session_id: Optional[str] = None, settings: List[Dict] = [],
                 llm_client: Optional[OllamaClient] = None):
            # We keep a local cache of known "PC" Characters loaded from YAML (the user picks them).
            self.characters: Dict[str, Character] = {}
            self.turn_index = 0
            self.automatic_running = False
            self.session_id = session_id if session_id else "default_session"
            self.settings = {setting['name']: setting for setting in settings}

            config_path = os.path.join("src", "multipersona_chat_app", "config", "chat_manager_config.yaml")
            self.config = self.load_config(config_path)

            self.summarization_threshold = self.config.get('summarization_threshold', 20)
            self.recent_dialogue_lines = self.config.get('recent_dialogue_lines', 5)
            self.to_summarize_count = self.summarization_threshold - self.recent_dialogue_lines

            self.similarity_threshold = self.config.get("similarity_threshold", 0.8)
            self.summary_of_summaries_count = self.config.get("summary_of_summaries_count", 5)
            self.max_similarity_retries = self.config.get("max_similarity_retries", 2)

            self.forced_update_interval = self.config.get("forced_update_interval", 5)
            self.msg_counter_since_forced_update: Dict[str, int] = {}

            db_path = os.path.join("output", "conversations.db")
            self.db = DBManager(db_path)

            self.moral_guidelines = utils.get_moral_guidelines()

            # Ensure the session exists in DB
            existing_sessions = {s['session_id']: s for s in self.db.get_all_sessions()}
            if self.session_id not in existing_sessions:
                self.db.create_session(self.session_id, f"Session {self.session_id}")
                if settings:
                    default_setting = settings[0]
                    self.set_current_setting(
                        default_setting['name'],
                        default_setting['description'],
                        default_setting['start_location']
                    )
                else:
                    self.current_setting = None
                    logger.error("No settings available to set as default.")
            else:
                stored_setting = self.db.get_current_setting(self.session_id)
                if stored_setting and stored_setting in self.settings:
                    setting = self.settings[stored_setting]
                    self.set_current_setting(
                        setting['name'],
                        self.db.get_current_setting_description(self.session_id) or setting['description'],
                        self.db.get_current_location(self.session_id) or setting['start_location']
                    )
                else:
                    if settings:
                        default_setting = settings[0]
                        self.set_current_setting(
                            default_setting['name'],
                            default_setting['description'],
                            default_setting['start_location']
                        )
                    else:
                        self.current_setting = None
                        logger.error("No matching stored setting and no default setting found. No setting applied.")

            self.llm_client = llm_client

            # --- Load session-level toggle settings (auto chat removed) ---
            session_settings = self.db.get_session_settings(self.session_id)
            self.model_selection = session_settings.get('model_selection', '')
            self.npc_manager_active = session_settings.get('npc_manager_active', False)
            self.show_private_info = session_settings.get('show_private_info', True)
            if self.npc_manager_active:
                self.npc_manager = NPCManager(
                    session_id=self.session_id,
                    db=self.db,
                    llm_client=llm_client
                )
            else:
                self.npc_manager = None

            self.llm_status_callback = None
            self._introduction_given: Dict[str, bool] = {}
            self.introduction_counter = 0
            self.introduction_sequence = {}
            self.initialize_introduction_status()

            self.auto_prompt_generation_interval = self.config.get('auto_prompt_generation_interval', 0)

            self.image_manager = ImageManager(
                config_path=os.path.join("src", "multipersona_chat_app", "config", "image_manager_config.yaml")
            )
        
    @staticmethod
    def load_config(config_path: str) -> dict:
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded successfully from {config_path}")
            return config if config else {}
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return {}

    def set_llm_status_callback(self, callback):
        """
        Provide an async callback that ChatManager can invoke to display real-time
        LLM status in a UI or log.
        """
        self.llm_status_callback = callback

    def enable_npc_manager(self):
        if not self.npc_manager:
            self.npc_manager = NPCManager(
                session_id=self.session_id,
                db=self.db,
                llm_client=self.llm_client
            )

    def disable_npc_manager(self):
        self.npc_manager = None

    @property
    def current_location(self) -> Optional[str]:
        messages = self.db.get_messages(self.session_id)
        if len(messages) == 0:
            return self.db.get_current_location(self.session_id)
        else:
            return self.get_combined_location()

    @property
    def current_setting_description(self) -> Optional[str]:
        return self.db.get_current_setting_description(self.session_id)

    def set_current_setting(self, setting_name: str, setting_description: str, start_location: str):
        """
        Updates the current setting in DB. If the session has no messages yet,
        we also set the session's initial location to 'start_location'.
        """
        self.current_setting = setting_name
        self.db.update_current_setting(self.session_id, self.current_setting)
        self.db.update_current_setting_description(self.session_id, setting_description)

        num_messages = len(self.db.get_messages(self.session_id))
        if num_messages == 0:
            self.db.update_current_location(self.session_id, start_location, None)
            logger.info(f"Setting changed to '{self.current_setting}'. Using start_location '{start_location}'.")
        else:
            logger.info(
                f"Setting changed to '{self.current_setting}' but session already has {num_messages} messages, "
                "so keeping existing session location as-is."
            )

    def get_character_names(self) -> List[str]:
        """
        Returns the list of *PC* characters that were explicitly added from YAML.
        (Not including NPCs created mid-session.)
        """
        return list(self.characters.keys())

    def add_character(self, char_name: str, char_instance: Character):
        self.characters[char_name] = char_instance
        current_session_loc = self.db.get_current_location(self.session_id) or ""
        self.db.add_character_to_session(
            self.session_id,
            char_name,
            initial_location=current_session_loc,
            initial_appearance=char_instance.appearance
        )

        meta = CharacterMetadata(is_npc=False, role="")
        self.db.save_character_metadata(self.session_id, char_name, meta)

        if char_instance.character_system_prompt and char_instance.dynamic_prompt_template:
            self.db.save_character_prompts(
                self.session_id,
                char_name,
                char_instance.character_system_prompt,
                char_instance.dynamic_prompt_template
            )
            logger.info(f"Stored system/dynamic prompts for '{char_name}' from YAML in DB.")
        else:
            logger.warning(f"No system/dynamic prompts found in YAML for '{char_name}'.")

        self.ensure_character_plan_exists(char_name)
        self.msg_counter_since_forced_update[char_name] = 0

        # Check if the character has already introduced themselves in the session
        has_introduced = any(
            m['message_type'] == 'character' and m['sender'] == char_name
            for m in self.db.get_visible_messages_for_character(self.session_id, char_name)
        )
        self._introduction_given[char_name] = has_introduced

        if char_name not in self.introduction_sequence:
            self.introduction_counter += 1
            self.introduction_sequence[char_name] = self.introduction_counter

        logger.debug(f"Introduction status for '{char_name}': {self._introduction_given[char_name]}")

    def remove_character(self, char_name: str):
        if char_name in self.characters:
            del self.characters[char_name]
        if char_name in self.msg_counter_since_forced_update:
            del self.msg_counter_since_forced_update[char_name]
        if char_name in self._introduction_given:
            del self._introduction_given[char_name]
        self.db.remove_character_from_session(self.session_id, char_name)

    def ensure_character_plan_exists(self, char_name: str):
        plan_data = self.db.get_character_plan(self.session_id, char_name)
        if plan_data is None:
            logger.info(f"No existing plan for '{char_name}'.")
        else:
            logger.debug(f"Plan for '{char_name}' found in DB. Goal: {plan_data['goal']}")

    def get_character_plan(self, char_name: str) -> CharacterPlan:
        plan_data = self.db.get_character_plan(self.session_id, char_name)
        if plan_data:
            goal = plan_data['goal'].strip()
            steps = plan_data['steps']
            if not goal and (not steps or all(not s.strip() for s in steps)):
                fallback_data = self.db.get_latest_nonempty_plan_in_history(self.session_id, char_name)
                if fallback_data:
                    return CharacterPlan(
                        goal=fallback_data['goal'],
                        steps=fallback_data['steps'],
                        why_new_plan_goal=fallback_data['why_new_plan_goal']
                    )
                else:
                    return CharacterPlan()
            else:
                return CharacterPlan(
                    goal=plan_data['goal'],
                    steps=plan_data['steps'],
                    why_new_plan_goal=plan_data['why_new_plan_goal']
                )
        else:
            fallback_data = self.db.get_latest_nonempty_plan_in_history(self.session_id, char_name)
            if fallback_data:
                return CharacterPlan(
                    goal=fallback_data['goal'],
                    steps=fallback_data['steps'],
                    why_new_plan_goal=fallback_data['why_new_plan_goal']
                )
            else:
                return CharacterPlan()

    def save_character_plan(self, char_name: str, plan: CharacterPlan):
        self.db.save_character_plan(
            self.session_id,
            char_name,
            plan.goal,
            plan.steps,
            plan.why_new_plan_goal
        )

    def get_participants_in_turn_order(self) -> List[str]:
        chars = self.db.get_session_characters(self.session_id)
        for c in chars:
            if c not in self.introduction_sequence:
                self.introduction_counter += 1
                self.introduction_sequence[c] = self.introduction_counter
        sorted_chars = sorted(chars, key=lambda c: self.introduction_sequence[c])
        return sorted_chars

    def get_upcoming_speaker(self) -> Optional[str]:
        try:
            return self.get_participants_in_turn_order()[self.turn_index]
        except IndexError:
            return None

    def initialize_introduction_status(self):
        session_chars = self.db.get_session_characters(self.session_id)
        for char_name in session_chars:
            if char_name in self.characters:
                has_introduced = any(
                    m['message_type'] == 'character' and m['sender'] == char_name
                    for m in self.db.get_visible_messages_for_character(self.session_id, char_name)
                )
                self._introduction_given[char_name] = has_introduced
                logger.debug(f"Initialized introduction status for '{char_name}': {has_introduced}")

    def restore_turn_index_from_db(self):
        """
        Finds the last message with message_type in ['character','user'].
        If found, sets self.turn_index so that the NEXT speaker after that one
        is chosen on the next call to proceed_turn().
        If no such message, keep self.turn_index = 0.
        """
        messages = self.db.get_messages(self.session_id)
        valid_msgs = [m for m in messages if m['message_type'] in ('character', 'user') and m['visible'] == 1]
        if not valid_msgs:
            self.turn_index = 0
            return

        last_msg = valid_msgs[-1]
        last_speaker = last_msg['sender']
        participants = self.get_participants_in_turn_order()
        if last_speaker not in participants:
            self.turn_index = 0
            return

        idx = participants.index(last_speaker)
        self.turn_index = (idx + 1) % len(participants)
        logger.info(f"Resuming turn order from DB. Last speaker: {last_speaker}, next index={self.turn_index}.")

    async def proceed_turn(self) -> Optional[str]:
        """
        Chooses the next speaker in round-robin order, skipping NPCs that 
        should not speak this turn. Generates that character's message.
        """
        participants = self.get_participants_in_turn_order()
        if not participants:
            logger.warning("No participants in session; proceed_turn does nothing.")
            return None

        logger.debug(f"[proceed_turn] Participants in round-robin order: {participants}")
        logger.debug(f"[proceed_turn] Current round-robin index before picking = {self.turn_index}")

        chosen_speaker = None
        count_attempts = 0
        while count_attempts < len(participants):
            c_name = participants[self.turn_index]
            logger.debug(f"[proceed_turn] Attempting speaker: {c_name}")

            meta = self.db.get_character_metadata(self.session_id, c_name)
            if meta and meta.is_npc and self.npc_manager:
                npc_should_speak = await self.npc_manager.npc_should_speak(
                    c_name, self.current_setting_description or ""
                )
                logger.debug(f"[proceed_turn] NPC '{c_name}' npc_should_speak()={npc_should_speak}")
                if not npc_should_speak:
                    logger.debug(f"[proceed_turn] Skipping NPC '{c_name}'.")
                    self.turn_index = (self.turn_index + 1) % len(participants)
                    count_attempts += 1
                    continue
            elif meta and meta.is_npc and not self.npc_manager:
                logger.debug(f"[proceed_turn] Skipping NPC '{c_name}' because npc_manager is disabled.")
                self.turn_index = (self.turn_index + 1) % len(participants)
                count_attempts += 1
                continue

            chosen_speaker = c_name
            self.turn_index = (self.turn_index + 1) % len(participants)
            break

        if not chosen_speaker:
            logger.debug("[proceed_turn] No speaker found (all NPCs refused, or no participants).")
            return None

        logger.info(f"[proceed_turn] Chosen speaker: {chosen_speaker}")
        was_introduced_before = self._introduction_given.get(chosen_speaker, False)
        await self.generate_character_message(chosen_speaker)

        if not was_introduced_before and self._introduction_given.get(chosen_speaker, False):
            logger.debug(f"[proceed_turn] {chosen_speaker} just introduced themselves. Ending turn.")
            return chosen_speaker

        # If everyone has introduced themselves, optionally try NPC creation:
        all_introduced = all(self._introduction_given.get(p, False) for p in participants)
        if all_introduced and self.npc_manager:
            all_msgs = self.db.get_messages(self.session_id)
            recent_lines = [f"{m['sender']}: {m['message']}" for m in all_msgs[-5:]]
            setting_desc = self.current_setting_description or ""
            new_npc_name = await self.npc_manager.maybe_create_npc(recent_lines, setting_desc)
            if new_npc_name:
                logger.info(f"[proceed_turn] A new NPC '{new_npc_name}' was created. They speak immediately!")
                self.introduction_counter += 1
                self.introduction_sequence[new_npc_name] = self.introduction_counter
                await self.generate_character_message(new_npc_name)

        return chosen_speaker

    def get_visible_history_for_character(self, character_name: str) -> List[Dict]:
        return self.db.get_visible_messages_for_character(self.session_id, character_name)

    def get_latest_single_utterance(self, character_name: str) -> str:
        history = self.get_visible_history_for_character(character_name)
        if not history:
            return ""
        last_msg = history[-1]
        return f"{last_msg['sender']}: {last_msg['message']}"

    async def add_message(self,
                          sender: str,
                          message: str,
                          visible: bool = True,
                          message_type: str = "user",
                          emotion: Optional[str] = None,
                          thoughts: Optional[str] = None):
        if message_type == "system" or message.strip() == "...":
            return None

        message_id = self.db.save_message(
            self.session_id,
            sender,
            message,
            int(visible),
            message_type,
            emotion,
            thoughts
        )
        self.db.add_message_visibility_for_session_characters(self.session_id, message_id)

        await self.check_summarization()

        if self.auto_prompt_generation_interval > 0:
            total_messages = len(self.db.get_messages(self.session_id))
            if total_messages % self.auto_prompt_generation_interval == 0:
                await self.generate_scene_prompt()

        return message_id

    async def generate_scene_prompt(self):
        """
        Generates only system_prompt.txt and user_prompt.txt for the current scene,
        saved in a dedicated 'image_prompts' folder. Incorporates previous summaries from
        non-NPC characters to enrich the scene prompt.
        No LLM calls or additional output files are produced.
        """
        setting_desc = self.db.get_current_setting_description(self.session_id) or ""
        if not setting_desc.strip() and self.current_setting in self.settings:
            setting_desc = self.settings[self.current_setting].get('description', '')

        guidelines = self.moral_guidelines or ""

        # Collect all non-NPC characters' data and gather previous summaries.
        session_chars = self.db.get_session_characters(self.session_id)
        char_data = []
        previous_summaries = ""
        for char_name in session_chars:
            meta = self.db.get_character_metadata(self.session_id, char_name)
            if meta and meta.is_npc:
                continue

            traits = ""
            if char_name in self.characters:
                traits = self.characters[char_name].fixed_traits

            appearance = self.db.get_character_appearance(self.session_id, char_name)
            location = self.db.get_character_location(self.session_id, char_name) or ""

            char_data.append({
                "traits": traits,
                "appearance": appearance,
                "location": location,
                "name": char_name
            })

            # Gather previous summaries for each non-NPC character.
            summaries = self.db.get_all_summaries(self.session_id, char_name)
            if summaries:
                previous_summaries += f"{char_name} summaries: " + " ".join(summaries) + "\n"

        # Gather last few lines of visible messages from the session for extra context.
        all_msgs = self.db.get_messages(self.session_id)
        recent_lines = []
        for m in all_msgs[-5:]:
            if m["message_type"] in ("character", "user"):
                recent_lines.append(f"{m['sender']}: {m['message']}")
        recent_dialogue_text = "\n".join(recent_lines)

        # Generate only the system_prompt.txt and user_prompt.txt files; no LLM calls.
        await self.image_manager.generate_concise_description(
            setting=setting_desc,
            moral_guidelines=guidelines,
            non_npc_characters=char_data,
            recent_dialogue=recent_dialogue_text,
            previous_summaries=previous_summaries
        )

    def start_automatic_chat(self):
        self.automatic_running = True

    def stop_automatic_chat(self):
        self.automatic_running = False

    async def check_summarization(self):
        all_msgs = self.db.get_messages(self.session_id)
        if not all_msgs:
            return

        participants = set(m["sender"] for m in all_msgs if m["message_type"] in ["user", "character"])
        for char_name in participants:
            if char_name not in self.db.get_session_characters(self.session_id):
                continue
            visible_for_char = self.db.get_visible_messages_for_character(self.session_id, char_name)
            if len(visible_for_char) >= self.summarization_threshold:
                await self.summarize_history_for_character(char_name)

    async def summarize_history_for_character(self, character_name: str):
        summarize_llm = OllamaClient('src/multipersona_chat_app/config/llm_config.yaml')
        if self.llm_client:
            summarize_llm.set_user_selected_model(self.llm_client.user_selected_model)

        if self.llm_status_callback:
            await self.llm_status_callback(
                f"[LLM] Summarizing history for {character_name}..."
            )

        while True:
            msgs = self.db.get_visible_messages_for_character(self.session_id, character_name)
            if len(msgs) < self.summarization_threshold:
                break

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"[LLM] Summarizing a chunk of {len(msgs[: self.to_summarize_count])} messages for {character_name}."
                )

            msgs.sort(key=lambda x: x['id'])
            chunk = msgs[: self.to_summarize_count]
            chunk_ids = [m['id'] for m in chunk]

            history_lines = []
            max_message_id_in_chunk = 0
            for m in chunk:
                mid = m["id"]
                sender = m["sender"]
                message = m["message"]
                emotion = m.get("emotion", None)
                thoughts = m.get("thoughts", None)
                if mid > max_message_id_in_chunk:
                    max_message_id_in_chunk = mid

                if sender in self.db.get_session_characters(self.session_id):
                    line_parts = [f"{sender}:"]
                    if emotion:
                        line_parts.append(f"(Emotion={emotion})")
                    if thoughts:
                        line_parts.append(f"(Thoughts={thoughts})")
                    line_parts.append(f"Message={message}")
                    line = " | ".join(line_parts)
                else:
                    line = f"{sender}: {message}"

                history_lines.append(line)

            plan_changes_notes = []
            plan_changes = self.db.get_plan_changes_for_range(
                self.session_id,
                character_name,
                0,
                max_message_id_in_chunk
            )
            for pc in plan_changes:
                note = ""
                if 'why_new_plan_goal' in pc and pc['why_new_plan_goal']:
                    note = f"Reason for plan change: {pc['why_new_plan_goal']}"
                plan_changes_notes.append(note)

            plan_changes_text = (
                "\n\nAdditionally, the following plan changes occurred:\n" + "\n".join(plan_changes_notes)
            ) if plan_changes_notes else ""

            history_text = "\n".join(history_lines) + plan_changes_text

            prompt = SUMMARIZE_PROMPT.format(
                character_name=character_name,
                history_text=history_text,
                moral_guidelines=self.moral_guidelines
            )

            summary = await asyncio.to_thread(summarize_llm.generate, prompt=prompt, use_cache=False)
            if not summary:
                summary = "No significant new events."

            self.db.save_new_summary(self.session_id, character_name, summary, max_message_id_in_chunk)
            self.db.hide_messages_for_character(self.session_id, character_name, chunk_ids)

            logger.info(
                f"Summarized and concealed a block of {len(chunk)} messages for '{character_name}'. "
                f"Newest remaining count: {len(self.db.get_visible_messages_for_character(self.session_id, character_name))}."
            )

        await self.combine_summaries_if_needed(character_name)
        if self.llm_status_callback:
            await self.llm_status_callback(f"[LLM] Finished summarizing history for {character_name}.")

    async def combine_summaries_if_needed(self, character_name: str):
        summarize_llm = OllamaClient('src/multipersona_chat_app/config/llm_config.yaml')
        if self.llm_client:
            summarize_llm.set_user_selected_model(self.llm_client.user_selected_model)

        while True:
            all_summary_records = self.db.get_all_summaries_records(self.session_id, character_name)
            if len(all_summary_records) < self.summary_of_summaries_count:
                break

            chunk = all_summary_records[: self.summary_of_summaries_count]
            chunk_ids = [r["id"] for r in chunk]
            chunk_summaries = [r["summary"] for r in chunk]
            covered_up_to = max(r["covered_up_to_message_id"] for r in chunk)

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"[LLM] Combining older summaries for {character_name} into one..."
                )

            prompt = COMBINE_SUMMARIES_PROMPT.format(
                character_name=character_name,
                count_summaries=len(chunk_summaries),
                chunk_text="\n\n".join(chunk_summaries),
                moral_guidelines=self.moral_guidelines
            )

            combined_summary = await asyncio.to_thread(summarize_llm.generate, prompt=prompt, use_cache=False)
            if not combined_summary:
                combined_summary = "No significant combined summary."

            self.db.save_new_summary(self.session_id, character_name, combined_summary, covered_up_to)
            self.db.delete_summaries_by_ids(self.session_id, character_name, chunk_ids)

            logger.info(
                f"Combined {len(chunk_summaries)} summaries into one for '{character_name}'. "
                f"Deleted old summary records: {chunk_ids}."
            )

            if self.llm_status_callback:
                await self.llm_status_callback(f"[LLM] Done combining summaries for {character_name}.")

    def get_latest_dialogue(self, character_name: str) -> str:
        visible_history = self.get_visible_history_for_character(character_name)
        recent_msgs = visible_history[-self.recent_dialogue_lines:]
        formatted_dialogue_lines = []
        for msg in recent_msgs:
            if msg['message_type'] == 'user':
                line = f"User: {msg['message']}"
            else:
                line = f"{msg['sender']}: {msg['message']}"
            formatted_dialogue_lines.append(line)
        return "\n".join(formatted_dialogue_lines)

    def build_prompt_for_character(self, character_name: str) -> Tuple[str, str]:
        existing_prompts = self.db.get_character_prompts(self.session_id, character_name)
        if not existing_prompts:
            raise ValueError(f"Existing prompts not found in the session for '{character_name}'.")

        system_prompt = existing_prompts['character_system_prompt']
        system_prompt = system_prompt.format(moral_guidelines=self.moral_guidelines)

        dynamic_prompt_template = existing_prompts['dynamic_prompt_template']

        latest_dialogue = self.get_latest_dialogue(character_name)
        latest_single_utterance = self.get_latest_single_utterance(character_name)
        all_summaries = self.db.get_all_summaries(self.session_id, character_name)
        chat_history_summary = "\n\n".join(all_summaries) if all_summaries else ""

        setting_description = self.current_setting_description or "A tranquil environment."
        location = self.get_combined_location()
        current_appearance = self.db.get_character_appearance(self.session_id, character_name)
        plan_obj = self.get_character_plan(character_name)
        steps_text = "\n".join(f"- {s}" for s in plan_obj.steps)
        plan_text = f"Goal: {plan_obj.goal}\nSteps:\n{steps_text}"

        try:
            formatted_prompt = dynamic_prompt_template.replace("{setting}", setting_description)
            formatted_prompt = formatted_prompt.replace("{chat_history_summary}", chat_history_summary)
            formatted_prompt = formatted_prompt.replace("{latest_dialogue}", latest_dialogue)
            formatted_prompt = formatted_prompt.replace("{latest_single_utterance}", latest_single_utterance)
            formatted_prompt = formatted_prompt.replace("{current_location}", location)
            formatted_prompt = formatted_prompt.replace("{current_appearance}", current_appearance)
            formatted_prompt = formatted_prompt.replace("{character_plan}", plan_text)
        except Exception as e:
            logger.error(f"Error replacing placeholders in dynamic_prompt_template: {e}")
            raise

        logger.debug(f"Built dynamic prompt for '{character_name}' turn.")
        return system_prompt, formatted_prompt

    def build_introduction_prompts_for_character(self, character_name: str) -> Tuple[str, str]:
        if character_name in self.characters:
            char_obj = self.characters[character_name]
            plan_obj = self.get_character_plan(character_name)
            plan_first_step = plan_obj.steps[0] if plan_obj.steps else ""

            system_prompt = CHARACTER_INTRODUCTION_SYSTEM_PROMPT_TEMPLATE.format(
                character_name=char_obj.name,
                character_description=char_obj.character_description,
                fixed_traits=char_obj.fixed_traits,
                appearance=char_obj.appearance,
                plan_first_step=plan_first_step,
                moral_guidelines=self.moral_guidelines
            )

            visible_history = self.db.get_visible_messages_for_character(self.session_id, character_name)
            if visible_history:
                last_msg = visible_history[-1]
                latest_text = f"{last_msg['sender']}: {last_msg['message']}"
            else:
                latest_text = ""

            all_summaries = self.db.get_all_summaries(self.session_id, character_name)
            chat_history_summary = "\n\n".join(all_summaries) if all_summaries else ""

            setting_description = self.current_setting_description or ""
            session_loc = self.db.get_current_location(self.session_id) or ""
            if not session_loc and self.current_setting in self.settings:
                session_loc = self.settings[self.current_setting].get('start_location', '')

            user_prompt = INTRODUCTION_TEMPLATE.format(
                name=character_name,
                character_name=character_name,
                appearance=char_obj.appearance,
                character_description=char_obj.character_description,
                fixed_traits=char_obj.fixed_traits,
                setting=setting_description,
                location=session_loc,
                chat_history_summary=chat_history_summary,
                latest_dialogue=latest_text,
                current_appearance=self.db.get_character_appearance(self.session_id, character_name),
                plan_first_step=plan_first_step
            )
            return system_prompt, user_prompt
        else:
            prompts = self.db.get_character_prompts(self.session_id, character_name)
            if not prompts:
                raise ValueError(f"No introduction or dynamic prompts found for NPC '{character_name}'.")

            npc_metadata = self.db.get_character_metadata(self.session_id, character_name)
            npc_description = getattr(npc_metadata, "role", "") if npc_metadata else ""
            npc_appearance = self.db.get_character_appearance(self.session_id, character_name) or ""
            plan_obj = self.get_character_plan(character_name)
            plan_first_step = plan_obj.steps[0] if plan_obj.steps else ""

            system_prompt = CHARACTER_INTRODUCTION_SYSTEM_PROMPT_TEMPLATE.format(
                character_name=character_name,
                character_description=npc_description,
                fixed_traits="",
                appearance=npc_appearance,
                plan_first_step=plan_first_step,
                moral_guidelines=self.moral_guidelines
            )

            all_summaries = self.db.get_all_summaries(self.session_id, character_name)
            chat_history_summary = "\n\n".join(all_summaries) if all_summaries else ""

            setting_description = self.current_setting_description or ""
            session_loc = self.db.get_current_location(self.session_id) or ""
            if not session_loc and self.current_setting in self.settings:
                session_loc = self.settings[self.current_setting].get('start_location', '')

            latest_text = ""
            visible_history = self.db.get_visible_messages_for_character(self.session_id, character_name)
            if visible_history:
                last_msg = visible_history[-1]
                latest_text = f"{last_msg['sender']}: {last_msg['message']}"

            user_prompt = INTRODUCTION_TEMPLATE.format(
                name=character_name,
                character_name=character_name,
                appearance=npc_appearance,
                character_description=npc_description,
                fixed_traits="",
                setting=setting_description,
                location=session_loc,
                chat_history_summary=chat_history_summary,
                latest_dialogue=latest_text,
                current_appearance=npc_appearance,
                plan_first_step=plan_first_step
            )
            return system_prompt, user_prompt

    def get_combined_location(self) -> str:
        char_locs = self.db.get_all_character_locations(self.session_id)
        char_apps = self.db.get_all_character_appearances(self.session_id)
        msgs = self.db.get_messages(self.session_id)
        participants = set(m["sender"] for m in msgs if m["message_type"] in ["user", "character"])

        if not participants:
            session_loc = self.db.get_current_location(self.session_id)
            if not session_loc and self.current_setting in self.settings:
                session_loc = self.settings[self.current_setting].get('start_location', '')
            if session_loc:
                return f"The setting is: {session_loc}"
            else:
                return "No characters present and no specific location known."

        parts = []
        for c_name in char_locs.keys():
            if c_name not in participants:
                continue
            c_loc = char_locs[c_name].strip()
            c_app = char_apps.get(c_name, "").strip()
            if not c_loc and not c_app:
                logger.warning(f"Character '{c_name}' has no known location or appearance.")
            elif c_loc and not c_app:
                parts.append(f"{c_name}'s location: {c_loc}")
            elif not c_loc and c_app:
                parts.append(f"{c_name} has appearance: {c_app}")
            else:
                parts.append(f"{c_name}'s location: {c_loc}, appearance: {c_app}")

        if not parts:
            session_loc = self.db.get_current_location(self.session_id)
            if not session_loc and self.current_setting in self.settings:
                session_loc = self.settings[self.current_setting].get('start_location', '')
            if session_loc:
                return f"The setting is: {session_loc}"
            else:
                return "No active character locations known."
        return " | ".join(parts)

    async def generate_character_message(self, character_name: str):
        logger.info(f"[generate_character_message] Generating message for '{character_name}'")
        all_msgs = self.db.get_messages(self.session_id)
        triggered_message_id = all_msgs[-1]['id'] if all_msgs else None

        await self.update_character_plan(character_name, triggered_message_id)

        if not self.character_has_introduced(character_name):
            logger.debug(f"[generate_character_message] '{character_name}' has not introduced themselves yet.")
            await self.generate_character_introduction_message(character_name)
            return

        try:
            if self.llm_status_callback:
                await self.llm_status_callback(f"[LLM] Generating normal turn interaction for {character_name}.")

            system_prompt, formatted_prompt = self.build_prompt_for_character(character_name)

            llm = OllamaClient('src/multipersona_chat_app/config/llm_config.yaml', output_model=Interaction)
            if self.llm_client:
                llm.set_user_selected_model(self.llm_client.user_selected_model)

            if self.llm_status_callback:
                await self.llm_status_callback(f"[LLM] Prompting LLM for {character_name}'s action/dialogue.")

            interaction = await asyncio.to_thread(
                llm.generate,
                prompt=formatted_prompt,
                system=system_prompt,
                use_cache=False
            )

            logger.debug(f"[generate_character_message] Raw LLM output for '{character_name}': {interaction}")

            if not interaction:
                logger.warning(f"No response from LLM for {character_name}.")
                if self.llm_status_callback:
                    await self.llm_status_callback(f"No interaction generated for {character_name}.")
                return

            if not isinstance(interaction, Interaction):
                logger.error(f"Malformed LLM output (not Interaction) for '{character_name}': {interaction}")
                if self.llm_status_callback:
                    await self.llm_status_callback(f"Invalid LLM output for {character_name}.")
                return

            interaction.action = utils.remove_markdown(interaction.action)
            interaction.dialogue = utils.remove_markdown(interaction.dialogue)
            interaction.emotion = utils.remove_markdown(interaction.emotion)
            interaction.thoughts = utils.remove_markdown(interaction.thoughts)

            final_interaction = await self.check_and_regenerate_if_repetitive(
                character_name, system_prompt, formatted_prompt, interaction
            )
            if not final_interaction:
                logger.warning(f"All regeneration attempts failed for {character_name}. Using original output.")
                final_interaction = interaction

            final_interaction.action = utils.remove_markdown(final_interaction.action)
            final_interaction.dialogue = utils.remove_markdown(final_interaction.dialogue)
            final_interaction.emotion = utils.remove_markdown(final_interaction.emotion)
            final_interaction.thoughts = utils.remove_markdown(final_interaction.thoughts)

            if (not final_interaction.action.strip()) and (not final_interaction.dialogue.strip()):
                logger.warning(
                    f"'{character_name}' returned empty action AND dialogue."
                )

            if final_interaction.action.strip() and final_interaction.dialogue.strip():
                formatted_message = f"*{final_interaction.action}*\n{final_interaction.dialogue}"
            elif final_interaction.action.strip():
                formatted_message = f"*{final_interaction.action}*"
            elif final_interaction.dialogue.strip():
                formatted_message = final_interaction.dialogue
            else:
                formatted_message = "None"

            msg_id = await self.add_message(
                character_name,
                formatted_message,
                visible=True,
                message_type="character",
                emotion=final_interaction.emotion,
                thoughts=final_interaction.thoughts
            )

            logger.info(f"Stored message for '{character_name}': {formatted_message}")

            location_was_triggered = False
            appearance_was_triggered = False

            if final_interaction.location_change_expected:
                logger.debug(f"{character_name} indicated location change.")
                await self.evaluate_location_update(character_name)
                location_was_triggered = True

            if final_interaction.appearance_change_expected:
                logger.debug(f"{character_name} indicated appearance change.")
                await self.evaluate_appearance_update(character_name)
                appearance_was_triggered = True

            if not location_was_triggered and not appearance_was_triggered:
                self.msg_counter_since_forced_update[character_name] = \
                    self.msg_counter_since_forced_update.get(character_name, 0) + 1

                if self.msg_counter_since_forced_update[character_name] >= self.forced_update_interval:
                    logger.info(
                        f"Forcing location & appearance update for '{character_name}' "
                        f"after {self.forced_update_interval} messages."
                    )
                    await self.evaluate_location_update(character_name)
                    await self.evaluate_appearance_update(character_name)
                    self.msg_counter_since_forced_update[character_name] = 0
            else:
                self.msg_counter_since_forced_update[character_name] = 0

            if self.llm_status_callback:
                await self.llm_status_callback(f"[LLM] Finished generating interaction for {character_name}.")

        except Exception as e:
            logger.error(f"Error generating message for {character_name}: {e}", exc_info=True)
            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Error generating an interaction for {character_name}."
                )

    async def generate_character_introduction_message(self, character_name: str):
        logger.info(f"Building introduction for character: {character_name}")
        system_prompt, user_prompt = self.build_introduction_prompts_for_character(character_name)

        introduction_llm_client = OllamaClient(
            'src/multipersona_chat_app/config/llm_config.yaml',
            output_model=CharacterIntroductionOutput
        )
        if self.llm_client:
            introduction_llm_client.set_user_selected_model(self.llm_client.user_selected_model)

        try:
            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Generating an initial introduction for {character_name}..."
                )

            introduction_response = await asyncio.to_thread(
                introduction_llm_client.generate,
                prompt=user_prompt,
                system=system_prompt
            )
            if isinstance(introduction_response, CharacterIntroductionOutput):
                intro_text = introduction_response.introduction_text.strip()
                app_seg = introduction_response.current_appearance
                cur_loc = introduction_response.current_location
                

                msg_id = await self.add_message(
                    character_name,
                    intro_text,
                    visible=True,
                    message_type="character"
                )

                new_app_segments = AppearanceSegments(
                    hair=app_seg.hair,
                    clothing=app_seg.clothing,
                    accessories_and_held_items=app_seg.accessories_and_held_items,
                    posture_and_body_language=app_seg.posture_and_body_language,
                    facial_expression=app_seg.facial_expression,
                    other_relevant_details=app_seg.other_relevant_details
                )

                triggered_message_id = msg_id if msg_id else None
                self.db.update_character_appearance(
                    self.session_id,
                    character_name,
                    new_app_segments,
                    triggered_message_id
                )
                self.db.update_character_location(self.session_id, character_name, cur_loc, triggered_message_id)
                logger.info(f"Saved introduction message for {character_name}.")
                if self.llm_status_callback:
                    await self.llm_status_callback(
                        f"Introduction completed for {character_name}."
                    )
                self._introduction_given[character_name] = True
            else:
                logger.warning(f"Invalid or empty introduction for {character_name}: {introduction_response}")
                if self.llm_status_callback:
                    await self.llm_status_callback(
                        f"An invalid introduction was returned for {character_name}."
                    )

        except Exception as e:
            logger.error(f"Error generating introduction for {character_name}: {e}", exc_info=True)
            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Error generating the introduction for {character_name}."
                )

    async def update_character_plan(self, character_name: str, triggered_message_id: Optional[int] = None):
        plan_client = OllamaClient(
            config_path='src/multipersona_chat_app/config/llm_config.yaml',
            output_model=CharacterPlan
        )
        if self.llm_client:
            plan_client.set_user_selected_model(self.llm_client.user_selected_model)

        existing_plan = self.get_character_plan(character_name)
        old_goal = existing_plan.goal.strip()
        old_steps = [s.strip() for s in existing_plan.steps if s.strip()]
        old_why = existing_plan.why_new_plan_goal.strip()

        my_appearance = self.db.get_character_appearance(self.session_id, character_name)
        others_appearance = self.db.get_characters_appearance_except_one(self.session_id, character_name)
        # Only include others who have introduced themselves (spoken at least once).
        others_appearance = {
            c: a for c, a in others_appearance.items() if self.character_has_introduced(c)
        }

        if character_name in self.characters:
            character_description = self.characters[character_name].character_description
        else:
            character_description = ""

        system_prompt = PLAN_UPDATE_SYSTEM_PROMPT.format(
            character_name=character_name,
            moral_guidelines=self.moral_guidelines
        )
        user_prompt = PLAN_UPDATE_USER_PROMPT.format(
            character_name=character_name,
            character_description=character_description,
            old_goal=old_goal,
            old_steps=old_steps,
            current_setting=self.current_setting or "(No current setting)",
            combined_location=self.get_combined_location(),
            my_appearance=my_appearance,
            others_appearance=others_appearance,
            latest_dialogue=self.get_latest_dialogue(character_name)
        )

        try:
            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Generating or updating the plan for {character_name}..."
                )

            plan_result = await asyncio.to_thread(
                plan_client.generate,
                prompt=user_prompt,
                system=system_prompt,
                use_cache=False
            )

            # Attempt to parse the first plan update result:
            new_plan: Optional[CharacterPlan] = None
            if plan_result:
                try:
                    if isinstance(plan_result, CharacterPlan):
                        new_plan = plan_result
                    else:
                        # Attempt JSON parsing into the CharacterPlan model:
                        new_plan = CharacterPlan.model_validate_json(plan_result)
                except Exception as e2:
                    logger.error(
                        f"Failed to parse new plan for '{character_name}' on first attempt. "
                        f"Error: {e2}"
                    )
                    new_plan = None

            # ---------------
            # If there's no *previous* plan (old_goal & old_steps are empty)
            # AND the newly generated plan is also effectively empty,
            # then we do a retry loop with a special instruction about being alone/with others.
            # ---------------
            plan_retry_amount = self.config.get("plan_retry_amount", 2)
            def plan_is_effectively_empty(plan_obj: CharacterPlan) -> bool:
                """Check if a CharacterPlan has no real goal and no real steps."""
                has_goal = bool(plan_obj.goal.strip())
                has_steps = any(s.strip() for s in plan_obj.steps)
                return (not has_goal) and (not has_steps)

            need_special_retry = False
            if not old_goal and not old_steps:
                # No previous plan
                if (not new_plan) or plan_is_effectively_empty(new_plan):
                    # LLM also returned no plan
                    need_special_retry = True

            if need_special_retry:
                # Figure out if the character is alone or with others who have spoken.
                messages = self.db.get_messages(self.session_id)
                other_speakers = set(
                    m['sender'] for m in messages
                    if m['message_type'] in ('character', 'user')
                    and m['sender'] != character_name
                )
                # Only keep those who have introduced themselves.
                other_speakers = {c for c in other_speakers if self.character_has_introduced(c)}

                if len(other_speakers) == 0:
                    special_context_line = (
                        f"You are currently alone in this setting. "
                        f"Please reflect on what {character_name} might want to do, explore, or accomplish. "
                        f"Provide a goal, some steps to achieve it, and explain briefly why this plan matters."
                    )
                else:
                    others_list = ", ".join(sorted(other_speakers))
                    special_context_line = (
                        f"You are with these other active participants: {others_list}. "
                        f"Please consider how {character_name} interacts with them or the environment. "
                        f"Provide a goal, steps to pursue that goal, and a short reason why this plan."
                    )

                # We will attempt plan_retry_amount times to get a non-empty plan.
                for attempt_idx in range(plan_retry_amount):
                    if self.llm_status_callback:
                        await self.llm_status_callback(
                            f"Special plan retry {attempt_idx + 1} for {character_name} (no prior plan)."
                        )

                    # Build a more direct prompt:
                    special_user_prompt = (
                        user_prompt
                        + "\n\n[NOTE]\n"
                        + "No plan was found previously. " + special_context_line
                    )

                    plan_retry_result = await asyncio.to_thread(
                        plan_client.generate,
                        prompt=special_user_prompt,
                        system=system_prompt,
                        use_cache=False
                    )
                    parsed_retry_plan: Optional[CharacterPlan] = None
                    if plan_retry_result:
                        try:
                            if isinstance(plan_retry_result, CharacterPlan):
                                parsed_retry_plan = plan_retry_result
                            else:
                                parsed_retry_plan = CharacterPlan.model_validate_json(plan_retry_result)
                        except Exception:
                            parsed_retry_plan = None

                    if parsed_retry_plan and not plan_is_effectively_empty(parsed_retry_plan):
                        # We finally got a workable plan
                        new_plan = parsed_retry_plan
                        break

                # If after all retries we still have nothing, new_plan remains None or effectively empty.

            # ---------------
            # If new_plan is still None or empty, we keep the old plan and return.
            # Otherwise, proceed with normal storing in DB
            # ---------------
            if not new_plan:
                logger.warning("Plan update returned no valid result. Keeping existing plan (if any).")
                if self.llm_status_callback:
                    await self.llm_status_callback(
                        f"No plan update was generated for {character_name}; retaining old plan."
                    )
                return
            else:
                new_goal = new_plan.goal.strip()
                new_steps = [s.strip() for s in new_plan.steps if s.strip()]
                new_why = new_plan.why_new_plan_goal.strip()

                # If still empty, keep the old plan and return:
                if (not new_goal) and (not new_steps):
                    logger.warning("Final plan remains empty after retries. Keeping existing plan.")
                    return

                # Now store or track the plan changes if any
                if (new_goal != old_goal) or (new_steps != old_steps) or (new_why != old_why):
                    change_explanation = self.build_plan_change_summary(old_goal, old_steps, new_goal, new_steps)
                    if new_why:
                        change_explanation += f" Additional reason: {new_why}"

                    self.db.save_character_plan_with_history(
                        self.session_id,
                        character_name,
                        new_goal,
                        new_steps,
                        new_why,
                        triggered_message_id,
                        change_explanation
                    )
                else:
                    self.db.save_character_plan_with_history(
                        self.session_id,
                        character_name,
                        new_goal,
                        new_steps,
                        new_why,
                        triggered_message_id,
                        "No change in plan"
                    )

                if self.llm_status_callback:
                    await self.llm_status_callback(
                        f"Plan updated for {character_name} (Goal: {new_goal})."
                    )

        except Exception as e:
            logger.error(f"Error generating plan for {character_name}: {e}", exc_info=True)
            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"An error occurred while generating/updating the plan for {character_name}."
                )

    def build_plan_change_summary(self, old_goal: str, old_steps: List[str], new_goal: str, new_steps: List[str]) -> str:
        changes = []
        if old_goal != new_goal:
            changes.append(f"Goal changed from '{old_goal}' to '{new_goal}'.")
        elif old_steps != new_steps:
            set_old = set(old_steps)
            set_new = set(new_steps)
            intersection = set_old.intersection(set_new)
            union = set_old.union(set_new)
            jaccard_similarity = len(intersection) / float(len(union)) if union else 1.0
            if jaccard_similarity < 0.5:
                changes.append(f"Plan steps have changed significantly to: {new_steps}.")
        return " ".join(changes)

    async def check_and_regenerate_if_repetitive(
        self,
        character_name: str,
        system_prompt: str,
        dynamic_prompt: str,
        interaction: Interaction
    ) -> Optional[Interaction]:
        """
        Checks the generated interaction (action + dialogue) for excessive similarity
        to the character's own recent messages and between its own fields.
        If violations are detected, regeneration is attempted up to
        `max_similarity_retries` times. If no candidate passes all checks,
        the candidate with the lowest similarity (i.e. most dissimilar) score is returned.
        """
        all_visible = self.db.get_visible_messages_for_character(self.session_id, character_name)
        same_speaker_lines = [m for m in all_visible if m["sender"] == character_name]
        recent_speaker_lines = same_speaker_lines[-5:] if len(same_speaker_lines) > 5 else same_speaker_lines

        embed_client = OllamaClient('src/multipersona_chat_app/config/llm_config.yaml')
        if self.llm_client:
            embed_client.set_user_selected_model(self.llm_client.user_selected_model)

        tries = 0
        max_tries = self.max_similarity_retries
        current_interaction = interaction

        best_candidate = current_interaction
        best_score = float('inf')

        while True:
            action_text = current_interaction.action
            dialogue_text = current_interaction.dialogue

            violation_detected = False
            violation_reasons = []

            # 1) Check for any real alphanumeric content.
            if (not re.search(r'[A-Za-z0-9]', action_text)) and (not re.search(r'[A-Za-z0-9]', dialogue_text)):
                logger.warning(
                    f"[{character_name}] Check failed: no alphanumeric content in both action and dialogue."
                )
                violation_detected = True
                violation_reasons.append(
                    "Your action and dialogue both seem empty or non-descriptive."
                )

            # 2) Check for placeholder angle brackets.
            if "<" in action_text or ">" in action_text or "<" in dialogue_text or ">" in dialogue_text:
                logger.warning(
                    f"[{character_name}] Check failed: angle bracket placeholder(s) detected."
                )
                violation_detected = True
                violation_reasons.append(
                    "Remove any placeholder angle brackets and replace them with a proper description."
                )

            # 3) Check if dialogue is exactly repeated in action.
            if dialogue_text.strip() and dialogue_text.strip() in action_text:
                logger.warning(
                    f"[{character_name}] Check failed: dialogue text repeated within the action field."
                )
                violation_detected = True
                violation_reasons.append(
                    "Your action text should not simply restate the entire dialogue."
                )

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"[RepetitionCheck] Computing embeddings (attempt {tries + 1}) for {character_name}."
                )

            action_embedding = embed_client.get_embedding(action_text)
            dialogue_embedding = embed_client.get_embedding(dialogue_text)
            combined_text = action_text + " " + dialogue_text
            actiondialogue_embedding = embed_client.get_embedding(combined_text)

            # 4) Compare action vs. dialogue similarity.
            cos_sim_action_dialogue = embed_client.compute_cosine_similarity(action_embedding, dialogue_embedding)
            jac_sim_action_dialogue = embed_client.compute_jaccard_similarity(action_text, dialogue_text)
            logger.debug(
                f"[{character_name}] Comparing action vs dialogue => "
                f"cos={cos_sim_action_dialogue:.3f}, jac={jac_sim_action_dialogue:.3f}"
            )
            if (cos_sim_action_dialogue >= self.similarity_threshold or
                jac_sim_action_dialogue >= self.similarity_threshold):
                logger.warning(
                    f"[{character_name}] Check failed: action and dialogue too similar "
                    f"(cos={cos_sim_action_dialogue:.3f}, jac={jac_sim_action_dialogue:.3f})."
                )
                violation_detected = True
                violation_reasons.append(
                    "Your action and dialogue read almost the same. Please differentiate them."
                )

            # 5) Compare current action/dialogue to recent lines from the same speaker.
            for line_obj in recent_speaker_lines:
                old_msg = line_obj["message"]
                old_embedding = embed_client.get_embedding(old_msg)

                cos_sim_action = embed_client.compute_cosine_similarity(action_embedding, old_embedding)
                cos_sim_dialogue = embed_client.compute_cosine_similarity(dialogue_embedding, old_embedding)
                cos_sim_both = embed_client.compute_cosine_similarity(actiondialogue_embedding, old_embedding)

                jac_sim_action = embed_client.compute_jaccard_similarity(action_text, old_msg)
                jac_sim_dialogue = embed_client.compute_jaccard_similarity(dialogue_text, old_msg)
                jac_sim_both = embed_client.compute_jaccard_similarity(combined_text, old_msg)

                logger.debug(
                    f"[{character_name}] Comparing current action/dialogue to a recent line:\n"
                    f"  Current vs old_msg='{old_msg[:60]}...' => "
                    f"cos_action={cos_sim_action:.3f}, cos_dialogue={cos_sim_dialogue:.3f}, "
                    f"cos_combined={cos_sim_both:.3f} | "
                    f"jac_action={jac_sim_action:.3f}, jac_dialogue={jac_sim_dialogue:.3f}, "
                    f"jac_combined={jac_sim_both:.3f}"
                )

                if (cos_sim_action >= self.similarity_threshold or
                    jac_sim_action >= self.similarity_threshold or
                    cos_sim_dialogue >= self.similarity_threshold or
                    jac_sim_dialogue >= self.similarity_threshold or
                    cos_sim_both >= self.similarity_threshold or
                    jac_sim_both >= self.similarity_threshold):
                    logger.warning(
                        f"[{character_name}] Check failed: output is too similar to a recent line "
                        f"(cos or jac >= {self.similarity_threshold})."
                    )
                    violation_detected = True
                    violation_reasons.append(
                        "Your new message closely duplicates your own recent statements. Please vary it."
                    )
                    break

            # Compute a similarity score for the current candidate (lower is better).
            candidate_score = max(cos_sim_action_dialogue, jac_sim_action_dialogue)
            for line_obj in recent_speaker_lines:
                old_msg = line_obj["message"]
                old_embedding = embed_client.get_embedding(old_msg)
                score_line = max(
                    embed_client.compute_cosine_similarity(action_embedding, old_embedding),
                    embed_client.compute_cosine_similarity(dialogue_embedding, old_embedding),
                    embed_client.compute_cosine_similarity(actiondialogue_embedding, old_embedding),
                    embed_client.compute_jaccard_similarity(action_text, old_msg),
                    embed_client.compute_jaccard_similarity(dialogue_text, old_msg),
                    embed_client.compute_jaccard_similarity(combined_text, old_msg)
                )
                candidate_score = max(candidate_score, score_line)

            if candidate_score < best_score:
                best_score = candidate_score
                best_candidate = current_interaction

            if not violation_detected:
                logger.debug(f"[{character_name}] All repetition checks passed; output is acceptable.")
                return current_interaction

            tries += 1
            if tries > max_tries:
                logger.warning(
                    f"[{character_name}] Exceeded maximum regeneration attempts ({max_tries}). "
                    f"Returning best candidate with similarity score {best_score:.3f}."
                )
                return best_candidate

            appended_warning = (
                "\nBelow are the issues we found:\n- " + "\n- ".join(violation_reasons) +
                "\n\nPlease revise your response to avoid these issues. "
                "Improve or vary the wording so it's not overly similar to previous statements, "
                "doesn't repeat dialogue in the action, and doesn't include placeholders.\n"
            )

            logger.info(
                f"[{character_name}] Violations: {violation_reasons} => Attempting regeneration #{tries}."
            )

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"[RepetitionCheck] Attempting regeneration {tries} for {character_name} due to similarity issues."
                )

            regen_client = OllamaClient(
                'src/multipersona_chat_app/config/llm_config.yaml',
                output_model=Interaction
            )
            if self.llm_client:
                regen_client.set_user_selected_model(self.llm_client.user_selected_model)

            revised_prompt = dynamic_prompt + appended_warning
            new_interaction = await asyncio.to_thread(
                regen_client.generate,
                prompt=revised_prompt,
                system=system_prompt,
                use_cache=False
            )

            if not new_interaction or not isinstance(new_interaction, Interaction):
                logger.warning(
                    f"[{character_name}] Regeneration attempt returned invalid output. Stopping."
                )
                return best_candidate

            current_interaction = new_interaction
            # Loop again to re-check the new interaction.


    def character_has_introduced(self, candidate_char_name: str) -> bool:
        return self._introduction_given.get(candidate_char_name, False)

    async def evaluate_location_update(self, character_name: str, visited: Optional[Set[str]] = None):
        if visited is None:
            visited = set()
        if character_name in visited:
            return
        visited.add(character_name)

        try:
            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Evaluating location update for {character_name}."
                )

            location_llm = OllamaClient(
                'src/multipersona_chat_app/config/llm_config.yaml',
                output_model=LocationUpdate
            )
            if self.llm_client:
                location_llm.set_user_selected_model(self.llm_client.user_selected_model)

            system_prompt = LOCATION_UPDATE_SYSTEM_PROMPT.format(
                character_name=character_name,
                moral_guidelines=self.moral_guidelines
            )
            dynamic_context = self.build_location_update_context(character_name)

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Prompting location update LLM for {character_name}."
                )

            update_response = await asyncio.to_thread(
                location_llm.generate,
                prompt=dynamic_context,
                system=system_prompt,
                use_cache=False
            )

            if not update_response or not isinstance(update_response, LocationUpdate):
                logger.info(f"No location update or invalid data for '{character_name}'.")
                return

            new_loc = (update_response.location or "").strip()
            transition_action = utils.remove_markdown((update_response.transition_action or "").strip())
            current_location = self.db.get_character_location(self.session_id, character_name) or ""

            if new_loc == current_location:
                new_loc = ""
                transition_action = ""

            if transition_action:
                await self.add_message(
                    character_name,
                    f"*{transition_action}*",
                    visible=True,
                    message_type="character"
                )

            if new_loc:
                messages = self.db.get_messages(self.session_id)
                last_msg = messages[-1] if messages else None
                triggered_id = last_msg['id'] if last_msg else None
                updated = self.db.update_character_location(self.session_id, character_name, new_loc, triggered_id)
                if updated:
                    logger.info(f"Location updated for {character_name} to '{new_loc}'. Rationale: {update_response.rationale}")
            else:
                logger.info(f"No location change for {character_name}. Rationale: {update_response.rationale}")

            for other_char in update_response.other_characters:
                if not other_char:
                    continue
                if other_char not in self.db.get_session_characters(self.session_id):
                    logger.info(f"Ignoring 'other_characters' entry: {other_char} not in session.")
                    continue
                if other_char not in visited:
                    await self.evaluate_location_update(other_char, visited=visited)

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Done checking location update for {character_name}."
                )

        except Exception as e:
            logger.error(f"Error in evaluate_location_update for {character_name}: {e}", exc_info=True)
            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Error while evaluating location update for {character_name}."
                )

    def build_location_update_context(self, character_name: str) -> str:
        visible_history = self.db.get_visible_messages_for_character(self.session_id, character_name)
        relevant_lines = []
        for m in visible_history[-5:]:
            if m["sender"] in self.db.get_session_characters(self.session_id):
                relevant_lines.append(f"{m['sender']}: Message={m['message']}")
            else:
                relevant_lines.append(f"{m['sender']}: {m['message']}")

        location_so_far = self.db.get_character_location(self.session_id, character_name) or "(Unknown)"
        plan = self.get_character_plan(character_name)
        plan_text = f"Goal: {plan.goal}\nSteps: {plan.steps}\nWhy: {plan.why_new_plan_goal}"

        return LOCATION_UPDATE_CONTEXT_TEMPLATE.format(
            character_name=character_name,
            location_so_far=location_so_far,
            plan_text=plan_text,
            relevant_lines=json.dumps(relevant_lines, ensure_ascii=False, indent=2)
        )

    async def evaluate_appearance_update(self, character_name: str, visited: Optional[Set[str]] = None):
        if visited is None:
            visited = set()
        if character_name in visited:
            return
        visited.add(character_name)

        try:
            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Evaluating appearance update for {character_name}."
                )

            appearance_llm = OllamaClient(
                'src/multipersona_chat_app/config/llm_config.yaml',
                output_model=AppearanceUpdate
            )
            if self.llm_client:
                appearance_llm.set_user_selected_model(self.llm_client.user_selected_model)

            system_prompt = APPEARANCE_UPDATE_SYSTEM_PROMPT.format(
                character_name=character_name,
                moral_guidelines=self.moral_guidelines
            )
            dynamic_context = self.build_appearance_update_context(character_name)

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Prompting appearance update LLM for {character_name}."
                )

            update_response = await asyncio.to_thread(
                appearance_llm.generate,
                prompt=dynamic_context,
                system=system_prompt,
                use_cache=False
            )

            if not update_response or not isinstance(update_response, AppearanceUpdate):
                logger.info(f"No appearance update or invalid data for '{character_name}'.")
                return

            new_segments = update_response.new_appearance
            transition_action = utils.remove_markdown((update_response.transition_action or "").strip())
            old_segments = self.db.get_current_appearance_segments(self.session_id, character_name)
            if old_segments is None:
                old_segments = {}

            hair_str = (new_segments.hair or "").strip()
            old_hair_str = (old_segments.get('hair') or "").strip()
            if hair_str == old_hair_str:
                new_segments.hair = ""

            clothing_str = (new_segments.clothing or "").strip()
            old_clothing_str = (old_segments.get('clothing') or "").strip()
            if clothing_str == old_clothing_str:
                new_segments.clothing = ""

            accessories_str = (new_segments.accessories_and_held_items or "").strip()
            old_accessories_str = (old_segments.get('accessories_and_held_items') or "").strip()
            if accessories_str == old_accessories_str:
                new_segments.accessories_and_held_items = ""

            posture_str = (new_segments.posture_and_body_language or "").strip()
            old_posture_str = (old_segments.get('posture_and_body_language') or "").strip()
            if posture_str == old_posture_str:
                new_segments.posture_and_body_language = ""

            facial_str = (new_segments.facial_expression or "").strip()
            old_facial_str = (old_segments.get('facial_expression') or "").strip()
            if facial_str == old_facial_str:
                new_segments.facial_expression = ""

            other_str = (new_segments.other_relevant_details or "").strip()
            old_other_str = (old_segments.get('other_relevant_details') or "").strip()
            if other_str == old_other_str:
                new_segments.other_relevant_details = ""

            something_changed = any([
                new_segments.hair,
                new_segments.clothing,
                new_segments.accessories_and_held_items,
                new_segments.posture_and_body_language,
                new_segments.facial_expression,
                new_segments.other_relevant_details
            ])

            if transition_action:
                await self.add_message(
                    character_name,
                    f"*{transition_action}*",
                    visible=True,
                    message_type="character"
                )

            if something_changed:
                messages = self.db.get_messages(self.session_id)
                last_msg = messages[-1] if messages else None
                triggered_id = last_msg['id'] if last_msg else None
                updated = self.db.update_character_appearance(self.session_id, character_name, new_segments, triggered_id)
                if updated:
                    logger.info(
                        f"Appearance updated for {character_name}. Rationale: {update_response.rationale} | "
                        f"Segments: {new_segments.dict()}"
                    )
            else:
                logger.info(f"No appearance change for {character_name}. Rationale: {update_response.rationale}")

            for other_char in update_response.other_characters:
                if not other_char:
                    continue
                if other_char not in self.db.get_session_characters(self.session_id):
                    logger.info(f"Ignoring 'other_characters' entry: {other_char} not in session.")
                    continue
                if other_char not in visited:
                    await self.evaluate_appearance_update(other_char, visited=visited)

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Done checking appearance update for {character_name}."
                )

        except Exception as e:
            logger.error(f"Error in evaluate_appearance_update for {character_name}: {e}", exc_info=True)
            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Error evaluating appearance update for {character_name}."
                )

    def build_appearance_update_context(self, character_name: str) -> str:
        visible_history = self.db.get_visible_messages_for_character(self.session_id, character_name)
        current_location = self.db.get_character_location(self.session_id, character_name) or "(Unknown)"
        relevant_lines = [f"{m['sender']}: {m['message']}" for m in visible_history[-7:]]
        old_appearance = self.db.get_character_appearance(self.session_id, character_name)
        plan = self.get_character_plan(character_name)
        next_steps = plan.steps[:1] if plan.steps else ["No immediate next steps."]
        plan_text = f"Next Steps: {next_steps}"

        return APPEARANCE_UPDATE_CONTEXT_TEMPLATE.format(
            character_name=character_name,
            current_location=current_location,
            old_appearance=old_appearance,
            plan_text=plan_text,
            relevant_lines=json.dumps(relevant_lines, ensure_ascii=False, indent=2)
        )

    async def update_all_characters_location_and_appearance_from_scratch(self):
        session_chars = self.db.get_session_characters(self.session_id)
        for c_name in session_chars:
            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Updating location & appearance from scratch for {c_name}."
                )

            if c_name in self.characters:
                char_obj = self.characters[c_name]
                character_description = char_obj.character_description
                fixed_traits = char_obj.fixed_traits
            else:
                character_description = ""
                fixed_traits = ""

            setting_name = self.current_setting or ""
            setting_data = self.settings.get(setting_name, {})
            setting_description = setting_data.get('description', "")
            start_location = setting_data.get('start_location', "")

            visible_msgs = self.db.get_visible_messages_for_character(self.session_id, c_name)
            visible_msgs.sort(key=lambda x: x['id'])
            lines_for_history = []
            for m in visible_msgs:
                sender = m["sender"]
                content = m["message"]
                if sender in self.db.get_session_characters(self.session_id):
                    lines_for_history.append(f"{sender}: Message={content}")
                else:
                    lines_for_history.append(f"{sender}: {content}")
            message_history = "\n".join(lines_for_history)

            all_summaries = self.db.get_all_summaries(self.session_id, c_name)
            summaries_text = "\n".join(all_summaries) if all_summaries else ""

            location_llm = OllamaClient(
                'src/multipersona_chat_app/config/llm_config.yaml',
                output_model=LocationFromScratch
            )
            if self.llm_client:
                location_llm.set_user_selected_model(self.llm_client.user_selected_model)

            location_system = LOCATION_FROM_SCRATCH_SYSTEM_PROMPT.format(
                setting_name=setting_name,
                setting_description=setting_description,
                start_location=start_location,
                character_name=c_name,
                character_description=character_description,
                fixed_traits=fixed_traits,
                message_history=message_history,
                summaries=summaries_text
            )
            location_user = LOCATION_FROM_SCRATCH_USER_PROMPT.format(character_name=c_name)

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Prompting from-scratch location derivation for {c_name}."
                )

            new_location_result = await asyncio.to_thread(
                location_llm.generate,
                prompt=location_user,
                system=location_system,
                use_cache=False
            )
            final_location = ""
            if new_location_result and isinstance(new_location_result, LocationFromScratch):
                final_location = new_location_result.location.strip()
            else:
                logger.warning(f"Failed to get structured location for {c_name}.")

            if final_location:
                messages = self.db.get_messages(self.session_id)
                last_msg = messages[-1] if messages else None
                triggered_id = last_msg['id'] if last_msg else None
                updated = self.db.update_character_location(self.session_id, c_name, final_location, triggered_id)
                if updated:
                    logger.info(f"[FromScratch] Location for {c_name} => '{final_location}'")

            appearance_llm = OllamaClient(
                'src/multipersona_chat_app/config/llm_config.yaml',
                output_model=AppearanceFromScratch
            )
            if self.llm_client:
                appearance_llm.set_user_selected_model(self.llm_client.user_selected_model)

            appearance_system = APPEARANCE_FROM_SCRATCH_SYSTEM_PROMPT.format(
                character_name=c_name,
                character_description=character_description,
                fixed_traits=fixed_traits,
                setting_description=setting_description,
                message_history=message_history,
                summaries=summaries_text
            )
            appearance_user = APPEARANCE_FROM_SCRATCH_USER_PROMPT.format(character_name=c_name)

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Prompting from-scratch appearance derivation for {c_name}."
                )

            new_appearance_result = await asyncio.to_thread(
                appearance_llm.generate,
                prompt=appearance_user,
                system=appearance_system,
                use_cache=False
            )
            if new_appearance_result and isinstance(new_appearance_result, AppearanceFromScratch):
                new_segments = AppearanceSegments(
                    hair=new_appearance_result.hair.strip(),
                    clothing=new_appearance_result.clothing.strip(),
                    accessories_and_held_items=new_appearance_result.accessories_and_held_items.strip(),
                    posture_and_body_language=new_appearance_result.posture_and_body_language.strip(),
                    facial_expression=new_appearance_result.facial_expression.strip(),
                    other_relevant_details=new_appearance_result.other_relevant_details.strip()
                )

                messages = self.db.get_messages(self.session_id)
                last_msg = messages[-1] if messages else None
                triggered_id = last_msg['id'] if last_msg else None
                updated = self.db.update_character_appearance(self.session_id, c_name, new_segments, triggered_id)
                if updated:
                    logger.info(f"[FromScratch] Appearance for {c_name} => {new_segments.model_dump()}")
            else:
                logger.warning(f"Failed to get structured appearance for {c_name}.")

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Done updating location & appearance from scratch for {c_name}."
                )

    def get_session_name(self) -> str:
        sessions = self.db.get_all_sessions()
        for session in sessions:
            if session['session_id'] == self.session_id:
                return session['name']
        return "Unnamed Session"

    def get_introduction_template(self) -> str:
        return INTRODUCTION_TEMPLATE
