import os
import logging
from typing import List, Dict, Tuple, Optional, Set
from models.character import Character
from db.db_manager import DBManager
from llm.ollama_client import OllamaClient
from datetime import datetime
from npc_manager import NPCManager
import yaml
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
    def __init__(self, session_id: Optional[str] = None, settings: List[Dict] = [], llm_client: Optional[OllamaClient] = None):
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

        moral_guidelines_path = os.path.join("src", "multipersona_chat_app", "config", "moral_guidelines.yaml")
        try:
            with open(moral_guidelines_path, 'r') as f:
                mg_data = yaml.safe_load(f)
                self.moral_guidelines = mg_data.get('moral_guidelines', '')
        except Exception as e:
            logger.error(f"Error loading moral guidelines: {e}")
            self.moral_guidelines = ""

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

        self.npc_manager_active = True
        self.last_npc_check_msg_id = 0

        npc_config_path = os.path.join("src", "multipersona_chat_app", "config", "npc_manager_config.yaml")
        self.npc_manager = NPCManager(
            session_id=self.session_id,
            db=self.db,
            llm_client=llm_client,
            config_path=npc_config_path
        )

        self.llm_status_callback = None

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
        self.llm_status_callback = callback
        if self.npc_manager:
            self.npc_manager.set_llm_status_callback(callback)

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
        self.current_setting = setting_name
        self.db.update_current_setting(self.session_id, self.current_setting)
        self.db.update_current_setting_description(self.session_id, setting_description)

        num_messages = len(self.db.get_messages(self.session_id))
        if num_messages == 0:
            self.db.update_current_location(self.session_id, start_location, None)
            logger.info(f"Setting changed to '{self.current_setting}'. Session not started, using start_location '{start_location}'.")
        else:
            logger.info(
                f"Setting changed to '{self.current_setting}' but session already has {num_messages} messages, "
                "keeping existing session location as-is."
            )

    def get_character_names(self) -> List[str]:
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

    def remove_character(self, char_name: str):
        if char_name in self.characters:
            del self.characters[char_name]
        if char_name in self.msg_counter_since_forced_update:
            del self.msg_counter_since_forced_update[char_name]
        self.db.remove_character_from_session(self.session_id, char_name)

    def ensure_character_plan_exists(self, char_name: str):
        plan_data = self.db.get_character_plan(self.session_id, char_name)
        if plan_data is None:
            logger.info(f"No existing plan for '{char_name}'. Not creating any default plan.")
        else:
            logger.debug(f"Plan for '{char_name}' already exists in DB. Goal: {plan_data['goal']}")

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
        self.db.save_character_plan(self.session_id, char_name, plan.goal, plan.steps, plan.why_new_plan_goal)

    def get_participants_in_turn_order(self) -> List[str]:
        """
        Returns the list of participants in the round-robin order:
        all user-added characters + "NPC Manager" as the last one.
        """
        return self.get_character_names() + ["NPC Manager"]

    
    def get_upcoming_speaker(self) -> Optional[str]:
        """
        Return who *would* speak next, WITHOUT incrementing any internal turn state.
        Safe to call repeatedly (e.g. for a label display) without skipping anyone.
        """
        npc_name = "NPC Manager"
        normal_participants = list(self.characters.keys())

        # If no characters, default to NPC Manager
        if not normal_participants:
            return npc_name

        # If needed, initialize tracking attributes
        if not hasattr(self, 'phase'):
            self.phase = "intro"
        if not hasattr(self, 'current_index'):
            self.current_index = 0
        if not hasattr(self, 'last_was_npc'):
            self.last_was_npc = False

        # PHASE 1: INTRODUCTIONS
        if self.phase == "intro":
            # If still introducing
            if self.current_index < len(normal_participants):
                return normal_participants[self.current_index]
            # Else intro phase is done, so it should move to roundrobin
            # But we won't forcibly fix it here, since we're not advancing.

            return normal_participants[-1]  # fallback (should not often hit)

        # PHASE 2: ROUND ROBIN
        if self.phase == "roundrobin":
            if self.last_was_npc:
                # Next is a normal character
                return normal_participants[self.current_index]
            else:
                # Next is the NPC Manager
                return npc_name

        # If we ever reach an unexpected state, default to NPC Manager
        return npc_name


    def proceed_turn(self) -> Optional[str]:
        """
        Advance the conversation turn and RETURN the speaker who now gets to speak.
        This updates internal counters so the next call yields the next turn in sequence.
        """
        # We can reuse the logic in get_upcoming_speaker to figure out who is *about* to speak.
        speaker = self.get_upcoming_speaker()

        npc_name = "NPC Manager"
        normal_participants = list(self.characters.keys())

        # If no characters, just return the NPC Manager
        if not normal_participants:
            return npc_name

        # --- Now "advance" state since we truly want to proceed. ---
        if not hasattr(self, 'phase'):
            self.phase = "intro"
        if not hasattr(self, 'current_index'):
            self.current_index = 0
        if not hasattr(self, 'last_was_npc'):
            self.last_was_npc = False

        if self.phase == "intro":
            if self.current_index < len(normal_participants):
                # We just used a normal character's introduction
                self.current_index += 1
                # If that was the last introduction, switch to round-robin
                if self.current_index == len(normal_participants):
                    self.phase = "roundrobin"
                    self.current_index = 0
                    # Changed this to True so the next turn is the first character, not the NPC Manager
                    self.last_was_npc = True

        elif self.phase == "roundrobin":
            if self.last_was_npc:
                # We are about to speak as a normal character, so move index forward
                self.current_index = (self.current_index + 1) % len(normal_participants)
                self.last_was_npc = False
            else:
                # We are about to speak as the NPC Manager
                self.last_was_npc = True

        return speaker

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
        return message_id

    def enable_npc_manager(self):
        self.npc_manager_active = True
        logger.info("NPC Manager has been enabled.")

    def disable_npc_manager(self):
        self.npc_manager_active = False
        logger.info("NPC Manager has been disabled.")

    async def check_summarization(self):
        all_msgs = self.db.get_messages(self.session_id)
        if not all_msgs:
            return

        participants = set(m["sender"] for m in all_msgs if m["message_type"] in ["user", "character"])
        for char_name in participants:
            if char_name not in self.characters:
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
                f"Summarizing history for {character_name} because their visible messages exceeded the threshold."
            )

        while True:
            msgs = self.db.get_visible_messages_for_character(self.session_id, character_name)
            if len(msgs) < self.summarization_threshold:
                break

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

                if sender in self.characters:
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
            await self.llm_status_callback(f"Finished summarizing history for {character_name}.")

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

            prompt = COMBINE_SUMMARIES_PROMPT.format(
                character_name=character_name,
                count_summaries=len(chunk_summaries),
                chunk_text="\n\n".join(chunk_summaries),
                moral_guidelines=self.moral_guidelines
            )

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Combining older summaries for {character_name} into a single consolidated summary."
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
                await self.llm_status_callback(f"Done combining summaries for {character_name}.")

    def get_latest_dialogue(self, character_name: str) -> str:
        visible_history = self.get_visible_history_for_character(character_name)
        recent_msgs = visible_history[-self.recent_dialogue_lines:]
        formatted_dialogue_lines = []
        for i, msg in enumerate(recent_msgs):
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

        logger.debug(f"Built prompt for character '{character_name}':\n{formatted_prompt}")
        return system_prompt, formatted_prompt

    def build_introduction_prompts_for_character(self, character_name: str) -> Tuple[str, str]:
        char = self.characters[character_name]
        plan_obj = self.get_character_plan(character_name)
        plan_first_step = plan_obj.steps[0] if plan_obj.steps else ""

        system_prompt = CHARACTER_INTRODUCTION_SYSTEM_PROMPT_TEMPLATE.format(
            character_name=char.name,
            character_description=char.character_description,
            fixed_traits=char.fixed_traits,
            appearance=char.appearance,
            plan_first_step=plan_first_step,
            moral_guidelines=self.moral_guidelines
        )

        visible_history = self.get_visible_history_for_character(character_name)
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
            appearance=char.appearance,
            character_description=char.character_description,
            fixed_traits=char.fixed_traits,
            setting=setting_description,
            location=session_loc,
            chat_history_summary=chat_history_summary,
            latest_dialogue=latest_text,
            current_appearance=self.db.get_character_appearance(self.session_id, character_name),
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

    def start_automatic_chat(self):
        self.automatic_running = True

    def stop_automatic_chat(self):
        self.automatic_running = False

    async def generate_character_message(self, character_name: str):
        """
        Generate a message for a character. 
        If 'character_name' is "NPC Manager", we handle the manager logic separately
        and do NOT call the manager again after it finishes its own turn (it decides itself).
        """
        if character_name == "NPC Manager":
            logger.info("Handling NPC Manager turn instead of normal character generation.")
            if self.npc_manager_active:
                # The manager logic is handled here (a single turn for NPCs).
                new_last_id = await self.npc_manager.check_npc_interactions(
                    current_setting_description=(self.current_setting_description or ""),
                    last_checked_msg_id=self.last_npc_check_msg_id
                )
                self.last_npc_check_msg_id = new_last_id
            return

        logger.info(f"Generating interaction for character: {character_name}")
        all_msgs = self.db.get_messages(self.session_id)
        triggered_message_id = all_msgs[-1]['id'] if all_msgs else None

        # First, update or generate the plan if needed.
        await self.update_character_plan(character_name, triggered_message_id)

        # Check if this character has already spoken before. If not, we do an introduction and stop.
        char_spoken_before = any(
            m for m in all_msgs
            if m["sender"] == character_name and m["message_type"] == "character"
        )

        if not char_spoken_before:
            # Generate a first-time introduction message.
            await self.generate_character_introduction_message(character_name)
            return

        # If we reach here, the character has spoken before => "normal" turn logic.
        try:
            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Generating next interaction for {character_name} to continue the conversation."
                )

            # Build the normal system + user prompt for LLM.
            system_prompt, formatted_prompt = self.build_prompt_for_character(character_name)

            llm = OllamaClient('src/multipersona_chat_app/config/llm_config.yaml', output_model=Interaction)
            if self.llm_client:
                llm.set_user_selected_model(self.llm_client.user_selected_model)

            # Generate the raw Interaction from the LLM.
            interaction = await asyncio.to_thread(
                llm.generate,
                prompt=formatted_prompt,
                system=system_prompt,
                use_cache=False
            )

            if not interaction:
                logger.warning(f"No response for {character_name}. Not storing.")
                if self.llm_status_callback:
                    await self.llm_status_callback(
                        f"No interaction generated for {character_name}; the LLM returned empty."
                    )
                return

            if not isinstance(interaction, Interaction):
                logger.error(f"Invalid interaction type from LLM: {type(interaction)}. Value: {interaction}")
                if self.llm_status_callback:
                    await self.llm_status_callback(f"Invalid or malformed LLM output for {character_name}.")
                return

            # Clean up the raw text (remove possible markdown, brackets, etc.).
            interaction.action = utils.remove_markdown(interaction.action)
            interaction.dialogue = utils.remove_markdown(interaction.dialogue)
            interaction.emotion = utils.remove_markdown(interaction.emotion)
            interaction.thoughts = utils.remove_markdown(interaction.thoughts)

            # Check for repetition or placeholders, possibly re-generate if needed.
            final_interaction = await self.check_and_regenerate_if_repetitive(
                character_name, system_prompt, formatted_prompt, interaction
            )
            if not final_interaction:
                logger.warning(f"Repetitive or invalid output could not be resolved for {character_name}.")
                final_interaction = interaction  # fallback to the original

            final_interaction.action = utils.remove_markdown(final_interaction.action)
            final_interaction.dialogue = utils.remove_markdown(final_interaction.dialogue)
            final_interaction.emotion = utils.remove_markdown(final_interaction.emotion)
            final_interaction.thoughts = utils.remove_markdown(final_interaction.thoughts)

            # Combine action & dialogue for a single message text.
            if final_interaction.action.strip() and final_interaction.dialogue.strip():
                formatted_message = f"*{final_interaction.action}*\n{final_interaction.dialogue}"
            elif final_interaction.action.strip():
                formatted_message = f"*{final_interaction.action}*"
            elif final_interaction.dialogue.strip():
                formatted_message = final_interaction.dialogue
            else:
                formatted_message = ""

            # Save the new message in the DB.
            msg_id = await self.add_message(
                character_name,
                formatted_message,
                visible=True,
                message_type="character",
                emotion=final_interaction.emotion,
                thoughts=final_interaction.thoughts
            )

            # Check if location or appearance should be updated.
            location_was_triggered = False
            appearance_was_triggered = False

            if final_interaction.location_change_expected:
                logger.debug(f"{character_name} indicated location change. Evaluating location update next.")
                await self.evaluate_location_update(character_name)
                location_was_triggered = True
            else:
                logger.info(f"No location change indicated for {character_name}.")

            if final_interaction.appearance_change_expected:
                logger.debug(f"{character_name} indicated appearance change. Evaluating appearance update next.")
                await self.evaluate_appearance_update(character_name)
                appearance_was_triggered = True
            else:
                logger.info(f"No appearance change indicated for {character_name}.")

            # If neither location nor appearance was triggered, we might do forced updates after N turns, etc.
            if not location_was_triggered and not appearance_was_triggered:
                self.msg_counter_since_forced_update[character_name] = \
                    self.msg_counter_since_forced_update.get(character_name, 0) + 1

                if self.msg_counter_since_forced_update[character_name] >= self.forced_update_interval:
                    logger.info(
                        f"Forced location & appearance update for {character_name} "
                        f"since interval={self.forced_update_interval} was reached."
                    )
                    await self.evaluate_location_update(character_name)
                    await self.evaluate_appearance_update(character_name)
                    self.msg_counter_since_forced_update[character_name] = 0
            else:
                self.msg_counter_since_forced_update[character_name] = 0

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Finished generating interaction for {character_name}."
                )

            # IMPORTANT: We have REMOVED the old block that immediately calls NPC Manager here!
            # We now rely on next_speaker() to say "NPC Manager" is next, thus the manager
            # gets its turn in a separate generate_character_message("NPC Manager") call.

        except Exception as e:
            logger.error(f"Error generating message for {character_name}: {e}", exc_info=True)
            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"An error occurred while generating an interaction for {character_name}."
                )

    async def evaluate_location_update(self, character_name: str, visited: Optional[Set[str]] = None):
        if visited is None:
            visited = set()
        if character_name in visited:
            return
        visited.add(character_name)

        try:
            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Evaluating location update for {character_name} because location change is expected or forced update triggered."
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
                    logger.info(f"Ignoring 'other_characters' entry: {other_char} not in current session.")
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
                    f"An error occurred while evaluating location update for {character_name}."
                )

    def build_location_update_context(self, character_name: str) -> str:
        visible_history = self.db.get_visible_messages_for_character(self.session_id, character_name)
        relevant_lines = []
        for m in visible_history[-5:]:
            if m["sender"] in self.characters:
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
                    f"Evaluating appearance update for {character_name} because appearance change is expected or forced update triggered."
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
                    logger.info(f"Ignoring 'other_characters' entry: {other_char} not in current session.")
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
                    f"An error occurred while evaluating appearance update for {character_name}."
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

    async def generate_character_introduction_message(self, character_name: str):
        logger.info(f"Building introduction prompts for character: {character_name}")
        system_prompt, introduction_prompt = self.build_introduction_prompts_for_character(character_name)

        introduction_llm_client = OllamaClient(
            'src/multipersona_chat_app/config/llm_config.yaml',
            output_model=CharacterIntroductionOutput
        )
        if self.llm_client:
            introduction_llm_client.set_user_selected_model(self.llm_client.user_selected_model)

        try:
            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Generating an initial introduction for {character_name}, since it's their first time speaking."
                )

            introduction_response = await asyncio.to_thread(
                introduction_llm_client.generate,
                prompt=introduction_prompt,
                system=system_prompt
            )
            if isinstance(introduction_response, CharacterIntroductionOutput):
                intro_text = introduction_response.introduction_text.strip()
                app_seg = introduction_response.current_appearance

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
                logger.info(f"Saved introduction message for {character_name}")
                if self.llm_status_callback:
                    await self.llm_status_callback(
                        f"Introduction completed for {character_name}."
                    )
            else:
                logger.warning(f"Invalid response received for introduction of {character_name}. Response: {introduction_response}")
                if self.llm_status_callback:
                    await self.llm_status_callback(
                        f"An invalid introduction was returned for {character_name}."
                    )

        except Exception as e:
            logger.error(f"Error generating introduction for {character_name}: {e}", exc_info=True)
            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"An error occurred while generating the introduction for {character_name}."
                )

    def get_session_name(self) -> str:
        sessions = self.db.get_all_sessions()
        for session in sessions:
            if session['session_id'] == self.session_id:
                return session['name']
        return "Unnamed Session"

    def get_introduction_template(self) -> str:
        return INTRODUCTION_TEMPLATE

    async def update_character_plan(self, character_name: str, triggered_message_id: Optional[int] = None):
        plan_client = OllamaClient(
            config_path='src/multipersona_chat_app/config/llm_config.yaml',
            output_model=CharacterPlan
        )
        if self.llm_client:
            plan_client.set_user_selected_model(self.llm_client.user_selected_model)

        existing_plan = self.get_character_plan(character_name)
        old_goal = existing_plan.goal
        old_steps = existing_plan.steps
        old_why = existing_plan.why_new_plan_goal

        my_appearance = self.db.get_character_appearance(self.session_id, character_name)
        others_appearance = self.db.get_characters_appearance_except_one(self.session_id, character_name)
        others_appearance = {
            c: a for c, a in others_appearance.items()
            if self.character_has_introduced(c)
        }
        character_description = self.characters[character_name].character_description

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
                    f"Generating (or updating) the plan for {character_name} due to new input or context."
                )

            plan_result = await asyncio.to_thread(
                plan_client.generate,
                prompt=user_prompt,
                system=system_prompt,
                use_cache=False
            )

            if not plan_result:
                logger.warning("Plan update returned no result. Keeping existing plan.")
                if self.llm_status_callback:
                    await self.llm_status_callback(
                        f"No plan update was generated for {character_name}; retaining old plan."
                    )
                return

            try:
                if isinstance(plan_result, CharacterPlan):
                    new_plan: CharacterPlan = plan_result
                else:
                    new_plan = CharacterPlan.model_validate_json(plan_result)

                new_goal = new_plan.goal
                new_steps = new_plan.steps
                new_why = new_plan.why_new_plan_goal.strip()

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
                        f"Plan has been updated for {character_name} (Goal: {new_goal})."
                    )

            except Exception as e2:
                logger.error(
                    f"Failed to parse new plan for '{character_name}'. Keeping old plan. Error: {e2}"
                )
                if self.llm_status_callback:
                    await self.llm_status_callback(
                        f"An error occurred while parsing the new plan for {character_name}, so the old plan remains."
                    )

        except Exception as e:
            logger.error(f"Error generating plan for {character_name}: {e}", exc_info=True)
            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"An error occurred while generating or updating the plan for {character_name}."
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
        all_visible = self.db.get_visible_messages_for_character(self.session_id, character_name)
        same_speaker_lines = [m for m in all_visible if m["sender"] == character_name]
        recent_speaker_lines = same_speaker_lines[-5:] if len(same_speaker_lines) > 5 else same_speaker_lines

        embed_client = OllamaClient('src/multipersona_chat_app/config/llm_config.yaml')
        if self.llm_client:
            embed_client.set_user_selected_model(self.llm_client.user_selected_model)

        tries = 0
        max_tries = self.max_similarity_retries
        current_interaction = interaction

        while True:
            action_text = current_interaction.action
            dialogue_text = current_interaction.dialogue

            violation_detected = False
            violation_reasons = []

            # Minimal check for empty or placeholder text
            if (not re.search(r'[A-Za-z0-9]', action_text)) and (not re.search(r'[A-Za-z0-9]', dialogue_text)):
                violation_detected = True
                violation_reasons.append(
                    "No alphanumeric characters found in both action and dialogue."
                )

            if "<" in action_text or ">" in action_text or "<" in dialogue_text or ">" in dialogue_text:
                violation_detected = True
                violation_reasons.append(
                    "Angle bracket placeholders detected in output. These are not allowed."
                )

            if dialogue_text.strip() and dialogue_text.strip() in action_text:
                violation_detected = True
                violation_reasons.append(
                    "Dialogue text is literally repeated in the action field."
                )

            action_embedding = embed_client.get_embedding(action_text)
            dialogue_embedding = embed_client.get_embedding(dialogue_text)
            actiondialogue_embedding = embed_client.get_embedding(action_text + ' ' + dialogue_text)

            cos_sim_action_dialogue = embed_client.compute_cosine_similarity(action_embedding, dialogue_embedding)
            jac_sim_action_dialogue = embed_client.compute_jaccard_similarity(action_text, dialogue_text)

            if (cos_sim_action_dialogue >= self.similarity_threshold or
                jac_sim_action_dialogue >= self.similarity_threshold):
                violation_detected = True
                violation_reasons.append("Action and dialogue are too similar.")

            if not violation_detected:
                for line_obj in recent_speaker_lines:
                    old_msg = line_obj["message"]
                    old_embedding = embed_client.get_embedding(old_msg)

                    cos_sim_action = embed_client.compute_cosine_similarity(action_embedding, old_embedding)
                    cos_sim_dialogue = embed_client.compute_cosine_similarity(dialogue_embedding, old_embedding)
                    cos_sim_both = embed_client.compute_cosine_similarity(actiondialogue_embedding, old_embedding)

                    jac_sim_action = embed_client.compute_jaccard_similarity(action_text, old_msg)
                    jac_sim_dialogue = embed_client.compute_jaccard_similarity(dialogue_text, old_msg)
                    jac_sim_both = embed_client.compute_jaccard_similarity(action_text + ' ' + dialogue_text, old_msg)

                    if (cos_sim_action >= self.similarity_threshold or jac_sim_action >= self.similarity_threshold or
                        cos_sim_dialogue >= self.similarity_threshold or jac_sim_dialogue >= self.similarity_threshold or
                        cos_sim_both >= self.similarity_threshold or jac_sim_both >= self.similarity_threshold):
                        violation_detected = True
                        violation_reasons.append(
                            "The new action/dialogue is too similar to a recent message."
                        )
                        break

            if not violation_detected:
                logger.info("All checks passed! Output is acceptable.")
                return current_interaction

            tries += 1
            if tries > max_tries:
                logger.warning("Exceeded maximum violation/regeneration attempts.")
                return None

            appended_warning = "\n\n".join(violation_reasons)
            appended_warning = (
                "\n\nIMPORTANT:\n"
                + appended_warning
                + "\nPlease regenerate without these violations. "
                + "Do not use angle brackets, and avoid duplication or empty text.\n"
            )

            logger.info(f"Violation(s) found: {violation_reasons}. Regenerating attempt #{tries}.")

            revised_prompt = dynamic_prompt + appended_warning
            regen_client = OllamaClient('src/multipersona_chat_app/config/llm_config.yaml', output_model=Interaction)
            if self.llm_client:
                regen_client.set_user_selected_model(self.llm_client.user_selected_model)

            new_interaction = await asyncio.to_thread(
                regen_client.generate,
                prompt=revised_prompt,
                system=system_prompt,
                use_cache=False
            )
            if not new_interaction or not isinstance(new_interaction, Interaction):
                logger.warning("No valid regeneration received; returning None.")
                return None

            current_interaction = new_interaction

    def character_has_introduced(self, candidate_char_name: str) -> bool:
        all_msgs = self.db.get_messages(self.session_id)
        for m in all_msgs:
            if m["sender"] == candidate_char_name and m["message_type"] == "character":
                return True
        return False

    async def update_all_characters_location_and_appearance_from_scratch(self):
        session_chars = self.db.get_session_characters(self.session_id)
        for c_name in session_chars:
            if c_name not in self.characters:
                continue

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Updating location and appearance from scratch for {c_name}."
                )

            char_obj = self.characters[c_name]
            setting_name = self.current_setting or ""
            setting_data = self.settings.get(setting_name, {})
            setting_description = setting_data.get('description', "")
            start_location = setting_data.get('start_location', "")
            character_description = char_obj.character_description
            fixed_traits = char_obj.fixed_traits

            visible_msgs = self.db.get_visible_messages_for_character(self.session_id, c_name)
            visible_msgs.sort(key=lambda x: x['id'])
            lines_for_history = []
            for m in visible_msgs:
                sender = m["sender"]
                content = m["message"]
                if sender in self.characters:
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
                logger.warning(f"Failed to get structured location for {c_name}. Skipping location update.")

            if final_location:
                messages = self.db.get_messages(self.session_id)
                last_msg = messages[-1] if messages else None
                triggered_id = last_msg['id'] if last_msg else None
                updated = self.db.update_character_location(self.session_id, c_name, final_location, triggered_id)
                if updated:
                    logger.info(f"[From Scratch] Updated location for {c_name} to: {final_location}")

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
                    logger.info("[From Scratch] Updated appearance for %s with new subfields: %s", c_name, new_segments.model_dump())
            else:
                logger.warning(f"Failed to get structured appearance for {c_name}. Skipping appearance update.")

            if self.llm_status_callback:
                await self.llm_status_callback(
                    f"Done updating location and appearance from scratch for {c_name}."
                )
