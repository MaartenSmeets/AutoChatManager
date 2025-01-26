# File: /home/maarten/AutoChatManager/src/multipersona_chat_app/image_manager.py
import os
import asyncio
import yaml
from datetime import datetime
import logging
from pydantic import BaseModel

from llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class ImagePrompt(BaseModel):
    """
    Represents the structured LLM output for scene descriptions.
    """
    short_prompt: str

class ImageManager:
    """
    Handles the creation of a concise, keyword-style scene description 
    suitable for image generation prompts, using structured LLM output
    with separate system and user prompts.
    """

    def __init__(self, config_path: str, llm_client: OllamaClient):
        self.config_path = config_path
        self.llm_client = llm_client
        self.system_prompt = ""
        self.user_prompt_template = ""
        self.max_concise_retries = 0
        self._load_config()

    def _load_config(self):
        """
        Loads configuration from image_manager_config.yaml.
        Expects an optional integer 'max_concise_retries' property 
        that determines how many times to retry making the output 
        concise if over 150 words.
        """
        try:
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f)
            self.system_prompt = data.get('system_prompt', '')
            self.user_prompt_template = data.get('user_prompt_template', '')
            # New property to configure how many times to retry if prompt is too long:
            self.max_concise_retries = data.get('max_concise_retries', 2)
        except Exception as e:
            logger.error(f"Failed to load image_manager_config from {self.config_path}: {e}")
            self.system_prompt = ""
            self.user_prompt_template = ""
            self.max_concise_retries = 2

    async def generate_concise_description(
        self,
        setting: str,
        moral_guidelines: str,
        non_npc_characters: list,
        llm_status_callback=None
    ) -> str:
        """
        Build a short, keywords-only description using a structured LLM output.

        :param setting: current setting description
        :param moral_guidelines: appended to the prompt
        :param non_npc_characters: list of dicts with keys:
            {
              'traits': str,
              'appearance': str,
              'location': str,
            }
          Names are excluded to keep them anonymous.
        :param llm_status_callback: optional async callback for status updates
        :return: short prompt text (potentially re-requested from the LLM if too long)
        """
        if not self.user_prompt_template.strip():
            logger.warning("No user_prompt_template found; returning empty description.")
            return "No template available."

        character_lines = []
        for char_info in non_npc_characters:
            location_part = (char_info.get('location') or "").strip()
            traits_part = (char_info.get('traits') or "").strip()
            appearance_part = (char_info.get('appearance') or "").strip()

            merged = []
            if location_part:
                merged.append(location_part)
            if traits_part:
                merged.append(traits_part)
            if appearance_part:
                merged.append(appearance_part)

            bullet_line = "- " + "; ".join(merged)
            character_lines.append(bullet_line)

        characters_text = "\n".join(character_lines)

        # Fill in the user prompt template
        final_prompt = (
            self.user_prompt_template
            .replace("{characters}", characters_text)
            .replace("{setting}", setting)
            .replace("{moral_guidelines}", moral_guidelines)
        )

        # Prepare the LLM client with structured output
        llm = OllamaClient(
            'src/multipersona_chat_app/config/llm_config.yaml',
            output_model=ImagePrompt
        )
        if self.llm_client.user_selected_model:
            llm.set_user_selected_model(self.llm_client.user_selected_model)

        if llm_status_callback:
            await llm_status_callback("[LLM] Generating concise scene description (structured output)...")

        # Call the LLM with the updated system prompt
        result_obj = await asyncio.to_thread(
            llm.generate,
            prompt=final_prompt,
            system=self.system_prompt,
            use_cache=False
        )

        if not result_obj or not isinstance(result_obj, ImagePrompt):
            logger.warning("No valid structured output received for image prompt. Falling back.")
            short_prompt = "No scene description generated."
        else:
            short_prompt = result_obj.short_prompt.strip('\"\'').replace('\n', '')

        # Possibly reduce prompt if it's over the word limit
        short_prompt = await self._enforce_concise(short_prompt, llm_status_callback)

        return short_prompt

    async def _enforce_concise(self, text: str, llm_status_callback=None) -> str:
        """
        If the text is over 150 words, re-requests a concise version from the LLM
        up to self.max_concise_retries times.
        """
        word_limit = 150
        tries = 0
        while self._word_count(text) > word_limit and tries < self.max_concise_retries:
            tries += 1
            if llm_status_callback:
                await llm_status_callback(
                    f"[ImageManager] Prompt text exceeds {word_limit} words. Retrying to make it more concise (attempt {tries})."
                )

            # Attempt to reduce the text
            new_text = await self._reduce_text(text)
            if new_text.strip():
                text = new_text.strip()
            else:
                # If we fail to get anything valid back, break to avoid empty looping
                break

        return text

    async def _reduce_text(self, text: str) -> str:
        """
        Requests the LLM to produce a shorter version of 'text' with fewer than 150 words,
        returning the same structured JSON with the 'short_prompt' field.
        """
        # We'll build a quick system prompt:
        system_prompt = (
            "You are an assistant that outputs a JSON with one field 'short_prompt'. "
            "The user has provided a text that is too long. Please reduce it to fewer than 150 words, "
            "preserving the main descriptive content. Keep the same JSON schema: { short_prompt: str }. "
        )
        # This user prompt includes the text we want to shorten.
        user_prompt = (
            f"Original text:\n\n{text}\n\n"
            "Rewrite this so it is under 150 words, maintaining the main ideas. Return valid JSON with a 'short_prompt' field."
        )

        reduce_llm = OllamaClient(
            'src/multipersona_chat_app/config/llm_config.yaml',
            output_model=ImagePrompt
        )
        if self.llm_client.user_selected_model:
            reduce_llm.set_user_selected_model(self.llm_client.user_selected_model)

        result_obj = await asyncio.to_thread(
            reduce_llm.generate,
            prompt=user_prompt,
            system=system_prompt,
            use_cache=False
        )

        if not result_obj or not isinstance(result_obj, ImagePrompt):
            logger.warning("Failed to get a valid concise output from the LLM. Returning original text.")
            return text

        return result_obj.short_prompt.strip('\"\'').replace('\n', '')

    def _word_count(self, text: str) -> int:
        """
        Simple helper to count words in a string.
        """
        return len(text.split())

    def save_prompt_to_file(self, prompt_text: str, output_folder: str = "output"):
        """
        Saves the prompt to the output folder with a timestamp prefix.
        """
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_scene_prompt.txt"
        filepath = os.path.join(output_folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(prompt_text)
        return filepath
