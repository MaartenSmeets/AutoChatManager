import os
import asyncio
import yaml
from datetime import datetime
import logging
from pydantic import BaseModel
from typing import Optional

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

    Additionally:
      - Saves system, user, and LLM-generated prompts to text files.
      - Checks if the final LLM output exceeds a word limit (150 words), and if so, 
        attempts to regenerate a more concise version up to a configurable number 
        of retries (loaded from image_manager_config.yaml as 'max_concise_retries').
    """

    def __init__(self, config_path: str, llm_client: OllamaClient):
        self.config_path = config_path
        self.llm_client = llm_client
        self.system_prompt = ""
        self.user_prompt_template = ""
        self.max_concise_retries = 3  # default if not specified in YAML
        self._load_config()

    def _load_config(self):
        """
        Loads configuration from the YAML file. Expected to include:
          - system_prompt
          - user_prompt_template
          - max_concise_retries (optional)
        """
        try:
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f)
            self.system_prompt = data.get('system_prompt', '')
            self.user_prompt_template = data.get('user_prompt_template', '')
            self.max_concise_retries = data.get('max_concise_retries', 3)
        except Exception as e:
            logger.error(f"Failed to load image_manager_config from {self.config_path}: {e}")
            self.system_prompt = ""
            self.user_prompt_template = ""
            self.max_concise_retries = 3

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
        """
        if not self.user_prompt_template.strip():
            logger.warning("No user_prompt_template found; returning empty description.")
            return "No template available."

        # Build a minimal bullet list from character data
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

        # Instantiate a local LLM client for structured output
        llm = OllamaClient(
            'src/multipersona_chat_app/config/llm_config.yaml',
            output_model=ImagePrompt
        )
        if self.llm_client.user_selected_model:
            llm.set_user_selected_model(self.llm_client.user_selected_model)

        # Optionally notify status
        if llm_status_callback:
            await llm_status_callback("[LLM] Generating concise scene description (structured output)...")

        # Save the system and user prompts to files before generation
        self.save_text_to_file(self.system_prompt, "system_prompt")
        self.save_text_to_file(final_prompt, "user_prompt")

        # Main LLM call
        result_obj = await asyncio.to_thread(
            llm.generate,
            prompt=final_prompt,
            system=self.system_prompt,
            use_cache=False
        )

        if not result_obj or not isinstance(result_obj, ImagePrompt):
            logger.warning("No valid structured output received for image prompt. Falling back.")
            return "No scene description generated."

        # Clean up the final text
        concise_result = result_obj.short_prompt.strip('\"\'').replace('\n', '')

        # If result is too large, attempt to concisely regenerate
        concise_result = await self._make_concise_if_needed(concise_result)

        # Save the final LLM output
        self.save_text_to_file(concise_result, "llm_output")

        return concise_result

    def save_prompt_to_file(self, prompt_text: str, output_folder: str = "output"):
        """
        Saves the prompt to the output folder with a timestamp prefix.
        Legacy method (still used internally if needed).
        """
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_scene_prompt.txt"
        filepath = os.path.join(output_folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(prompt_text)
        return filepath

    def save_text_to_file(self, text: str, file_prefix: str, output_folder: str = "output") -> str:
        """
        Saves arbitrary text to the output folder, using a timestamp plus the provided prefix.
        """
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file_prefix}.txt"
        filepath = os.path.join(output_folder, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            logger.error(f"Error saving file {filepath}: {e}")
        return filepath

    async def _make_concise_if_needed(self, text: str) -> str:
        """
        If 'text' exceeds 150 words, tries up to 'max_concise_retries' times to
        have the LLM produce a shorter version (still using the same structured output model).
        """
        def count_words(t: str) -> int:
            return len(t.split())

        current_text = text
        attempt = 0

        while attempt < self.max_concise_retries and count_words(current_text) > 150:
            attempt += 1
            logger.info(f"Prompt exceeds 150 words, attempting concise rewrite (attempt {attempt})...")

            # We'll re-use the ImagePrompt structure for conciseness requests
            llm = OllamaClient(
                'src/multipersona_chat_app/config/llm_config.yaml',
                output_model=ImagePrompt
            )
            if self.llm_client.user_selected_model:
                llm.set_user_selected_model(self.llm_client.user_selected_model)

            # System prompt & user prompt for the conciseness request
            system_prompt = (
                "You are an assistant rewriting a prompt to be shorter. "
                "Return the final text in the 'short_prompt' field, with minimal words (under 150)."
            )
            user_prompt = (
                "Rewrite this text to be under 150 words, preserving overall meaning:\n\n"
                f"{current_text}"
            )

            # Save the intermediate system/user prompts and output for debugging
            attempt_tag = f"concise_attempt{attempt}"
            self.save_text_to_file(system_prompt, f"system_prompt_{attempt_tag}")
            self.save_text_to_file(user_prompt, f"user_prompt_{attempt_tag}")

            # Generate
            result_obj = await asyncio.to_thread(
                llm.generate,
                prompt=user_prompt,
                system=system_prompt,
                use_cache=False
            )

            if not result_obj or not isinstance(result_obj, ImagePrompt):
                logger.warning("Concise rewrite attempt returned no valid output.")
                break

            new_text = result_obj.short_prompt.strip('\"\'').replace('\n', '')
            self.save_text_to_file(new_text, f"llm_output_{attempt_tag}")
            current_text = new_text

        return current_text
