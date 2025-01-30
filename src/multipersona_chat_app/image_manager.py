import os
import asyncio
import yaml
from datetime import datetime
import logging
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)

class ImageManager:
    """
    Handles the creation of concise user/system prompts for image-related
    purposes. In this updated version, no LLM calls are made. We only
    produce and save user_prompt.txt and system_prompt.txt in a specified
    folder (e.g., 'image_prompts').
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.system_prompt = ""
        self.user_prompt_template = ""
        self.max_concise_retries = 3  # retained from config if needed
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
        recent_dialogue: str = ""
    ) -> None:
        """
        Builds user/system prompts and saves them to 'image_prompts' folder.
        No LLM calls or extra output files are generated.
        """
        if not self.user_prompt_template.strip():
            logger.warning("No user_prompt_template found; returning without generating prompts.")
            return

        # Build a textual representation of the non-NPC characters
        characters_list = []
        for i, char_info in enumerate(non_npc_characters, start=1):
            lines = [f"Character {char_info.get('name', 'Unknown')}:"]
            location = (char_info.get('location') or "").strip()
            traits = (char_info.get('traits') or "").strip()
            appearance = (char_info.get('appearance') or "").strip()

            if location:
                lines.append(f"- Location: {location}")
            if traits:
                lines.append(f"- Traits: {traits}")
            if appearance:
                lines.append(f"- Appearance: {appearance}")

            characters_list.append("\n".join(lines))

        characters_text = "\n\n".join(characters_list)

        # Fill in the user prompt template
        final_prompt = (
            self.user_prompt_template
                .replace("{characters}", characters_text)
                .replace("{setting}", setting)
                .replace("{moral_guidelines}", moral_guidelines)
                .replace("{recent_dialogue}", recent_dialogue)
        )

        # Save system_prompt.txt and user_prompt.txt to the 'image_prompts' folder
        self.save_text_to_file(self.system_prompt, "system_prompt", output_folder="image_prompts")
        self.save_text_to_file(final_prompt, "user_prompt", output_folder="image_prompts")

    def save_text_to_file(self, text: str, file_prefix: str, output_folder: str = "image_prompts") -> str:
        """
        Saves the given text to a timestamped file (file_prefix + .txt)
        in the specified output folder.
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
