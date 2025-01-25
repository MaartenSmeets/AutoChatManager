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
    suitable for image generation prompts, now using structured LLM output.
    """

    def __init__(self, config_path: str, llm_client: OllamaClient):
        self.config_path = config_path
        self.llm_client = llm_client
        self.template = ""
        self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f)
            self.template = data.get('scene_prompt_template', '')
        except Exception as e:
            logger.error(f"Failed to load image_manager_config from {self.config_path}: {e}")
            self.template = ""

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
        if not self.template.strip():
            logger.warning("No scene_prompt_template found; returning empty description.")
            return "No template available."

        # Build a short bullet for each character
        character_lines = []
        for char_info in non_npc_characters:
            location_part = (char_info['location'] or "").strip()
            traits_part = (char_info['traits'] or "").strip()
            appearance_part = (char_info['appearance'] or "").strip()

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

        # Fill the template
        final_prompt = (
            self.template
            .replace("{setting}", setting)
            .replace("{characters}", characters_text)
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

        # Call the LLM
        result_obj = await asyncio.to_thread(
            llm.generate,
            prompt=final_prompt,
            system="",     # System prompt left blank here, but can be set if needed
            use_cache=False
        )

        # Parse the structured output
        if not result_obj or not isinstance(result_obj, ImagePrompt):
            logger.warning("No valid structured output received for image prompt. Falling back.")
            return "No scene description generated."
        
        # Return the LLM-generated short prompt
        return result_obj.short_prompt.strip('\"\'').replace('\n', '')

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
