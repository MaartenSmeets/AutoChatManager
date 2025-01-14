"""
Utility functions for loading settings and characters.
"""
import os
import yaml
import logging
from models.character import Character
from typing import List, Dict
import re

logger = logging.getLogger(__name__)

def load_settings() -> List[Dict]:
    """
    Load the list of settings from the settings.yaml file.
    """
    settings_path = os.path.join("src", "multipersona_chat_app", "config", "settings.yaml")
    logger.debug(f"Loading settings from {settings_path}")
    try:
        with open(settings_path, 'r') as f:
            data = yaml.safe_load(f)
            if isinstance(data, list):
                logger.info("Settings loaded successfully.")
                return data
            else:
                logger.warning("Settings file does not contain a list. Returning empty list.")
                return []
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        return []

def get_available_characters(directory: str) -> Dict[str, Character]:
    """
    Load all characters from the given directory. Each character is defined in a YAML file.
    """
    logger.debug(f"Retrieving available characters from directory: {directory}")
    characters = {}
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.yaml'):
                yaml_path = os.path.join(directory, filename)
                try:
                    char = Character.from_yaml(yaml_path)
                    characters[char.name] = char
                    logger.info(f"Loaded character: {char.name}")
                except Exception as e:
                    logger.error(f"Error loading character from {yaml_path}: {e}")
    except FileNotFoundError:
        logger.error(f"Characters directory '{directory}' not found.")
    return characters

def remove_markdown(text):
    """Remove Markdown formatting from the given text."""
    # Remove Markdown headings
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Remove Markdown bold and italic formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)        # Italic
    text = re.sub(r'__(.*?)__', r'\1', text)          # Bold with underscores
    text = re.sub(r'_(.*?)_', r'\1', text)            # Italic with underscores

    # Remove inline code formatting
    text = re.sub(r'`([^`]*)`', r'\1', text)

    # Remove strikethrough formatting
    text = re.sub(r'~~(.*?)~~', r'\1', text)

    # Remove any extra spacing or newlines
    text = re.sub(r'\n{2,}', '\n', text)

    return text.strip()