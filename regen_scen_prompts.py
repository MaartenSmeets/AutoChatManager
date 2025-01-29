#!/usr/bin/env python3
"""
Regenerate scene prompts by combining system and user prompts into a single
LLM call that returns a structured JSON output with a single string field.

Finds matching system/user prompt files by shared prefix in alphabetical order,
calls a local LLM endpoint for a structured JSON response, parses that JSON (with
Pydantic) into a one-field model ("short_prompt"), and saves the final short_prompt
to <prefix>_scene_prompt.txt in the output folder.

A valid prompt file pair:
- <prefix>_system_prompt.txt
- <prefix>_user_prompt.txt

This script will generate:
- <prefix>_scene_prompt.txt
"""

import os
import json
import requests
import logging
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)

class ScenePromptOutput(BaseModel):
    """
    We expect a JSON object like: {"short_prompt": "some string"}
    """
    short_prompt: str

def call_ollama(
    system_prompt: str,
    user_prompt: str,
    model_url: str,
    model_name: str,
    temperature: float
) -> Optional[ScenePromptOutput]:
    """
    Calls a local LLM endpoint with streaming output.
    We expect the model to return lines of JSON, culminating in a JSON
    object that matches ScenePromptOutput.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": user_prompt,
        "system": system_prompt,
        "options": {"temperature": temperature},
        "stream": True,
        "format": ScenePromptOutput.model_json_schema()
    }

    try:
        response = requests.post(model_url, json=payload, headers=headers, stream=True, timeout=300)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"HTTP error contacting LLM: {e}")
        return None

    full_output = ""
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logging.warning(f"Skipping non-JSON line: {line}")
            continue

        if "error" in data:
            logging.error(f"Error from LLM: {data['error']}")
            return None

        full_output += data.get("response", "")

        if data.get("done"):
            try:
                return ScenePromptOutput.model_validate_json(full_output)
            except Exception as e:
                logging.error(f"Could not parse final output as ScenePromptOutput: {e}")
                return None

    logging.error("LLM stream ended without a 'done' signal.")
    return None

def generate_scene_prompt(
    original_system_prompt: str,
    original_user_prompt: str,
    model_url: str,
    model_name: str,
    temperature: float,
    max_retries: int = 3
) -> Optional[str]:
    """
    Attempts to generate a scene prompt up to 'max_retries' times.
    - If the returned prompt is fewer than 100 words, we retry by appending extra detail instructions.
    - If the returned prompt is more than 200 words, we retry by switching to a condensing system prompt.
    - Otherwise, we accept the prompt.
    """

    # System prompt for reducing/condensing a too-long prompt.
    condensing_system_prompt = r"""
You are an expert at condensing image generation prompts. Remove non-visual elements
that don't contribute to image quality (like sounds and smells), but retain atmospheric 
elements that enhance visual generation like textures, mood, and artistic style. 
Preserve essential visual elements, style descriptors, and composition details while 
removing redundant phrases and unnecessary modifiers. Maintain the original artistic 
intent but with fewer words.

Return the result as JSON in this format:
{"short_prompt": "<condensed version emphasizing visual elements, mood, composition, textures, and style that enhance image generation>"}

Note: Fields enclosed in angle brackets (<...>) are placeholders. When generating output, 
do not include the angle brackets. Replace the placeholder text with detailed content 
that fits the context.
""".strip()

    # Template for user prompt when condensing is needed.
    condense_user_prompt_template = (
        "Condense this image generation prompt while preserving key visual elements and style:\n\n{}"
    )

    # Additional note added when the output is too short.
    additional_detail_note = (
        "\n\nAdditional note: The condensed prompt was too short previously. "
        "Please include enough descriptive elements so that characters, setting, and "
        "important details are recognizable. Keep essential context and style."
    )

    current_system_prompt = original_system_prompt
    current_user_prompt = original_user_prompt

    attempt = 0
    result = None

    while attempt < max_retries:
        attempt += 1
        logging.info(f"Attempt {attempt} of {max_retries}...")

        result = call_ollama(
            system_prompt=current_system_prompt,
            user_prompt=current_user_prompt,
            model_url=model_url,
            model_name=model_name,
            temperature=temperature
        )

        if not result:
            logging.warning("LLM returned invalid or no output. Retrying with the same prompts.")
            continue

        scene_prompt_text = result.short_prompt.strip()
        # Count words instead of characters
        word_count = len(scene_prompt_text.split())

        if word_count < 100:
            logging.info(
                f"Prompt too short ({word_count} words). Adding detail instructions and retrying."
            )
            current_user_prompt += additional_detail_note

        elif word_count > 200:
            logging.info(
                f"Prompt too long ({word_count} words). Switching to condensing system prompt and retrying."
            )
            current_system_prompt = condensing_system_prompt
            current_user_prompt = condense_user_prompt_template.format(scene_prompt_text)

        else:
            # It's within acceptable range (100-200 words)
            return scene_prompt_text

    # If we exit the loop, we didn't get a prompt in the desired word range.
    # Return the last result if it exists, otherwise None.
    if result:
        logging.warning(
            "Exceeded maximum retries. Returning the last valid prompt even if it's out of the 100-200 word range."
        )
        return result.short_prompt.strip()
    else:
        logging.error("Exceeded maximum retries and never received valid output.")
        return None

def regenerate_scene_prompts():
    """
    Main routine with in-code configuration.
    Scans for prompt file pairs and generates scene prompts in the output directory.
    """
    input_dir = "./output"
    output_dir = "./output"
    model_url = "http://localhost:11434/api/generate"

    # Configure model name if needed
    model_name = "nchapman/l3.3-70b-euryale-v2.3:latest"

    # Number of retries
    max_retries = 3

    temperature = 0.7

    if not os.path.isdir(input_dir):
        logging.error(f"Input directory '{input_dir}' does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    all_files = os.listdir(input_dir)
    all_files.sort()

    prefix_map = {}
    for filename in all_files:
        if filename.endswith("_system_prompt.txt"):
            prefix = filename.replace("_system_prompt.txt", "")
            prefix_map.setdefault(prefix, {})["system"] = filename
        elif filename.endswith("_user_prompt.txt"):
            prefix = filename.replace("_user_prompt.txt", "")
            prefix_map.setdefault(prefix, {})["user"] = filename

    for prefix in sorted(prefix_map.keys()):
        prompt_files = prefix_map[prefix]
        sys_file = prompt_files.get("system")
        usr_file = prompt_files.get("user")

        if not sys_file or not usr_file:
            logging.warning(f"Incomplete pair for prefix '{prefix}'. Skipping.")
            continue

        sys_path = os.path.join(input_dir, sys_file)
        usr_path = os.path.join(input_dir, usr_file)

        with open(sys_path, "r", encoding="utf-8") as f:
            system_text = f.read().strip()

        with open(usr_path, "r", encoding="utf-8") as f:
            user_text = f.read().strip()

        logging.info(f"Generating scene prompt for prefix '{prefix}'...")
        final_prompt = generate_scene_prompt(
            original_system_prompt=system_text,
            original_user_prompt=user_text,
            model_url=model_url,
            model_name=model_name,
            temperature=temperature,
            max_retries=max_retries
        )

        if not final_prompt:
            logging.error(f"Failed to obtain a valid prompt for '{prefix}' after {max_retries} retries. Skipping.")
            continue

        out_filename = f"{prefix}_scene_prompt.txt"
        out_filepath = os.path.join(output_dir, out_filename)

        try:
            with open(out_filepath, "w", encoding="utf-8") as f_out:
                f_out.write(final_prompt + "\n")
            logging.info(f"Saved scene prompt to: {out_filepath}")
        except Exception as ex:
            logging.error(f"Failed to write scene prompt for '{prefix}': {ex}")

if __name__ == "__main__":
    regenerate_scene_prompts()
