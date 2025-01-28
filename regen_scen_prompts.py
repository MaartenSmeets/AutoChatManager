#!/usr/bin/env python3
"""
Regenerate scene prompts by combining system and user prompts into a single
LLM call that returns a structured JSON output with a single string field.

- Reads configuration from a local YAML file (no command-line parameters).
- Finds matching system/user prompt files by shared prefix in alphabetical order.
- Calls a local LLM endpoint for a structured JSON response.
- Parses that JSON (with Pydantic) into a one-field model: short_prompt.
- Saves the final short_prompt to <prefix>_scene_prompt.txt in the output folder.

Required config file (example: 'scene_regeneration_config.yaml') structure:

--------------------------------------------------------------------------------
# scene_regeneration_config.yaml
input_directory: "./input_prompts"
output_directory: "./output_prompts"
model_url: "http://localhost:11434/api/generate"
model_name: "llama2-7b"    # or your local model name
temperature: 0.7
--------------------------------------------------------------------------------

Each pair of prompt files in the input_directory should look like:

- {prefix}_system_prompt.txt
- {prefix}_user_prompt.txt

Example filenames:
  "fantasy_castle_system_prompt.txt"
  "fantasy_castle_user_prompt.txt"

The script will generate "fantasy_castle_scene_prompt.txt" in the output_directory.
"""

import os
import yaml
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
    Pydantic model to hold the structured output from the LLM.
    We expect a JSON object like: {"short_prompt": "some string"}
    """
    short_prompt: str

def load_config(config_path: str = "scene_regeneration_config.yaml") -> dict:
    """
    Loads YAML configuration from the given path.
    Returns a dictionary with keys:
      - input_directory
      - output_directory
      - model_url
      - model_name
      - temperature
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if data else {}
    except FileNotFoundError:
        logging.error(f"Configuration file not found at '{config_path}'.")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML config: {e}")
        return {}
    except Exception as e:
        logging.error(f"Unexpected error loading config: {e}")
        return {}

def call_ollama(
    system_prompt: str,
    user_prompt: str,
    model_url: str,
    model_name: str,
    temperature: float
) -> Optional[ScenePromptOutput]:
    """
    Calls a local LLM (e.g. Ollama) endpoint at model_url with streaming output.
    We expect the model to return lines of JSON, culminating in a JSON
    object that matches ScenePromptOutput.

    If successful, returns a ScenePromptOutput instance. Otherwise, None.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": user_prompt,    # user role content
        "system": system_prompt,  # system role content
        "options": {"temperature": temperature},
        "stream": True
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
        # Each line is expected to be valid JSON with possible fields:
        # { "response": "...", "done": false }, or {"done": true} at the end, etc.
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
            # Attempt to parse final combined text as JSON
            try:
                parsed = ScenePromptOutput.parse_raw(full_output)
                return parsed
            except Exception as e:
                logging.error(f"Could not parse final output as ScenePromptOutput: {e}")
                return None

    logging.error("LLM stream ended without a 'done' signal.")
    return None


def regenerate_scene_prompts():
    """
    Main routine (no command-line args).
    - Loads config from scene_regeneration_config.yaml
    - Scans input_directory for pairs of system/user prompt files in alphabetical order.
    - For each pair with prefix P, calls the LLM, then writes P_scene_prompt.txt to output_directory.
    """
    config = load_config("scene_regeneration_config.yaml")
    if not config:
        logging.error("No valid configuration loaded. Aborting.")
        return

    input_dir = "./output"
    output_dir = "./output"
    model_url = "http://localhost:11434/api/generate"
    model_name = "llama2-7b"    # or your local model name
    temperature = 0.7

    if not os.path.isdir(input_dir):
        logging.error(f"Input directory '{input_dir}' does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Collect all potential system prompt files
    # A valid pair is something like <prefix>_system_prompt.txt and <prefix>_user_prompt.txt
    all_files = os.listdir(input_dir)
    all_files.sort()  # alphabetical

    # We'll store discovered prefixes in a dictionary
    # Key = prefix, Value = {"system": "full path", "user": "full path"}
    prefix_map = {}

    for filename in all_files:
        # We expect the pattern <prefix>_system_prompt.txt or <prefix>_user_prompt.txt
        if filename.endswith("_system_prompt.txt"):
            prefix = filename.replace("_system_prompt.txt", "")
            prefix_map.setdefault(prefix, {})["system"] = filename
        elif filename.endswith("_user_prompt.txt"):
            prefix = filename.replace("_user_prompt.txt", "")
            prefix_map.setdefault(prefix, {})["user"] = filename
        else:
            # Not a relevant prompt file
            continue

    # Now iterate over each prefix in alphabetical order
    for prefix in sorted(prefix_map.keys()):
        prompt_files = prefix_map[prefix]
        sys_file = prompt_files.get("system")
        usr_file = prompt_files.get("user")

        if not sys_file or not usr_file:
            logging.warning(f"Incomplete pair for prefix '{prefix}'. Skipping.")
            continue

        sys_path = os.path.join(input_dir, sys_file)
        usr_path = os.path.join(input_dir, usr_file)

        # Load system prompt
        with open(sys_path, "r", encoding="utf-8") as f:
            system_text = f.read()

        # Load user prompt
        with open(usr_path, "r", encoding="utf-8") as f:
            user_text = f.read()

        logging.info(f"Generating scene prompt for prefix '{prefix}'...")
        result_obj = call_ollama(system_text, user_text, model_url, model_name, temperature)
        if not result_obj:
            logging.error(f"LLM returned no valid structured output for '{prefix}'. Skipping.")
            continue

        # result_obj.short_prompt is the final scene prompt
        scene_prompt_text = result_obj.short_prompt.strip()

        out_filename = f"{prefix}_scene_prompt.txt"
        out_filepath = os.path.join(output_dir, out_filename)

        try:
            with open(out_filepath, "w", encoding="utf-8") as f_out:
                f_out.write(scene_prompt_text + "\n")
            logging.info(f"Saved scene prompt to: {out_filepath}")
        except Exception as ex:
            logging.error(f"Failed to write scene prompt for '{prefix}': {ex}")


if __name__ == "__main__":
    regenerate_scene_prompts()
