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

def regenerate_scene_prompts():
    """
    Main routine with in-code configuration.
    Scans for prompt file pairs and generates scene prompts in the output directory.
    """
    input_dir = "./output"
    output_dir = "./output"
    model_url = "http://localhost:11434/api/generate"
    model_name = "huihui_ai/deepseek-r1-abliterated:70b"
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
            system_text = f.read()

        with open(usr_path, "r", encoding="utf-8") as f:
            user_text = f.read()

        logging.info(f"Generating scene prompt for prefix '{prefix}'...")
        result_obj = call_ollama(system_text, user_text, model_url, model_name, temperature)
        if not result_obj:
            logging.error(f"LLM returned no valid structured output for '{prefix}'. Skipping.")
            continue

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
