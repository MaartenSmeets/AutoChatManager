import os
import glob
import requests
import json
import uuid
import logging
import traceback

# -------------------------------------------------------------------------
# Configuration parameters
# -------------------------------------------------------------------------
# Where is SD WebUI running?
CONFIG_SDWEBUI_SERVER_URL = "http://127.0.0.1:7860"

# Model to use in SD WebUI
CONFIG_SDWEBUI_MODEL = "waiANINSFWPONYXL_v130.safetensors"

# Content to prepend to every prompt (formerly CONFIG_SDWEBUI_LORA)
CONFIG_SDWEBUI_PREPEND = "score_9, score_8_up, score_7_up, source_anime"

# Negative prompt
CONFIG_NEGATIVE = (
    "worst quality,bad quality,jpeg artifacts, source_cartoon, 3d, (censor),monochrome,blurry, lowres,watermark"
)

# Sampler and model inference settings
CONFIG_SAMPLER_NAME = "DPM++ 2M Karras"
CONFIG_NUM_IMAGES = 3     # Number of images to generate per prompt
CONFIG_BATCH_SIZE = 1     # Typically keep this at 1; n_iter will handle how many images total
CONFIG_IMAGE_WIDTH = 1024
CONFIG_IMAGE_HEIGHT = 1024
CONFIG_CFG_SCALE = 7
CONFIG_STEPS = 30
CONFIG_SEED = -1

# Directory and file filter for prompts
CONFIG_PROMPTS_DIR = "output"    # Directory containing prompt files
CONFIG_PROMPT_FILTER = "*scene_prompt.txt"   # File filter for prompt files

# For logging / tracking
CONFIG_CLIENT_ID = str(uuid.uuid4())

# -------------------------------------------------------------------------
# SD WebUI payload template (SDXL example)
# -------------------------------------------------------------------------
SDWEBUI_PAYLOAD_TEMPLATE = """{
        "prompt": "",
        "steps": 30,
        "sampler_name": "",
        "cfg_scale": 4.0,
        "width": 1024,
        "height": 1024,
        "negative_prompt": "",
        "seed": -1,
        "override_settings": {
            "sd_model_checkpoint": "iniverse_v1.safetensors",
            "CLIP_stop_at_last_layers": "2"
        },
        "override_settings_restore_afterwards": true,
        "save_images": true,
        "alwayson_scripts": {
            "ADetailer": {
                "args": [
                    true,
                    false,
                    {
                        "ad_cfg_scale": 7,
                        "ad_checkpoint": "Use same checkpoint",
                        "ad_clip_skip": 1,
                        "ad_confidence": 0.3,
                        "ad_controlnet_guidance_end": 1,
                        "ad_controlnet_guidance_start": 0,
                        "ad_controlnet_model": "None",
                        "ad_controlnet_module": "None",
                        "ad_controlnet_weight": 1,
                        "ad_denoising_strength": 0.4,
                        "ad_dilate_erode": 4,
                        "ad_inpaint_height": 512,
                        "ad_inpaint_only_masked": true,
                        "ad_inpaint_only_masked_padding": 32,
                        "ad_inpaint_width": 512,
                        "ad_mask_blur": 4,
                        "ad_mask_k_largest": 0,
                        "ad_mask_max_ratio": 1,
                        "ad_mask_merge_invert": "None",
                        "ad_mask_min_ratio": 0,
                        "ad_model": "face_yolov8n.pt",
                        "ad_model_classes": "",
                        "ad_negative_prompt": "",
                        "ad_noise_multiplier": 1,
                        "ad_prompt": "",
                        "ad_restore_face": false,
                        "ad_sampler": "DPM++ 2M",
                        "ad_scheduler": "Use same scheduler",
                        "ad_steps": 28,
                        "ad_tab_enable": true,
                        "ad_use_cfg_scale": false,
                        "ad_use_checkpoint": false,
                        "ad_use_clip_skip": false,
                        "ad_use_inpaint_width_height": false,
                        "ad_use_noise_multiplier": false,
                        "ad_use_sampler": false,
                        "ad_use_steps": false,
                        "ad_use_vae": false,
                        "ad_vae": "Use same VAE",
                        "ad_x_offset": 0,
                        "ad_y_offset": 0,
                        "is_api": []
                    },
                    {
                        "ad_cfg_scale": 7,
                        "ad_checkpoint": "Use same checkpoint",
                        "ad_clip_skip": 1,
                        "ad_confidence": 0.3,
                        "ad_controlnet_guidance_end": 1,
                        "ad_controlnet_guidance_start": 0,
                        "ad_controlnet_model": "None",
                        "ad_controlnet_module": "None",
                        "ad_controlnet_weight": 1,
                        "ad_denoising_strength": 0.4,
                        "ad_dilate_erode": 4,
                        "ad_inpaint_height": 512,
                        "ad_inpaint_only_masked": true,
                        "ad_inpaint_only_masked_padding": 32,
                        "ad_inpaint_width": 512,
                        "ad_mask_blur": 4,
                        "ad_mask_k_largest": 0,
                        "ad_mask_max_ratio": 1,
                        "ad_mask_merge_invert": "None",
                        "ad_mask_min_ratio": 0,
                        "ad_model": "None",
                        "ad_model_classes": "",
                        "ad_negative_prompt": "",
                        "ad_noise_multiplier": 1,
                        "ad_prompt": "",
                        "ad_restore_face": false,
                        "ad_sampler": "DPM++ 2M",
                        "ad_scheduler": "Use same scheduler",
                        "ad_steps": 28,
                        "ad_tab_enable": false,
                        "ad_use_cfg_scale": false,
                        "ad_use_checkpoint": false,
                        "ad_use_clip_skip": false,
                        "ad_use_inpaint_width_height": false,
                        "ad_use_noise_multiplier": false,
                        "ad_use_sampler": false,
                        "ad_use_steps": false,
                        "ad_use_vae": false,
                        "ad_vae": "Use same VAE",
                        "ad_x_offset": 0,
                        "ad_y_offset": 0,
                        "is_api": []
                    },
                    {
                        "ad_cfg_scale": 7,
                        "ad_checkpoint": "Use same checkpoint",
                        "ad_clip_skip": 1,
                        "ad_confidence": 0.3,
                        "ad_controlnet_guidance_end": 1,
                        "ad_controlnet_guidance_start": 0,
                        "ad_controlnet_model": "None",
                        "ad_controlnet_module": "None",
                        "ad_controlnet_weight": 1,
                        "ad_denoising_strength": 0.4,
                        "ad_dilate_erode": 4,
                        "ad_inpaint_height": 512,
                        "ad_inpaint_only_masked": true,
                        "ad_inpaint_only_masked_padding": 32,
                        "ad_inpaint_width": 512,
                        "ad_mask_blur": 4,
                        "ad_mask_k_largest": 0,
                        "ad_mask_max_ratio": 1,
                        "ad_mask_merge_invert": "None",
                        "ad_mask_min_ratio": 0,
                        "ad_model": "None",
                        "ad_model_classes": "",
                        "ad_negative_prompt": "",
                        "ad_noise_multiplier": 1,
                        "ad_prompt": "",
                        "ad_restore_face": false,
                        "ad_sampler": "DPM++ 2M",
                        "ad_scheduler": "Use same scheduler",
                        "ad_steps": 28,
                        "ad_tab_enable": false,
                        "ad_use_cfg_scale": false,
                        "ad_use_checkpoint": false,
                        "ad_use_clip_skip": false,
                        "ad_use_inpaint_width_height": false,
                        "ad_use_noise_multiplier": false,
                        "ad_use_sampler": false,
                        "ad_use_steps": false,
                        "ad_use_vae": false,
                        "ad_vae": "Use same VAE",
                        "ad_x_offset": 0,
                        "ad_y_offset": 0,
                        "is_api": []
                    },
                    {
                        "ad_cfg_scale": 7,
                        "ad_checkpoint": "Use same checkpoint",
                        "ad_clip_skip": 1,
                        "ad_confidence": 0.3,
                        "ad_controlnet_guidance_end": 1,
                        "ad_controlnet_guidance_start": 0,
                        "ad_controlnet_model": "None",
                        "ad_controlnet_module": "None",
                        "ad_controlnet_weight": 1,
                        "ad_denoising_strength": 0.4,
                        "ad_dilate_erode": 4,
                        "ad_inpaint_height": 512,
                        "ad_inpaint_only_masked": true,
                        "ad_inpaint_only_masked_padding": 32,
                        "ad_inpaint_width": 512,
                        "ad_mask_blur": 4,
                        "ad_mask_k_largest": 0,
                        "ad_mask_max_ratio": 1,
                        "ad_mask_merge_invert": "None",
                        "ad_mask_min_ratio": 0,
                        "ad_model": "None",
                        "ad_model_classes": "",
                        "ad_negative_prompt": "",
                        "ad_noise_multiplier": 1,
                        "ad_prompt": "",
                        "ad_restore_face": false,
                        "ad_sampler": "DPM++ 2M",
                        "ad_scheduler": "Use same scheduler",
                        "ad_steps": 28,
                        "ad_tab_enable": false,
                        "ad_use_cfg_scale": false,
                        "ad_use_checkpoint": false,
                        "ad_use_clip_skip": false,
                        "ad_use_inpaint_width_height": false,
                        "ad_use_noise_multiplier": false,
                        "ad_use_sampler": false,
                        "ad_use_steps": false,
                        "ad_use_vae": false,
                        "ad_vae": "Use same VAE",
                        "ad_x_offset": 0,
                        "ad_y_offset": 0,
                        "is_api": []
                    }
                ]
            }
        }
    }"""

# -------------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -------------------------------------------------------------------------
# Switch the SD WebUI model
# -------------------------------------------------------------------------
def switch_model(model_name: str) -> bool:
    """Switch the SDWebUI model to the specified checkpoint name."""
    options_url = f"{CONFIG_SDWEBUI_SERVER_URL}/sdapi/v1/options"
    try:
        response = requests.get(options_url)
        response.raise_for_status()
        current_options = response.json()
        current_options["sd_model_checkpoint"] = model_name
        response = requests.post(options_url, json=current_options)
        response.raise_for_status()
        logging.info(f"Successfully switched to model: {model_name}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to switch model to {model_name}: {e}")
        return False

# -------------------------------------------------------------------------
# Generate images via SD WebUI
# -------------------------------------------------------------------------
def generate_images_sdwebui(prompt: str, negative_prompt: str, images_to_generate: int) -> None:
    """
    Generate images using SD WebUI for a given prompt and negative prompt.

    Args:
        prompt (str): The final prompt to send to SD WebUI.
        negative_prompt (str): Negative prompt to send.
        images_to_generate (int): How many images (n_iter) to generate per call.
    """
    url = f"{CONFIG_SDWEBUI_SERVER_URL}/sdapi/v1/txt2img"
    try:
        # Convert the SDWEBUI_PAYLOAD_TEMPLATE from JSON string to a Python dict
        payload = json.loads(SDWEBUI_PAYLOAD_TEMPLATE)

        # Fill in fields from existing config
        payload["prompt"] = prompt.strip()
        payload["negative_prompt"] = negative_prompt or ""
        payload["cfg_scale"] = CONFIG_CFG_SCALE
        payload["steps"] = CONFIG_STEPS
        payload["seed"] = CONFIG_SEED
        payload["sampler_name"] = CONFIG_SAMPLER_NAME
        payload["width"] = CONFIG_IMAGE_WIDTH
        payload["height"] = CONFIG_IMAGE_HEIGHT
        payload["batch_size"] = CONFIG_BATCH_SIZE
        payload["n_iter"] = images_to_generate

        # Override model
        payload["override_settings"]["sd_model_checkpoint"] = CONFIG_SDWEBUI_MODEL

        # ADetailer modifications (inpaint width/height remain 512 as in the template)
        # Example for the first set of ADetailer args (the one that's enabled by default):
        ad_detailer_args = payload["alwayson_scripts"]["ADetailer"]["args"][2]
        ad_detailer_args["ad_prompt"] = prompt.strip()
        ad_detailer_args["ad_negative_prompt"] = negative_prompt or ""
        ad_detailer_args["ad_sampler"] = CONFIG_SAMPLER_NAME
        ad_detailer_args["ad_steps"] = CONFIG_STEPS
        ad_detailer_args["ad_cfg_scale"] = CONFIG_CFG_SCALE
        
        # Keep "ad_inpaint_width" and "ad_inpaint_height" at 512 as per requirement
        # The rest remain as is or "Use same checkpoint" as per the template

        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        images = result.get('images', [])

        if images:
            logging.info(f"Generated {len(images)} images for prompt: {prompt}")
        else:
            logging.warning(f"No images returned for prompt: {prompt}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Error during image generation: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON response: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during image generation: {e}")

# -------------------------------------------------------------------------
# Read prompts from files in a directory and generate images
# -------------------------------------------------------------------------
def generate_images_from_directory(
    directory: str,
    file_filter: str,
    prepend_text: str,
    negative_prompt: str,
    images_to_generate: int
):
    """
    For each file in the given directory that matches the file_filter,
    read its contents as a prompt, prepend the specified text, and
    generate images using SDWebUI.

    Args:
        directory (str): Path to the directory containing prompt files.
        file_filter (str): Pattern to match prompt files (e.g., "*.txt").
        prepend_text (str): Text to prepend to every prompt.
        negative_prompt (str): Negative prompt to use in generation.
        images_to_generate (int): Number of images to generate per file.
    """
    prompt_files = glob.glob(os.path.join(directory, file_filter))
    if not prompt_files:
        logging.warning(f"No files found in directory '{directory}' matching filter '{file_filter}'.")
        return

    for file_path in prompt_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_prompt = f.read().strip()
                # Build the final prompt
                final_prompt = f"{prepend_text}, {file_prompt}" if prepend_text else file_prompt
                logging.info(f"Generating images for prompt file: {file_path}")
                generate_images_sdwebui(final_prompt, negative_prompt, images_to_generate)
        except Exception as e:
            logging.error(f"Failed to process file '{file_path}': {e}")

# -------------------------------------------------------------------------
# Main execution
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Optionally switch models before generating
    if not switch_model(CONFIG_SDWEBUI_MODEL):
        logging.error(f"Failed to switch to model {CONFIG_SDWEBUI_MODEL}. Aborting.")
    else:
        # Generate images using all prompt files in the specified directory
        generate_images_from_directory(
            directory=CONFIG_PROMPTS_DIR,
            file_filter=CONFIG_PROMPT_FILTER,
            prepend_text=CONFIG_SDWEBUI_PREPEND,
            negative_prompt=CONFIG_NEGATIVE,
            images_to_generate=CONFIG_NUM_IMAGES
        )
        logging.info("Image generation process completed.")
