import requests
import logging
from typing import Optional, Type, List
from pydantic import BaseModel
import yaml
import json
import os
import numpy as np

from db.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, config_path: str, output_model: Optional[Type[BaseModel]] = None):
        self.config = self.load_config(config_path)
        self.output_model = output_model
        # Initialize cache
        cache_file = os.path.join("output", "llm_cache")
        self.cache_manager = CacheManager(cache_file)

        # Track a user-selected model if chosen in the UI (None means use config)
        self.user_selected_model: Optional[str] = None

    @staticmethod
    def load_config(config_path: str) -> dict:
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.debug(f"Configuration loaded successfully from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found at path: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            raise

    def set_user_selected_model(self, model_name: Optional[str]):
        """
        Set the model chosen by the user. If model_name is None or empty,
        we revert to using the config file's model_name.
        """
        if model_name and model_name.strip():
            logger.debug(f"User-selected model set to: {model_name}")
            self.user_selected_model = model_name
        else:
            logger.info("User-selected model is cleared; will revert to config.")
            self.user_selected_model = None

    def list_local_models(self) -> List[str]:
        """
        Call the Ollama endpoint /api/tags to list available local models.
        Return a list of model names or an empty list if there's an error.
        """
        url = "http://localhost:11434/api/tags"  # Hardcoded to Ollama default
        headers = {'Content-Type': 'application/json'}

        try:
            logger.info(f"Fetching local models from {url}")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or "models" not in data:
                logger.warning("Invalid response format from /api/tags")
                return []
            # Each item is a dict with 'name' and other metadata
            names = [m.get("name", "") for m in data["models"] if "name" in m]
            return names
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching local models: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error while fetching local models: {e}")
            return []

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        use_cache: bool = True
    ) -> Optional[BaseModel or str]:
        """
        Generate a response from the model. If user_selected_model is set,
        try that first. If it fails, fall back to the config's model_name.
        """
        # Attempt to use user-selected model if available, else config's model
        initial_model_name = self.user_selected_model or self.config.get('model_name')
        fallback_model_name = self.config.get('model_name')

        # We'll try up to two passes:
        # 1) user-selected model (if any)
        # 2) fallback to config model if the first fails
        tried_models = []
        for attempt_index, chosen_model in enumerate([initial_model_name, fallback_model_name], start=1):
            if chosen_model in tried_models or not chosen_model:
                # Skip if we already tried this model or it's empty
                continue
            tried_models.append(chosen_model)

            logger.debug(f"Using model '{chosen_model}' for attempt #{attempt_index}.")
            response_value = self._perform_inference(
                chosen_model,
                prompt,
                max_tokens,
                temperature,
                system,
                use_cache
            )
            if response_value is not None:
                if attempt_index == 2:
                    # This means the user-selected model failed, and we had to revert
                    logger.warning(f"Reverted to config model '{fallback_model_name}' after user-selected model failed.")
                return response_value

        # If neither model worked, return None
        logger.error("All attempts to generate response failed. Returning None.")
        return None

    def _perform_inference(
        self,
        model_name: str,
        prompt: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        system: Optional[str],
        use_cache: bool
    ) -> Optional[BaseModel or str]:
        """
        Perform the actual streaming request to the Ollama /api/generate endpoint
        using a specific model_name. If it fails, return None.

        This function also checks `skip_system_prompt` in the config. If true,
        the system prompt is not sent as 'system' but instead prepended to the
        user prompt. 
        """
        # Check cache first, if enabled
        if use_cache:
            cached_response = self.cache_manager.get_cached_response(prompt, model_name)
            if cached_response is not None:
                logger.debug("Returning cached LLM response from cache.")
                if self.output_model:
                    try:
                        return self.output_model.parse_raw(cached_response)
                    except:
                        logger.error("Error parsing cached response. Treating as invalid and returning None.")
                        return None
                else:
                    return cached_response

        # Prepare headers
        headers = {
            'Content-Type': 'application/json',
        }
        api_key = self.config.get('api_key')
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        # Decide whether to skip the system prompt
        skip_system_prompt = self.config.get('skip_system_prompt', False)
        if skip_system_prompt and system:
            # Prepend the system text to the user prompt
            prompt = f"{system}\n\n{prompt}"
            # We do NOT set 'system' in the payload
            system = None

        payload = {
            'model': model_name,
            'prompt': prompt,
            "stream": True,
            'options': {
                'temperature': temperature if temperature is not None else self.config.get('temperature', 0.7)
            }
        }

        # If we still have a system prompt to send, include it
        if system:
            payload['system'] = system

        if self.output_model:
            payload['format'] = self.output_model.model_json_schema()

        max_retries = self.config.get('max_retries', 3)

        # Debug-level logs for request URL, headers, and caching info
        logger.debug(f"Request URL: {self.config.get('api_url')}")
        log_headers = headers.copy()
        if 'Authorization' in log_headers:
            log_headers['Authorization'] = 'Bearer ***'
        logger.debug(f"Request Headers: {log_headers}")

        # The user wants request payload and final structured output at INFO level
        logger.info(f"Request Payload: {payload}")

        for attempt in range(1, max_retries + 1):
            try:
                with requests.post(
                    self.config.get('api_url'),
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=self.config.get('timeout', 300)
                ) as response:
                    logger.info(f"Received response with status code: {response.status_code}")
                    logger.debug(f"Response Headers: {response.headers}")
                    response.raise_for_status()

                    output = ""
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        logger.debug(f"Raw response line: {line}")

                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning("Received a line that could not be JSON-decoded, skipping...")
                            continue

                        if "error" in data:
                            logger.error(f"Error in response data: {data['error']}")
                            raise Exception(data["error"])

                        content = data.get("response", "")
                        output += content

                        if data.get("done", False):
                            # If we have an output model, parse it as structured data
                            if self.output_model:
                                try:
                                    parsed_output = self.output_model.model_validate_json(output)
                                    logger.info(f"Structured Output: {parsed_output.dict()}")
                                    if use_cache:
                                        self.cache_manager.store_response(prompt, model_name, output)
                                    return parsed_output
                                except Exception as e:
                                    logger.error(f"Error parsing model output: {e}")
                                    return None
                            else:
                                if use_cache:
                                    self.cache_manager.store_response(prompt, model_name, output)
                                logger.info("Final unstructured output stored in cache.")
                                return output

                    logger.error("No 'done' signal received before the stream ended.")
                    return None
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt} failed with RequestException: {e}")
                if attempt == max_retries:
                    logger.error(f"All {max_retries} attempts for model '{model_name}' failed. Giving up.")
                    return None
                else:
                    logger.info(f"Retrying... (Attempt {attempt + 1} of {max_retries})")
            except Exception as e:
                logger.error(f"An error occurred during inference with model '{model_name}': {e}")
                return None

    #
    # NEW: Embedding and similarity helpers
    #
    def get_embedding(self, sentence: str) -> List[float]:
        """
        Generate an embedding for 'sentence' using the Ollama /api/embeddings endpoint.
        """
        url = self.config.get('api_url_embeddings') or "http://localhost:11434/api/embeddings"
        model_name = self.config.get('embedding_model_name') or "snowflake-arctic-embed2"

        headers = {'Content-Type': 'application/json'}
        api_key = self.config.get('api_key')
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        data = {
            'model': model_name,
            'prompt': sentence
        }

        log_headers = headers.copy()
        if 'Authorization' in log_headers:
            log_headers['Authorization'] = 'Bearer ***'

        logger.debug("Sending request to Ollama Embeddings API")
        logger.debug(f"Request URL: {url}")
        logger.debug(f"Request Headers: {log_headers}")
        logger.debug(f"Request Payload: {data}")

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            logger.debug(f"Received response with status code: {response.status_code}")
            logger.debug(f"Response Headers: {response.headers}")
            response.raise_for_status()
            emb_data = response.json().get('embedding', [])
            logger.debug(f"Embedding data received: {emb_data}")
            return emb_data
        except requests.exceptions.RequestException as e:
            logger.error(f"RequestException while fetching embedding: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching embedding: {e}")
            return []

    def compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two embedding vectors.
        """
        if not vec1 or not vec2:
            return 0.0
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)
        denom = (np.linalg.norm(arr1) * np.linalg.norm(arr2))
        if denom == 0:
            return 0.0
        return float(np.dot(arr1, arr2) / denom)

    def compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Compute a simple Jaccard similarity between sets of words in text1 and text2.
        Lowercasing and splitting on whitespace for simplicity.
        """
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        if not set1 or not set2:
            return 0.0
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / float(len(union)) if union else 0.0
