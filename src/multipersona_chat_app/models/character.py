from pydantic import BaseModel
from typing import Optional
import yaml

class Character(BaseModel):
    name: str
    character_system_prompt: Optional[str] = None
    dynamic_prompt_template: Optional[str] = None
    appearance: str
    character_description: str
    fixed_traits: str = ""  # NEW FIELD for unchanging physical properties

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "Character":
        """Load a Character instance from a YAML file."""
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        character_system_prompt = data.get('character_system_prompt')
        dynamic_prompt_template = data.get('dynamic_prompt_template')
        appearance = data.get('appearance', "")
        character_description = data.get('character_description', "")
        fixed_traits = data.get('fixed_traits', "")

        return cls(
            name=data['name'],
            character_system_prompt=character_system_prompt,
            dynamic_prompt_template=dynamic_prompt_template,
            appearance=appearance,
            character_description=character_description,
            fixed_traits=fixed_traits
        )

    def format_prompt(self, setting: str, chat_history_summary: str, latest_dialogue: str, location: str) -> str:
        """Format the dynamic_prompt_template with given variables, removing irrelevant headings if data is missing."""
        def optional_section(title: str, content: str):
            content = content.strip()
            if content:
                return f"### {title} ###\n{content}\n\n"
            else:
                return f"No {title.lower()} data available.\n\n"

        setting_section = optional_section("Setting", setting)
        current_location_section = optional_section("Current Location", location)
        history_section = optional_section("Chat History Summary", chat_history_summary)
        dialogue_section = optional_section("Latest Dialogue", latest_dialogue)
        current_outfit_section = optional_section("Current Outfit", self.appearance)

        prompt = self.dynamic_prompt_template.format(
            setting=setting_section.strip(),
            chat_history_summary=history_section.strip(),
            latest_dialogue=dialogue_section.strip(),
            current_location=current_location_section.strip(),
            current_outfit=current_outfit_section.strip()
        )

        return prompt
