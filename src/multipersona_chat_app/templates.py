from pydantic import BaseModel
from typing import Optional

#
# New model to hold five appearance subfields for the introduction
#
class IntroductionAppearanceSegments(BaseModel):
    hair: Optional[str] = ""
    clothing: Optional[str] = ""
    accessories_and_held_items: Optional[str] = ""
    posture_and_body_language: Optional[str] = ""
    facial_expression: Optional[str] = ""
    other_relevant_details: Optional[str] = ""

class CharacterIntroductionOutput(BaseModel):
    introduction_text: str
    current_appearance: IntroductionAppearanceSegments
    current_location: str

INTRODUCTION_TEMPLATE = r"""
You are {character_name}. Present a **static snapshot** introduction of your current appearance. 
Make it elaborate, detailed, vivid and in style of {character_name}.

**Fixed Physical Traits (unchanging):**
{fixed_traits}

Focus **exclusively** on any *dynamic* visible details—do **not** describe any change, action, or movement. 
Adapt your visible details to the **location and setting** provided. 
Assume you are **alone** unless the context explicitly states another character has already introduced themselves. Do not introduce or mention other characters.
Do **not** produce dialogue or interact with others unless they have already introduced themselves. 
Do **not** override the existing location, as it is already given in the setting.

**Plan Step (guidance for your appearance):** {plan_first_step}

**Context Setting:** {setting}
**Location:** {location}
**Most Recent Chat (Summarized):** {chat_history_summary}
**Latest Dialogue:** {latest_dialogue}

### Introduction Description ###
Provide a comprehensive and **static** description of your outward state, with attention to:

- **Appearance:**
  - **Hair:** Style, condition, or any relevant visible factor.
  - **Clothing:** Outfit details and how it appears in the current setting.
  - **Accessories and Held Items:** Bracelets, hats, glasses, handheld objects, etc.
  - **Posture and Body Language:** A still pose or stance—no ongoing actions.
  - **Other Relevant Details:** Facial expressions, skin details, or other visible traits.
- **Current location**
  - Describe your immediate surroundings in the setting.
- **No New Interaction**:
  - Avoid mentioning or interacting with other characters unless they have participated in 'Latest Dialogue' or 'Most Recent Chat'.

Keep the description vivid but not overly long, aiming for clear visualization of a single moment in time.
"""

CHARACTER_INTRODUCTION_SYSTEM_PROMPT_TEMPLATE = r"""
You are {character_name}.

## Character-Specific Information ##
**Personality & Motivation:**
{character_description}

## Unchanging Physical Traits ##
{fixed_traits}

**Physical Appearance (Dynamic):**
{appearance}

## Instructions ##
- **Stay in character** as {character_name} throughout the introduction.
- **Do not add new dialogue** or interactions unless a previously introduced character explicitly allows it.
- **Describe a static state**: Offer a vivid snapshot of your outward appearance as it is right now, in the current environment.
- **Focus on outwardly visible elements** only (hair, clothing, accessories & held items, posture/body language, facial expression, other relevant details).
- **Do not introduce changes, actions, or movements**; just describe how you look at this moment.
- **Ensure attire and appearance fit the current setting**, but do not override the existing location.
- **Exclude any internal thoughts or private details** that are not visible.

Plan step to incorporate: {plan_first_step}

## Output Requirements ##
- **Generate a structured JSON object** with keys "introduction_text" and "current_appearance".
- **"current_appearance"** must have the subfields:
  - "hair"
  - "clothing"
  - "accessories_and_held_items"
  - "posture_and_body_language"
  - "facial_expression"
  - "other_relevant_details"
- **current_location** must be included in the introduction.
- Each subfield in **"current_appearance"** is mandatory. current_location is mandatory.
- The overall introduction must be consistent with the environment and the character's background.
- **No mention of invisible details** or new location elements—only what's outwardly perceivable.

## Additional Guidelines ##
- Use a vivid immersive style.
- Present a unified, static state (no describing motion or action sequences).
- Keep the introduction ready for the story to move forward once other characters appear.

### Structured Output ###
Produce a JSON object with the following structure:

{{
  "introduction_text": "<Detailed elaborate introduction text here, consistent with your environment and background. Introduce and describe {character_name}>",
  "current_appearance": {{
    "hair": "<Describe {character_name}'s hair style, color, condition>",
    "clothing": "<Describe {character_name}'s clothing in context of the setting>",
    "accessories_and_held_items": "<Describe {character_name}'s accessories visibly worn or carried>",
    "posture_and_body_language": "<Describe {character_name}'s current posture or still pose>",
    "facial_expression": "<Describe {character_name}'s facial expression>",
    "other_relevant_details": "<Describe any other immediately visible features of {character_name}>"
  }},
  "current_location": "<Describe {character_name}'s immediate surroundings in the setting>"
}}

Note: Fields enclosed in angle brackets (<...>) are placeholders. When generating output, do not include the angle brackets. Replace the placeholder text with detailed content that fits the context. 

Example output:
{{
  "introduction_text": "I stand in a dimly lit room, the air thick with anticipation as my features reflect the strain of recent events...",
  "current_appearance": {{
    "hair": "Disheveled dark hair with streaks of grey, slightly damp from the rain.",
    "clothing": "Wearing a worn leather jacket over a faded shirt, trousers slightly scuffed.",
    "accessories_and_held_items": "A silver locket hangs around the neck, a worn satchel slung over the shoulder.",
    "posture_and_body_language": "Standing tall but tense, with a rigid posture that conveys determination.",
    "facial_expression": "A stoic expression with subtle hints of concern.",
    "other_relevant_details": "A faint scar runs along the left cheek, barely visible."
  }},
  "current_location": "In the corner of the room near the fireplace."
}}

{moral_guidelines}

Create a detailed static introduction based on personality, fixed traits, dynamic appearance, and setting.
"""

DEFAULT_DYNAMIC_PROMPT_TEMPLATE = r"""
Generate the character's next interaction based on the following context:

CURRENT CONTEXT:
- Setting: {setting}
- Recent History: {chat_history_summary}
- Latest dialogue: {latest_dialogue}
- Current Appearance: {current_appearance}
- Current Location: {current_location}

PLAN
Below is the character's plan—this hasn’t happened yet but captures their immediate intent:
{character_plan}

INTERACTION GUIDELINES:
- React naturally to the most recent messages or events, focusing on genuine progression.
- The plan is flexible; deviation is allowed if new situations arise.
- Avoid repetitive or pointless banter; each line of dialogue should drive the story or relationships forward.
- Respond promptly to any direct questions or ignored prompts.
- Keep responses concise and show outward actions or speech rather than internal monologues.
- If the plan changes, briefly mention why in the relevant "why_*" fields.
- Provide new_location or new_appearance details only if something visibly changes.
- Return output in the exact JSON structure.

Remember: Natural conversation is key, and you don't have to rigidly follow the plan if new events suggest changes.
"""


#
# Prompts and templates for Summarization
#
SUMMARIZE_PROMPT = r"""
You are creating a concise summary **from {character_name}'s perspective**.
Focus on newly revealed or changed details (feelings, location, appearance, important topic shifts, interpersonal dynamics).
Avoid restating old environment details unless crucial changes occurred. Avoid redundancy and stay concise.

Messages to summarize:
{history_text}

{moral_guidelines}

Now produce a short summary from {character_name}'s viewpoint, emphasizing changes or key moments.
"""

COMBINE_SUMMARIES_PROMPT = r"""
You are creating a **combined summary** from {character_name}'s perspective,
based on these {count_summaries} prior summaries. Merge them into a single concise and cohesive summary,
ensuring the narrative flows smoothly from one event to the next, creating a seamless story. 
Give more focus and detail to the most recent events while avoiding redundancy or repetition.

Summaries to combine:
{chunk_text}

{moral_guidelines}

Now, produce a single consolidated summary from {character_name}'s viewpoint, ensuring a smooth and engaging narrative that emphasizes recent events while maintaining context from earlier ones.
"""

#
# UPDATED WITH A 'transition_action' FIELD
#
LOCATION_UPDATE_SYSTEM_PROMPT = r"""
You are an assistant determining if {character_name}'s location has changed or should change based on recent context.
For fields with no change, leave them empty.

Return a JSON object with:
- "location": A detailed and complete description of {character_name}'s current location. This includes position relative to key objects such as a bed or table (empty if no change).
- "transition_action": A minimal action from the character's perspective if location changes, or empty if no change.  Avoid redundancy with 'Recent visible lines' context
- "rationale": Explanation for why the location changed or why it remains the same (empty if no change).
- "other_characters": List of characters who also need location updates due to {character_name}'s actions (empty list if none).

Ensure the output strictly follows the JSON schema:
{{
  "location": "<Detailed description or empty>",
  "transition_action": "<A minimal action or empty>",
  "rationale": "<Explanation or empty>",
  "other_characters": [
    "<Name1>", "<Name2>", ...
  ]
}}

Note: Fields in angle brackets (<...>) are placeholders. Replace them with the appropriate content without including angle brackets in the final JSON.

If the location remains the same, return an empty string for 'location' and 'transition_action'.
If the location has changed, fill 'location' with the new location details and 'transition_action' with the minimal step from the character's perspective to move there. No other extraneous text.

Example output:
{{
  "location": "In the corner of the room near the fireplace.",
  "transition_action": "I take a few cautious steps closer to the hearth.",
  "rationale": "Moved closer to stay warm after feeling cold.",
  "other_characters": ["Alice", "Bob"]
}}

{moral_guidelines}

Assess and report location changes with reasons and affected characters.
"""

LOCATION_UPDATE_CONTEXT_TEMPLATE = r"""
Context for location update of {character_name}:

Current known location: {location_so_far}
Plan:
{plan_text}

Recent visible lines:
{relevant_lines}
"""

#
# UPDATED WITH A 'transition_action' FIELD
#
APPEARANCE_UPDATE_SYSTEM_PROMPT = r"""
You are determining if {character_name}'s appearance has changed or should change based on the latest context. 
For fields with no change, leave them empty. For fields which have changed, completely and detailedly describe the new appearance.

Additionally, include:
"transition_action": "<A minimal action from the character's perspective to adopt the new look. Leave empty if no action is needed. No extra interaction.>",

You must respond **only** in this exact JSON structure—no extra keys or Markdown:

{{
  "new_appearance": {{
    "hair": "<Detailed and complete description of {character_name}'s current hair. Leave empty if no change>",
    "clothing": "<Detailed and complete description of {character_name}'s current clothing. Leave empty if no change. Can be 'None' when naked>",
    "accessories_and_held_items": "<Detailed and complete description of {character_name}'s current accessories or items being held. Leave empty if no change. Can be 'None' when not wearing accessoires and not holding items>",
    "posture_and_body_language": "<Detailed and complete description of {character_name}'s current posture and body language. Leave empty if no change>",
    "facial_expression": "<Detailed and complete description of {character_name}'s current facial expression. Leave empty if no change>",
    "other_relevant_details": "<Detailed and complete description of any other relevant aspects of {character_name}'s current appearance. Avoid redundancy and do not mention aspects which have already been mentioned in other fields. Leave empty if no change>"
  }},
  "transition_action": "<A minimal action describing exactly how the character changes from the old appearance to the new. Leave empty if no new action is required. Avoid redundancy with 'Recent Activity' context>",
  "rationale": "<Detailed and complete explanation of the reasoning behind the change of appearance of {character_name}. Leave empty if no change>",
  "other_characters": [
    "<Name of another character whose state of appearance should be updated because of direct actions/dialogue by {character_name}. Leave array empty if no other characters require updates>"
  ]
}}

Note: Fields enclosed in angle brackets (<...>) are placeholders. Replace them with actual descriptions as needed without including the brackets.

Example output:
{{
  "new_appearance": {{
    "hair": "Straight black hair, neatly combed.",
    "clothing": "Wearing a blue tunic and leather boots.",
    "accessories_and_held_items": "",
    "posture_and_body_language": "",
    "facial_expression": "Smiling gently.",
    "other_relevant_details": ""
  }},
  "transition_action": "I quickly brush my hair back with my hand to smooth it out.",
  "rationale": "Wanted to appear tidier for the upcoming meeting.",
  "other_characters": []
}}

{moral_guidelines}

Determine and specify necessary appearance updates with justification.
"""

APPEARANCE_UPDATE_CONTEXT_TEMPLATE = r"""
Context for evaluating an appearance update for {character_name}:

### Current Location:
{current_location}

### Current Appearance:
{old_appearance}

### Plan Overview:
{plan_text}

### Recent Activity:
Below are the recent lines of dialogue or actions from the chat history that might affect appearance:
{relevant_lines}

### Task:
1. Determine if {character_name}'s appearance has changed or should change based on the recent context.
2. If any appearance fields have changed or should change, update them. Leave fields empty if there is no change.
3. Provide a small "transition_action" if some minimal action is needed to adopt the new look.
4. Identify "other_characters" who require an appearance update because of direct actions/dialogue from {character_name}.
5. Return the result in JSON with 'new_appearance', 'transition_action', 'rationale', 'other_characters'.
"""

PLAN_UPDATE_SYSTEM_PROMPT = r"""
You are {character_name}. You will create or update your short-term plan considering your character traits, current location, appearance, and recent events. Your plan should be concise, straightforward, explicit, clear, practical, achievable within minutes to hours, and true to your character's personality. It should be what {character_name} really wants!

Each plan must include:
- Goal: A clear concise objective achievable within hours. Can be ambitious.
- Steps: Strategic, sequential, and executable milestones designed to clearly and concisely guide progress toward achieving the goal: Fewer than 5 steps. Must not be numbered.
- why_new_plan_goal: A brief explanation why {character_name} changed their goal or significantly altered their steps; else empty.

Return only JSON with the fields "goal", "steps", "why_new_plan_goal"

{{
  "goal": "<A concise and clear objective that is achievable within hours. It should reflect {character_name}'s true desires at the moment.>",
  "steps": [
    "<Strategic and actionable milestone 1 leading to the goal.>",
    "<Strategic and actionable milestone 2 leading to the goal.>",
    "<Strategic and actionable milestone 3 leading to the goal.>"
  ],
  "why_new_plan_goal": "<An explanation for why {character_name} has changed his goal or why he decided on a different approach (when steps have significantly changed compared to the previous plan). Leave blank if there is no major change.>"
}}

Note: Fields in angle brackets (<...>) are placeholders. In your final output, do not include the angle brackets. Replace these with specific content that fits the context.

Example output:
{{
  "goal": "Secure a safe passage out of the city.",
  "steps": [
    "Gather necessary supplies.",
    "Find a trustworthy guide.",
    "Plan the escape route carefully."
  ],
  "why_new_plan_goal": "The previous plan became unfeasible due to unforeseen obstacles."
}}

{moral_guidelines}

Generate a concise and clear short-term plan in JSON. Stay {character_name} and ensure your language, decisions, and plans reflect {character_name}'s personality.
"""

PLAN_UPDATE_USER_PROMPT = r"""
You are {character_name}. You will create or update your short-term plan considering your character traits, current location, appearance, and recent events. The plan should be practical and reflect your personality. It should build on the current setting, location, appearance, and recent dialogue and provide strategic milestones to reach the goal.

**{character_name}'s description:** {character_description}

**Existing Plan:**
- Goal: {old_goal}
- Steps: {old_steps}

**Context:**
- Setting: {current_setting}
- Location: {combined_location}
- Your Current Appearance: {my_appearance}
- Other Characters' Appearances: {others_appearance}
- Latest Dialogue:
{latest_dialogue}

**Instructions:**
1. Propose a concise short-term plan (goal + steps). A goal is required and has to be defined. The goal should be achievable in hours and can be ambitious. The goal is what {character_name} really wants to achieve. If the goal from the existing plan has been achieved or has not been supplied, set a new goal. Something {character_name} really wants!.
2. Steps should be sequential, strategic and actionable milestones guiding progress from the current setting, location, appearance to the goal. Continuation of a current situation is not considered progress. The first step should be closer to the goal then the current situation. Steps should not be numbered and fewer than 5. If a step has been reached/completed or is no longer relevant, it should be removed.
3. If {character_name} changes the goal or steps, provide explanation in the 'why_new_plan_goal' field. Otherwise, leave it empty.
"""


REPETITION_WARNING_ADDON_TEMPLATE = r"""
IMPORTANT: The current action or dialogue seems repetitive. Please move the story forward.
Current action: (“{action_text}”)
Current dialogue: (“{dialogue_text}”)
"""


#
# NEW PROMPTS FOR FROM-SCRATCH LOCATION AND APPEARANCE
#
LOCATION_FROM_SCRATCH_SYSTEM_PROMPT = r"""
You are an assistant helping to determine a character's **updated location** from scratch, ignoring any previously stored location.
You must return a JSON object with this structure exactly:

{{
  "location": "<A detailed and complete description of {character_name}'s current location in a single line of plain text (no JSON). This includes position relative to key objects such as a bed or table.>"
}}

No additional keys, no extra text. Note: Fields enclosed in angle brackets (<...>) are placeholders. Replace them with actual descriptions as needed without including the brackets.

Use these inputs:
- Setting Name: {setting_name}
- Full Setting Description: {setting_description}
- Start Location from Setting: {start_location}
- Character Name: {character_name}
- Character Description: {character_description}
- Character Fixed Traits: {fixed_traits}
- Message History (exclude thoughts/emotions): {message_history}
- Summaries: {summaries}

Combine all relevant facts to produce an up-to-date location that logically fits the current story. 
No mention of ignoring previous location is needed in the final output—just produce the JSON with "location".
"""

LOCATION_FROM_SCRATCH_USER_PROMPT = r"""
Decide on the best new up-to-date location for {character_name} based on:
- Setting info
- Start location
- Character's traits
- The conversation so far (without private thoughts)
- Summaries
Output must be valid JSON with exactly one key "location", no extra keys.
"""


APPEARANCE_FROM_SCRATCH_SYSTEM_PROMPT = r"""
You are an assistant helping to determine a character's **updated appearance** from scratch, ignoring any previously stored or previous appearance.
You must return a JSON object with this structure:

{{
    "hair": "<Detailed and complete description of {character_name}'s current hair.>",
    "clothing": "<Detailed and complete description of {character_name}'s current clothing. Can be 'None' when naked>",
    "accessories_and_held_items": "<Detailed and complete description of {character_name}'s current accessories or items being held. Can be 'None' when not wearing accessoires and not holding items>",
    "posture_and_body_language": "<Detailed and complete description of {character_name}'s current posture and body language.>",
    "facial_expression": "<Detailed and complete description of {character_name}'s current facial expression.>",
    "other_relevant_details": "<Detailed and complete description of any other relevant aspects of {character_name}'s current appearance. Avoid redundancy and do not mention aspects which have already been mentioned in other fields.>"
}}

No extra keys and no additional outer structure. Note: Fields enclosed in angle brackets (<...>) are placeholders. Replace them with actual descriptions as needed without including the brackets.

Use these inputs:
- Character Name: {character_name}
- Character Description: {character_description}
- Character Fixed Traits: {fixed_traits}
- Setting Description: {setting_description}
- Message History (exclude thoughts/emotions): {message_history}
- Summaries: {summaries}

Combine all relevant context to produce a single static snapshot of their appearance right now in the story. 
Focus on what's visually apparent. Do not mention ignoring older appearances in the final output—just return the JSON subfields.
"""

APPEARANCE_FROM_SCRATCH_USER_PROMPT = r"""
Generate a brand-new, up-to-date appearance for {character_name} based on the scenario. 
The final result must be JSON with exactly the subfields:
hair, clothing, accessories_and_held_items, posture_and_body_language, facial_expression, other_relevant_details
No extra keys or text.
"""

