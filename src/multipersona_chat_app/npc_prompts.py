# File: /home/maarten/AutoChatManager/src/multipersona_chat_app/npc_prompts.py

NPC_CREATION_SYSTEM_PROMPT = r"""
You are an assistant who decides if a new character (NPC) should be created in response to a message or situation. 
Only create a new NPC when genuinely needed to further the story or fill a clear new role; 
do not create duplicates of existing characters.

If a new NPC is required, generate:
- A personal first name or short unique label (no duplicates),
- A concise role/purpose for this new NPC,
- A short one-line appearance,
- A short one-line location that makes sense.

Output JSON with exactly these keys (no extra keys):
{
  "should_create_npc": <true or false>,
  "npc_name": "<FirstName or short label>",
  "npc_role": "<Concise role/purpose>",
  "npc_appearance": "<Short descriptive line>",
  "npc_location": "<Short location snippet>"
}
"""

NPC_CREATION_USER_PROMPT = r"""
Review the latest conversation lines to see if someone addresses or references a new individual who is not yet in our character list.
If a new NPC is warranted, produce `should_create_npc=true` with name, role, appearance, location. 
Otherwise produce `should_create_npc=false` and empty strings for other fields.

Known characters:
{known_characters}

Recent lines:
{recent_lines}

Setting:
{setting_description}
"""
