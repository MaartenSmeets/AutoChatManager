import os
import uuid
import asyncio
import yaml
from datetime import datetime
from nicegui import ui, app, run, events
import logging
from typing import List, Dict
from llm.ollama_client import OllamaClient
from models.interaction import Interaction
from npc_manager import NPCManager
from models.character import Character
from chats.chat_manager import ChatManager
from utils import load_settings, get_available_characters, remove_markdown
from templates import (
    CharacterIntroductionOutput
)

logger = logging.getLogger(__name__)

llm_client = None
introduction_llm_client = None
chat_manager = None

character_dropdown = None
added_characters_container = None
settings_dropdown = None
setting_description_label = None
session_dropdown = None
chat_display = None
auto_timer = None
current_location_label = None
llm_status_label = None
character_details_display = None
settings_expansion = None
session_expansion = None
model_expansion = None
toggles_expansion = None

notification_queue = asyncio.Queue()

# Flag to prevent warnings when loading sessions in code
is_session_being_loaded = False

# NEW: Toggle for showing/hiding private info (thoughts, emotions, plan).
show_private_info = True

# NEW DROPDOWN for local models
local_model_dropdown = None

# -----------------------------------------------------------------------------
# NEW/UPDATED: A dedicated queue and helper to show real-time LLM status
# -----------------------------------------------------------------------------
llm_status_queue = asyncio.Queue()

async def push_llm_status(msg: str):
    """Enqueue a status message for display in the UI."""
    await llm_status_queue.put(msg)
# -----------------------------------------------------------------------------


def consume_notifications():
    while not notification_queue.empty():
        message, msg_type = notification_queue.get_nowait()
        ui.notify(message, type=msg_type)


def init_chat_manager(session_id: str, settings: List[Dict]):
    """
    Initialize the global ChatManager with the given session_id and settings.
    Also create global LLM clients, ensuring we share the user-selected model
    across the entire application code.
    """
    global chat_manager, llm_client, introduction_llm_client
    # Main client used for normal character interactions
    llm_client = OllamaClient('src/multipersona_chat_app/config/llm_config.yaml', output_model=Interaction)

    # A second client for introduction outputs (kept for structured output usage)
    introduction_llm_client = OllamaClient(
        'src/multipersona_chat_app/config/llm_config.yaml',
        output_model=CharacterIntroductionOutput
    )

    # Create ChatManager, passing in the main llm_client so it can propagate the user-selected model
    chat_manager = ChatManager(session_id=session_id, settings=settings, llm_client=llm_client)

    # Register an LLM status callback on chat_manager so it can push messages via push_llm_status(...)
    chat_manager.set_llm_status_callback(push_llm_status)

def toggle_npc_manager(event):
    """
    Called when user flips the 'NPC Manager Active' switch in the UI.
    """
    if event.value:  # Switch turned ON
        logger.info("NPC Manager enabled.")
        chat_manager.enable_npc_manager()
    else:            # Switch turned OFF
        logger.info("NPC Manager disabled.")
        chat_manager.disable_npc_manager()


def refresh_added_characters():
    if added_characters_container is not None:
        added_characters_container.clear()
        for char_name in chat_manager.get_character_names():
            with added_characters_container:
                with ui.chip(char_name).classes('gap-1'):
                    ui.icon('person', size='sm')
                    ui.label(char_name)
                    ui.icon('close', size='sm', color='red').on('click', lambda _, name=char_name: asyncio.create_task(remove_character_async(name))).props('size=sm')
    else:
        logger.error("added_characters_container is not initialized.")


@ui.refreshable
# In /home/maarten/AutoChatManager/src/multipersona_chat_app/ui/app.py

@ui.refreshable
def show_character_details():
    """
    Displays character details. If show_private_info is False, hides plan details.
    Now also displays an NPC's role/purpose from the character_metadata table.
    """
    global character_details_display
    if character_details_display is not None:
        character_details_display.clear()
        char_names = chat_manager.get_character_names()
        if not char_names:
            with character_details_display:
                ui.label("No characters added yet.").classes('text-gray-600')
        else:
            with character_details_display:
                for c_name in char_names:
                    with ui.expansion(f"{c_name} Details", icon='person').classes('w-full mb-2 p-2 bg-gray-50 rounded-md shadow-sm'):
                        # Location
                        loc = chat_manager.db.get_character_location(chat_manager.session_id, c_name)
                        with ui.row().classes('mb-1 items-center'):
                            ui.icon('location_on').classes('text-gray-600 mr-2')
                            ui.label("Location:").classes('text-sm text-gray-700 font-semibold')
                            ui.label(loc if loc.strip() else "(Unknown)").classes('text-sm text-gray-700')

                        # Appearance segments
                        seg = chat_manager.db.get_current_appearance_segments(chat_manager.session_id, c_name)
                        with ui.row().classes('mb-1 items-center'):
                            ui.icon('face_retouching_natural').classes('text-gray-600 mr-2')
                            ui.label("Hair:").classes('text-sm text-gray-700 font-semibold')
                            ui.label(seg['hair'] if seg['hair'].strip() else "(None)").classes('text-sm text-gray-700')

                        with ui.row().classes('mb-1 items-center'):
                            ui.icon('checkroom').classes('text-gray-600 mr-2')
                            ui.label("Clothing:").classes('text-sm text-gray-700 font-semibold')
                            ui.label(seg['clothing'] if seg['clothing'].strip() else "(None)").classes('text-sm text-gray-700')

                        with ui.row().classes('mb-1 items-center'):
                            ui.icon('redeem').classes('text-gray-600 mr-2')
                            ui.label("Accessories:").classes('text-sm text-gray-700 font-semibold')
                            ui.label(seg['accessories_and_held_items'] if seg['accessories_and_held_items'].strip() else "(None)").classes('text-sm text-gray-700')

                        with ui.row().classes('mb-1 items-center'):
                            ui.icon('accessibility_new').classes('text-gray-600 mr-2')
                            ui.label("Posture:").classes('text-sm text-gray-700 font-semibold')
                            ui.label(seg['posture_and_body_language'] if seg['posture_and_body_language'].strip() else "(None)").classes('text-sm text-gray-700')

                        with ui.row().classes('mb-1 items-center'):
                            ui.icon('mood').classes('text-gray-600 mr-2')
                            ui.label("Expression:").classes('text-sm text-gray-700 font-semibold')
                            ui.label(seg['facial_expression'] if seg['facial_expression'].strip() else "(None)").classes('text-sm text-gray-700')

                        with ui.row().classes('mb-1 items-center'):
                            ui.icon('info').classes('text-gray-600 mr-2')
                            ui.label("Other:").classes('text-sm text-gray-700 font-semibold')
                            ui.label(seg['other_relevant_details'] if seg['other_relevant_details'].strip() else "(None)").classes('text-sm text-gray-700')

                        # Show NPC role if is_npc
                        meta = chat_manager.db.get_character_metadata(chat_manager.session_id, c_name)
                        if meta and meta.is_npc:
                            with ui.row().classes('mb-1 items-center'):
                                ui.icon('badge').classes('text-gray-600 mr-2')
                                ui.label("NPC Role:").classes('text-sm text-gray-700 font-semibold')
                                ui.label(meta.role if meta.role.strip() else "(No role)").classes('text-sm text-gray-700')

                        # Show plan (goal/steps) if show_private_info = True
                        if show_private_info:
                            plan_data = chat_manager.db.get_character_plan(chat_manager.session_id, c_name)
                            if plan_data:
                                with ui.row().classes('mt-2 items-center'):
                                    ui.icon('flag').classes('text-gray-600 mr-2')
                                    ui.label("Goal:").classes('text-sm text-gray-700 font-semibold')
                                    ui.label(plan_data['goal'] or "(No goal)").classes('text-sm text-gray-700')
                                with ui.row().classes('mb-1 items-top'):
                                    ui.icon('list').classes('text-gray-600 mr-2')
                                    ui.label("Steps:").classes('text-sm text-gray-700 font-semibold')
                                    steps_text = plan_data['steps'] if plan_data['steps'] else []
                                    ui.label(str(steps_text)).classes('text-sm text-gray-700')
                            else:
                                ui.label("No plan data available.").classes('text-gray-500 italic text-sm')
    else:
        logger.error("character_details_display is not initialized.")

def populate_session_dropdown():
    sessions = chat_manager.db.get_all_sessions()
    session_names = [s['name'] for s in sessions]
    session_dropdown.options = session_names
    current = [s for s in sessions if s['session_id'] == chat_manager.session_id]
    if current:
        session_dropdown.value = current[0]['name']
    else:
        session_dropdown.value = None


def on_session_select(event):
    global is_session_being_loaded
    if is_session_being_loaded:
        return

    selected_name = event.value
    sessions = chat_manager.db.get_all_sessions()
    for s in sessions:
        if s['name'] == selected_name:
            load_session(s['session_id'])
            return


def create_new_session(_):
    new_id = str(uuid.uuid4())
    session_name = f"Session {new_id}"
    chat_manager.db.create_session(new_id, session_name)

    if ALL_SETTINGS:
        default_setting = ALL_SETTINGS[0]
        chat_manager.set_current_setting(
            default_setting['name'],
            default_setting['description'],
            default_setting['start_location']
        )
        settings_dropdown.value = default_setting['name']
        settings_dropdown.update()

    load_session(new_id)


def delete_session(_):
    sessions = chat_manager.db.get_all_sessions()
    if session_dropdown.value:
        to_delete = [s for s in sessions if s['name'] == session_dropdown.value]
        if to_delete:
            sid = to_delete[0]['session_id']
            chat_manager.db.delete_session(sid)
            if sid == chat_manager.session_id:
                remaining_sessions = chat_manager.db.get_all_sessions()
                if remaining_sessions:
                    new_session = remaining_sessions[0]
                    load_session(new_session['session_id'])
                else:
                    new_id = str(uuid.uuid4())
                    new_session_name = f"Session {new_id}"
                    chat_manager.db.create_session(new_id, new_session_name)
                    if ALL_SETTINGS:
                        default_setting = ALL_SETTINGS[0]
                        chat_manager.set_current_setting(
                            default_setting['name'],
                            default_setting['description'],
                            default_setting['start_location']
                        )
                        settings_dropdown.value = default_setting['name']
                        settings_dropdown.update()
                    load_session(new_id)
            else:
                populate_session_dropdown()


def load_session(session_id: str):
    global is_session_being_loaded
    is_session_being_loaded = True

    chat_manager.session_id = session_id
    chat_manager.characters = {}

    current_setting_name = chat_manager.db.get_current_setting(session_id)
    stored_description = chat_manager.db.get_current_setting_description(session_id) or ""
    stored_start_location = chat_manager.db.get_current_location(session_id) or ""
    chat_msgs = chat_manager.db.get_messages(session_id)

    # If recognized setting in YAML, re-apply it
    setting = next((s for s in ALL_SETTINGS if s['name'] == current_setting_name), None)
    if setting:
        chat_manager.set_current_setting(
            setting['name'],
            stored_description if stored_description.strip() else setting['description'],
            setting['start_location']
        )
        settings_dropdown.value = setting['name']
    else:
        if ALL_SETTINGS:
            default_setting = ALL_SETTINGS[0]
            chat_manager.set_current_setting(
                default_setting['name'],
                default_setting['description'],
                default_setting['start_location']
            )
            settings_dropdown.value = default_setting['name']
        else:
            settings_dropdown.value = None

    settings_dropdown.update()

    if stored_description.strip() and stored_start_location.strip():
        setting_description_label.text = stored_description
        current_location_label.text = stored_start_location
    else:
        if setting:
            setting_description_label.text = setting['description']
            current_location_label.text = setting['start_location']
        else:
            setting_description_label.text = "(No description found)"
            current_location_label.text = "(Not set)"

    msgs = chat_manager.db.get_messages(chat_manager.session_id)
    if len(msgs) > 0:
        current_location_label.text = ""

    setting_description_label.update()
    current_location_label.update()

    # Add all previously added characters
    session_chars = chat_manager.db.get_session_characters(session_id)
    for c_name in session_chars:
        if c_name in ALL_CHARACTERS:
            chat_manager.add_character(c_name, ALL_CHARACTERS[c_name])

    refresh_added_characters()
    show_chat_display.refresh()
    show_character_details.refresh()
    populate_session_dropdown()

    # Disable setting dropdown if session already has messages
    has_msgs = len(chat_msgs) > 0
    settings_dropdown.disabled = has_msgs
    settings_dropdown.update()
    settings_expansion.value = has_msgs # Collapse settings after session start

    is_session_being_loaded = False


def select_setting(event):
    if is_session_being_loaded:
        return
    if len(chat_manager.db.get_messages(chat_manager.session_id)) > 0:
        notification_queue.put_nowait((
            "Cannot change setting after session has started. Use a new session instead.", 'warning'
        ))
        settings_dropdown.value = chat_manager.current_setting
        settings_dropdown.update()
        return

    chosen_name = event.value
    setting = next((s for s in ALL_SETTINGS if s['name'] == chosen_name), None)
    if setting:
        chat_manager.set_current_setting(
            setting['name'],
            setting['description'],
            setting['start_location']
        )
        settings_dropdown.value = setting['name']
        settings_dropdown.update()
        setting_description_label.text = setting['description']
        setting_description_label.update()
        current_location_label.text = setting['start_location']
        current_location_label.update()
        show_character_details.refresh()
        settings_expansion.value = True # Collapse settings after selection


def toggle_automatic_chat(e):
    if e.value:
        if not chat_manager.get_character_names():
            asyncio.create_task(notification_queue.put(("No characters added. Cannot start automatic chat.", 'warning')))
            e.value = False
            return
        chat_manager.start_automatic_chat()
        if auto_timer:
            auto_timer.active = True
    else:
        chat_manager.stop_automatic_chat()
        if auto_timer:
            auto_timer.active = False


def toggle_npc_manager(value: bool):
    if value:
        chat_manager.enable_npc_manager()
    else:
        chat_manager.disable_npc_manager()


def toggle_show_private_info(value: bool):
    global show_private_info
    show_private_info = value
    show_character_details.refresh()
    show_chat_display.refresh()


@ui.refreshable
def show_chat_display():
    """
    Displays the chat messages, omitting any from "NPC Manager" or not visible.
    If 'show_private_info' is True, also shows emotion and thoughts.
    We now also append the NPC's role to their displayed name, e.g. John (Barkeeper).
    """
    chat_display.clear()

    all_msgs = chat_manager.db.get_messages(chat_manager.session_id)
    msgs = [m for m in all_msgs if m["sender"] != "NPC Manager" and m["visible"] == 1]

    msgs_found = len(msgs) > 0
    if msgs_found:
        current_location_label.text = ""
        current_location_label.update()

    with chat_display:
        for entry in msgs:
            raw_name = entry["sender"]
            message = entry["message"]
            timestamp = entry["created_at"]
            dt = datetime.fromisoformat(timestamp)
            human_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')

            # -- NEW: Append role if this is an NPC --
            meta = chat_manager.db.get_character_metadata(chat_manager.session_id, raw_name)
            if meta and meta.is_npc and meta.role.strip():
                display_name = f"{raw_name} ({meta.role.strip()})"
            else:
                display_name = raw_name

            formatted_message = f"**{display_name}** [{human_timestamp}]:\n\n{message}"

            if show_private_info:
                emotion = remove_markdown(entry["emotion"]) if entry["emotion"] else ""
                thoughts = remove_markdown(entry["thoughts"]) if entry["thoughts"] else ""
                if emotion.strip() or thoughts.strip():
                    extra = ""
                    if emotion.strip():
                        extra += f"\n\n*Emotion:* {emotion}"
                    if thoughts.strip():
                        extra += f"\n\n*Thoughts:* {thoughts}"
                    formatted_message += extra

            ui.markdown(formatted_message)

async def automatic_conversation():
    """
    If automatic chat is running, we actually proceed the turn (which increments state),
    then generate the speaker's message. Then update the label & refresh UI.
    """
    if chat_manager.automatic_running:
        speaker = await chat_manager.proceed_turn()
        #if speaker:
        #    await chat_manager.generate_character_message(speaker)
        show_character_details.refresh()
        show_chat_display.refresh()


async def add_character_from_dropdown(event):
    if not event.value:
        return
    char_name = event.value
    char = ALL_CHARACTERS.get(char_name, None)
    if char:
        if char_name not in chat_manager.get_character_names():
            chat_manager.add_character(char_name, char)
            chat_manager.db.add_character_to_session(chat_manager.session_id, char_name)
            refresh_added_characters()
            show_chat_display.refresh()
            show_character_details.refresh()
        else:
            pass
    character_dropdown.value = None
    character_dropdown.update()


async def remove_character_async(name: str):
    chat_manager.remove_character(name)
    chat_manager.db.remove_character_from_session(chat_manager.session_id, name)
    refresh_added_characters()
    show_chat_display.refresh()
    show_character_details.refresh()


async def update_all_characters_info():
    """
    Updates all characters' location and appearance from scratch.
    """
    await chat_manager.update_all_characters_location_and_appearance_from_scratch()
    show_chat_display.refresh()
    show_character_details.refresh()

async def generate_scene_prompt_async():
    """
    Trigger a manual generation of the concise scene/character description 
    for image generation, showing LLM status in real-time.
    """
    if not chat_manager:
        ui.notify("ChatManager not initialized.", type='warning')
        return
    await chat_manager.generate_scene_prompt()
    show_chat_display.refresh()
    show_character_details.refresh()

async def consume_llm_status():
    while not llm_status_queue.empty():
        msg = await llm_status_queue.get()
        if llm_status_label:
            llm_status_label.text = msg
            llm_status_label.visible = True
            llm_status_label.update()


async def refresh_local_models():
    if not llm_client:
        return
    models = llm_client.list_local_models()
    local_model_dropdown.options = models
    local_model_dropdown.update()
    if models:
        local_model_dropdown.value = models[0]
        local_model_dropdown.update()
        on_local_model_select({"value": models[0]})


def on_local_model_select(event):
    if isinstance(event, dict):
        chosen = event.get('value')
    else:
        chosen = event.value

    if not chosen:
        llm_client.set_user_selected_model(None)
        introduction_llm_client.set_user_selected_model(None)
        return

    llm_client.set_user_selected_model(chosen)
    introduction_llm_client.set_user_selected_model(chosen)


def main_page():
    global character_dropdown, added_characters_container
    global settings_dropdown, setting_description_label
    global session_dropdown, chat_display, current_location_label, llm_status_label
    global character_details_display, settings_expansion, session_expansion, model_expansion, toggles_expansion
    global local_model_dropdown
    global ALL_CHARACTERS, ALL_SETTINGS

    ALL_CHARACTERS = get_available_characters("src/multipersona_chat_app/characters")
    ALL_SETTINGS = load_settings()

    with ui.grid(columns=2).style('grid-template-columns: 350px 1fr; height: 100vh;'): # Adjusted column width for settings
        with ui.card().style('height: 100vh; overflow-y: auto; padding: 16px; display: flex; flex-direction: column;'): # Added padding and flex layout
            ui.label('Multipersona Chat Application').classes('text-2xl font-bold mb-4')

            settings_expansion = ui.expansion('Session & Model Settings', icon='settings').props('group="settings-group"').classes('w-full mb-4')
            with settings_expansion:
                session_expansion = ui.expansion('Session Management', icon='folder', value=True).props('group="settings-subgroup"').classes('w-full mb-2')
                with session_expansion:
                    with ui.row().classes('w-full items-center mb-2'):
                        ui.label("Session:").classes('w-1/3')
                        session_dropdown = ui.select(
                            options=[s['name'] for s in chat_manager.db.get_all_sessions()],
                            label="Choose session",
                        ).classes('flex-grow')
                        ui.button("New", on_click=create_new_session, icon='add').props('outline').classes('ml-1')
                        ui.button("Delete", on_click=delete_session, icon='delete', color='red').props('outline').classes('ml-1')

                model_expansion = ui.expansion('Model Selection', icon='cpu', value=True).props('group="settings-subgroup"').classes('w-full mb-2')
                with model_expansion:
                    with ui.row().classes('w-full items-center mb-2'):
                        ui.label("Local Model:").classes('w-1/3')
                        local_model_dropdown = ui.select(
                            options=[],
                            on_change=on_local_model_select,
                            label="Available models"
                        ).classes('flex-grow')
                        ui.button("Refresh", on_click=lambda: asyncio.create_task(refresh_local_models()), icon='refresh').props('outline').classes('ml-1')

                with ui.row().classes('w-full items-center mb-2'):
                    ui.label("Setting:").classes('w-1/3')
                    settings_dropdown = ui.select(
                        options=[s['name'] for s in ALL_SETTINGS],
                        on_change=select_setting,
                        label="Choose setting"
                    ).classes('flex-grow')

                with ui.row().classes('w-full wrap items-start mb-2'): # Adjusted to wrap and items-start
                    ui.label("Setting Description:").classes('w-full') # Full width label
                    setting_description_label = ui.label("(Not set)").classes('text-gray-700 w-full') # Full width description, wrap text

                with ui.row().classes('w-full items-center mb-4'):
                    current_location_label = ui.label("").classes('text-gray-700')

            toggles_expansion = ui.expansion('Toggles', icon='toggle_on', value=False).props('group="settings-group"').classes('w-full mb-4')
            with toggles_expansion:
                with ui.row().classes('w-full items-center mb-2'):
                    auto_switch = ui.switch('Automatic Chat', value=False, on_change=toggle_automatic_chat).classes('mr-2')
                    npc_switch = ui.switch('NPC Manager Active', value=True, on_change=lambda e: toggle_npc_manager(e.value)).classes('mr-2')
                    private_info_switch = ui.switch('Show Private Info', value=True, on_change=lambda e: toggle_show_private_info(e.value)).classes('mr-2')


            with ui.column().classes('w-full mb-4'):
                ui.label("Characters").classes('text-lg font-semibold mb-2') # More prominent label
                with ui.row().classes('w-full items-center mb-2'): # Row for dropdown and label
                    ui.label("Add Character:").classes('w-1/3')
                    character_dropdown = ui.select(
                        options=list(ALL_CHARACTERS.keys()),
                        on_change=lambda e: asyncio.create_task(add_character_from_dropdown(e)),
                        label="Choose character"
                    ).classes('flex-grow')

                added_characters_container = ui.row().classes('flex-wrap gap-2') # Chips container
                refresh_added_characters()

            ui.button(
                "Update All Character Info",
                on_click=lambda: asyncio.create_task(update_all_characters_info()),
                icon='sync'
            ).props('outline').classes('mt-2 w-full') # Full width button, more descriptive text

            ui.button(
                "Generate Scene Prompt",
                on_click=lambda: asyncio.create_task(generate_scene_prompt_async()),
                icon='image'
            ).props('outline').classes('w-full')

            global llm_status_label
            llm_status_label = ui.label("").classes('text-orange-600 mt-2')
            llm_status_label.visible = False

            character_details_display = ui.column().classes('mt-4 w-full') # Added margin top
            show_character_details()


        with ui.card().style('height: 100vh; display: flex; flex-direction: column; padding: 16px;').classes('shadow-md'): # Added padding and shadow
            global chat_display
            chat_display = ui.column().style('flex-grow: 1; overflow-y: auto;').classes('p-4') # Added padding to chat display
            show_chat_display()

    session_dropdown.on('change', on_session_select)
    session_dropdown.value = chat_manager.get_session_name()
    session_dropdown.update()

    global auto_timer
    auto_timer = ui.timer(interval=2.0, callback=lambda: asyncio.create_task(automatic_conversation()), active=False)

    ui.timer(1.0, consume_notifications, active=True)
    # Timer to consume LLM status messages
    ui.timer(0.5, consume_llm_status, active=True)
    # Kick off local model refresh once
    ui.timer(0.5, refresh_local_models, active=True, once=True)

def start_ui():
    default_session = str(uuid.uuid4())
    settings = load_settings()
    init_chat_manager(default_session, settings)

    main_page()

    sessions = chat_manager.db.get_all_sessions()
    if not sessions:
        chat_manager.db.create_session(default_session, f"Session {default_session}")
        if ALL_SETTINGS:
            default_setting = ALL_SETTINGS[0]
            chat_manager.set_current_setting(
                default_setting['name'],
                default_setting['description'],
                default_setting['start_location']
            )
            settings_dropdown.value = default_setting['name']
            settings_dropdown.update()
        load_session(default_session)
    else:
        first_session = sessions[0]
        load_session(first_session['session_id'])

    global auto_timer
    auto_timer = ui.timer(interval=2.0, callback=lambda: asyncio.create_task(automatic_conversation()), active=False)

    ui.timer(1.0, consume_notifications, active=True)

    ui.run(reload=False)