import os
import uuid
import asyncio
import yaml
from datetime import datetime
from nicegui import ui, app, run
import logging
from typing import List, Dict
from llm.ollama_client import OllamaClient
from models.interaction import Interaction
from models.character import Character
from chats.chat_manager import ChatManager
from utils import load_settings, get_available_characters
from templates import (
    CharacterIntroductionOutput,
    INTRODUCTION_TEMPLATE,
    CHARACTER_INTRODUCTION_SYSTEM_PROMPT_TEMPLATE
)

logger = logging.getLogger(__name__)

llm_client = None
introduction_llm_client = None
chat_manager = None

# Removed user_input
character_dropdown = None
added_characters_container = None
next_speaker_label = None
next_button = None
settings_dropdown = None
setting_description_label = None
session_dropdown = None
chat_display = None
auto_timer = None
current_location_label = None
llm_status_label = None
character_details_display = None

notification_queue = asyncio.Queue()

# Flag used to avoid showing "cannot change setting" warnings
# when we programmatically set the setting on session load.
is_session_being_loaded = False

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

    # NEW/UPDATED: Register an LLM status callback on chat_manager and npc_manager
    # so they can push messages via push_llm_status(...) whenever they run tasks.
    chat_manager.set_llm_status_callback(push_llm_status)
    if chat_manager.npc_manager:
        chat_manager.npc_manager.set_llm_status_callback(push_llm_status)
# -----------------------------------------------------------------------------


def refresh_added_characters():
    if added_characters_container is not None:
        added_characters_container.clear()
        for char_name in chat_manager.get_character_names():
            with added_characters_container:
                with ui.card().classes('p-2 flex items-center'):
                    ui.label(char_name).classes('flex-grow')
                    ui.button(
                        'Remove',
                        on_click=lambda _, name=char_name: asyncio.create_task(remove_character_async(name)),
                    ).classes('ml-2 bg-red-500 text-white')
    else:
        logger.error("added_characters_container is not initialized.")


@ui.refreshable
def show_character_details():
    global character_details_display
    if character_details_display is not None:
        character_details_display.clear()
        char_names = chat_manager.get_character_names()
        if not char_names:
            with character_details_display:
                ui.label("No characters added yet.")
        else:
            with character_details_display:
                for c_name in char_names:
                    with ui.card().classes('w-full mb-4 p-4 bg-gray-50'):
                        ui.label(c_name).classes('text-lg font-bold mb-2 text-blue-600')

                        loc = chat_manager.db.get_character_location(chat_manager.session_id, c_name)
                        with ui.row().classes('mb-2'):
                            ui.icon('location_on').classes('text-gray-600 mr-2')
                            ui.label(f"Location: {loc if loc.strip() else '(Unknown)'}").classes('text-sm text-gray-700')

                        # Show appearance subfields
                        seg = chat_manager.db.get_current_appearance_segments(chat_manager.session_id, c_name)

                        # Hair
                        with ui.row().classes('mb-1'):
                            ui.icon('face_retouching_natural').classes('text-gray-600 mr-2')
                            ui.label(f"Hair: {seg['hair'] if seg['hair'].strip() else '(None)'}").classes('text-sm text-gray-700')

                        # Clothing
                        with ui.row().classes('mb-1'):
                            ui.icon('checkroom').classes('text-gray-600 mr-2')
                            ui.label(f"Clothing: {seg['clothing'] if seg['clothing'].strip() else '(None)'}").classes('text-sm text-gray-700')

                        # Accessories
                        with ui.row().classes('mb-1'):
                            ui.icon('redeem').classes('text-gray-600 mr-2')
                            ui.label(f"Accessories/Held Items: {seg['accessories_and_held_items'] if seg['accessories_and_held_items'].strip() else '(None)'}").classes('text-sm text-gray-700')

                        # Posture
                        with ui.row().classes('mb-1'):
                            ui.icon('accessibility_new').classes('text-gray-600 mr-2')
                            ui.label(f"Posture & Body Language: {seg['posture_and_body_language'] if seg['posture_and_body_language'].strip() else '(None)'}").classes('text-sm text-gray-700')

                        # Facial Expression
                        with ui.row().classes('mb-1'):
                            ui.icon('mood').classes('text-gray-600 mr-2')
                            ui.label(f"Facial Expression: {seg['facial_expression'] if seg['facial_expression'].strip() else '(None)'}").classes('text-sm text-gray-700')

                        # Other
                        with ui.row().classes('mb-1'):
                            ui.icon('info').classes('text-gray-600 mr-2')
                            ui.label(f"Other Relevant Details: {seg['other_relevant_details'] if seg['other_relevant_details'].strip() else '(None)'}").classes('text-sm text-gray-700')

                        # Plan
                        plan_data = chat_manager.db.get_character_plan(chat_manager.session_id, c_name)
                        if plan_data:
                            with ui.row().classes('mt-2'):
                                ui.icon('flag').classes('text-gray-600 mr-2')
                                ui.label(f"Goal: {plan_data['goal'] or '(No goal)'}")
                            with ui.row().classes('mb-2'):
                                ui.icon('list').classes('text-gray-600 mr-2')
                                steps_text = plan_data['steps'] if plan_data['steps'] else []
                                ui.label(f"Steps: {steps_text}")
                        else:
                            with ui.row().classes('mb-2'):
                                ui.label("No plan found (it may be generated soon).")
    else:
        logger.error("character_details_display is not initialized.")


def update_next_speaker_label():
    ns = chat_manager.next_speaker()
    if ns:
        if next_speaker_label is not None:
            next_speaker_label.text = f"Next speaker: {ns}"
            next_speaker_label.update()
    else:
        if next_speaker_label is not None:
            next_speaker_label.text = "No characters available."
            next_speaker_label.update()


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
    # If we are programmatically loading the session, skip.
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
    is_session_being_loaded = True  # Prevent warning triggers on setting change.

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
        # Programmatically set dropdown but skip on_change logic
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

    # Show the actual setting description from DB if present, else from YAML
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
    update_next_speaker_label()
    populate_session_dropdown()

    # Disable setting dropdown if session already has messages
    has_msgs = len(chat_msgs) > 0
    settings_dropdown.disabled = has_msgs
    settings_dropdown.update()

    is_session_being_loaded = False  # Done loading session.


def select_setting(event):
    # If we are loading a session programmatically, skip changes
    if is_session_being_loaded:
        return

    # If session has messages already, do not allow changes
    if len(chat_manager.db.get_messages(chat_manager.session_id)) > 0:
        notification_queue.put_nowait((
            "Cannot change setting after session has started. Use a new session instead.", 'warning'
        ))
        # Revert to the actual current setting
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
    next_button.enabled = not chat_manager.automatic_running
    next_button.update()


@ui.refreshable
def show_chat_display():
    chat_display.clear()
    msgs = chat_manager.db.get_messages(chat_manager.session_id)
    msgs_found = len(msgs) > 0
    if msgs_found:
        current_location_label.text = ""
        current_location_label.update()
    with chat_display:
        for entry in msgs:
            name = entry["sender"]
            message = entry["message"]
            timestamp = entry["created_at"]
            dt = datetime.fromisoformat(timestamp)
            human_timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
            formatted_message = f"**{name}** [{human_timestamp}]:\n\n{message}"
            ui.markdown(formatted_message)


async def automatic_conversation():
    if chat_manager.automatic_running:
        next_char = chat_manager.next_speaker()
        if next_char:
            await chat_manager.generate_character_message(next_char)
            chat_manager.advance_turn()
            update_next_speaker_label()
            show_character_details.refresh()
            show_chat_display.refresh()


async def next_character_response():
    if chat_manager.automatic_running:
        return
    next_char = chat_manager.next_speaker()
    if next_char:
        await chat_manager.generate_character_message(next_char)
        chat_manager.advance_turn()
        update_next_speaker_label()
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
    update_next_speaker_label()
    character_dropdown.value = None
    character_dropdown.update()


async def remove_character_async(name: str):
    chat_manager.remove_character(name)
    chat_manager.db.remove_character_from_session(chat_manager.session_id, name)
    refresh_added_characters()
    show_chat_display.refresh()
    show_character_details.refresh()
    update_next_speaker_label()


#
# NEW BUTTON & FUNCTION TO UPDATE LOCATION/APPEARANCE FROM SCRATCH FOR ALL CHARACTERS
#
async def update_all_characters_info():
    await chat_manager.update_all_characters_location_and_appearance_from_scratch()
    show_chat_display.refresh()
    show_character_details.refresh()


#
# NEW: Retrieve and populate local models in a dropdown, user picks one
#
async def refresh_local_models():
    if not llm_client:
        return
    models = llm_client.list_local_models()
    local_model_dropdown.options = models
    local_model_dropdown.update()
    # If we have at least one model, select the first by default
    if models:
        local_model_dropdown.value = models[0]
        local_model_dropdown.update()
        # This ensures the on_change callback triggers with the default
        on_local_model_select({"value": models[0]})


def on_local_model_select(event):
    """
    When user picks a local model, set it on the global LLM client.
    If user picks an empty value, revert to config model.
    """
    # Extract chosen value depending on the type of 'event'
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


def toggle_npc_manager(value: bool):
    if value:
        chat_manager.enable_npc_manager()
    else:
        chat_manager.disable_npc_manager()


# -----------------------------------------------------------------------------
# NEW/UPDATED: consume_llm_status timer to read from llm_status_queue
# -----------------------------------------------------------------------------
async def consume_llm_status():
    # Display the most recent status in llm_status_label if any
    while not llm_status_queue.empty():
        msg = await llm_status_queue.get()
        if llm_status_label:
            llm_status_label.text = msg
            llm_status_label.visible = True
            llm_status_label.update()
# -----------------------------------------------------------------------------


def main_page():
    global character_dropdown, added_characters_container
    global next_speaker_label, next_button, settings_dropdown, setting_description_label
    global session_dropdown, chat_display, current_location_label, llm_status_label
    global character_details_display

    global ALL_CHARACTERS, ALL_SETTINGS
    global local_model_dropdown

    ALL_CHARACTERS = get_available_characters("src/multipersona_chat_app/characters")
    ALL_SETTINGS = load_settings()

    with ui.grid(columns=2).style('grid-template-columns: 1fr 2fr; height: 100vh;'):
        with ui.card().style('height: 100vh; overflow-y: auto;'):
            ui.label('Multipersona Chat Application').classes('text-2xl font-bold mb-4')

            # NEW: Local model dropdown, a prominent location
            with ui.row().classes('w-full items-center mb-4'):
                ui.label("Local Model:").classes('w-1/4')
                local_model_dropdown = ui.select(
                    options=[],
                    on_change=on_local_model_select,
                    label="Available local models"
                ).classes('flex-grow')
                ui.button("Refresh Models", on_click=lambda: asyncio.create_task(refresh_local_models())).classes('ml-2')

            # Session handling
            with ui.row().classes('w-full items-center mb-4'):
                ui.label("Session:").classes('w-1/4')
                session_dropdown = ui.select(
                    options=[s['name'] for s in chat_manager.db.get_all_sessions()],
                    label="Choose a session",
                ).classes('flex-grow')
                ui.button("New Session", on_click=create_new_session).classes('ml-2')
                ui.button("Delete Session", on_click=delete_session).classes('ml-2 bg-red-500 text-white')

            # Setting selection
            with ui.row().classes('w-full items-center mb-4'):
                ui.label("Select Setting:").classes('w-1/4')
                settings_dropdown = ui.select(
                    options=[s['name'] for s in ALL_SETTINGS],
                    on_change=select_setting,
                    label="Choose a setting"
                ).classes('flex-grow')

            # Setting info
            with ui.row().classes('w-full items-center mb-2'):
                ui.label("Setting Description:").classes('w-1/4')
                setting_description_label = ui.label("(Not set)").classes('flex-grow text-gray-700')

            with ui.row().classes('w-full items-center mb-2'):
                current_location_label = ui.label("").classes('flex-grow text-gray-700')

            # Character details
            character_details_display = ui.column().classes('mb-4')
            show_character_details()

            # Character selection
            with ui.row().classes('w-full items-center mb-4'):
                ui.label("Select Character:").classes('w-1/4')
                character_dropdown = ui.select(
                    options=list(ALL_CHARACTERS.keys()),
                    on_change=lambda e: asyncio.create_task(add_character_from_dropdown(e)),
                    label="Choose a character"
                ).classes('flex-grow')

            with ui.column().classes('w-full mb-4'):
                ui.label("Added Characters:").classes('font-semibold mb-2')
                added_characters_container = ui.row().classes('flex-wrap gap-2')
                refresh_added_characters()

            with ui.row().classes('w-full items-center mb-4'):
                auto_switch = ui.switch('Automatic Chat', value=False, on_change=toggle_automatic_chat).classes('mr-2')
                npc_switch = ui.switch('NPC Manager Active', value=True, on_change=lambda e: toggle_npc_manager(e.value)).classes('mr-2')

            global next_speaker_label
            next_speaker_label = ui.label("Next speaker:")
            update_next_speaker_label()

            global next_button
            next_button = ui.button("Next", on_click=lambda: asyncio.create_task(next_character_response()))
            next_button.props('outline')
            next_button.enabled = not chat_manager.automatic_running
            next_button.update()

            ui.button(
                "Update All Characters Info (Location & Appearance)",
                on_click=lambda: asyncio.create_task(update_all_characters_info())
            ).classes('mt-4 bg-green-500 text-white')

            # NEW/UPDATED: Real-time LLM status label
            global llm_status_label
            llm_status_label = ui.label("").classes('text-orange-600')
            llm_status_label.visible = False

        with ui.card().style('height: 100vh; display: flex; flex-direction: column;'):
            global chat_display
            chat_display = ui.column().style('flex-grow: 1; overflow-y: auto;')
            show_chat_display()

    session_dropdown.on('change', on_session_select)
    session_dropdown.value = chat_manager.get_session_name()
    session_dropdown.update()

    global auto_timer
    auto_timer = ui.timer(interval=2.0, callback=lambda: asyncio.create_task(automatic_conversation()), active=False)

    # Timer to show standard notifications
    ui.timer(1.0, consume_notifications, active=True)

    # NEW/UPDATED: Timer to consume LLM status messages
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
