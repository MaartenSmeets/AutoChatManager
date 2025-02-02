# File: /home/maarten/AutoChatManager/src/multipersona_chat_app/db/db_manager.py

import sqlite3
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from models.interaction import AppearanceSegments
from models.character_metadata import CharacterMetadata

logger = logging.getLogger(__name__)

def merge_location_update(old_location: str, new_location: str) -> str:
    if not new_location.strip():
        return old_location
    return new_location

def merge_appearance_subfield(old_val: str, new_val: str) -> str:
    if not new_val.strip():
        return old_val
    return new_val

class DBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize_database()

    def _ensure_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize_database(self):
        conn = self._ensure_connection()
        c = conn.cursor()

        # sessions table (updated to include model selection and toggle settings)
        c.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                current_setting TEXT,
                current_location TEXT,
                setting_description TEXT,
                model_selection TEXT DEFAULT '',
                npc_manager_active INTEGER DEFAULT 0,
                auto_chat INTEGER DEFAULT 0,
                show_private_info INTEGER DEFAULT 1
            )
        ''')

        # messages table
        c.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                message TEXT NOT NULL,
                visible INTEGER DEFAULT 1,
                message_type TEXT DEFAULT 'user',
                emotion TEXT,
                thoughts TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        ''')

        # message_visibility table
        c.execute('''
            CREATE TABLE IF NOT EXISTS message_visibility (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                character_name TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                visible INTEGER DEFAULT 1,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id),
                FOREIGN KEY(message_id) REFERENCES messages(id)
            )
        ''')

        # summaries table
        c.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                character_name TEXT NOT NULL,
                summary TEXT NOT NULL,
                covered_up_to_message_id INTEGER,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        ''')

        # location_history table
        c.execute('''
            CREATE TABLE IF NOT EXISTS location_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                location TEXT NOT NULL,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                triggered_by_message_id INTEGER,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id),
                FOREIGN KEY(triggered_by_message_id) REFERENCES messages(id)
            )
        ''')

        # session_characters table (holds name, location, appearance subfields)
        c.execute('''
            CREATE TABLE IF NOT EXISTS session_characters (
                session_id TEXT NOT NULL,
                character_name TEXT NOT NULL,
                current_location TEXT,
                current_appearance TEXT,
                hair TEXT,
                clothing TEXT,
                accessories_and_held_items TEXT,
                posture_and_body_language TEXT,
                facial_expression TEXT,
                other_relevant_details TEXT,
                PRIMARY KEY (session_id, character_name),
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        ''')

        # appearance_history table
        c.execute('''
            CREATE TABLE IF NOT EXISTS appearance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                character_name TEXT NOT NULL,
                hair TEXT,
                clothing TEXT,
                accessories_and_held_items TEXT,
                posture_and_body_language TEXT,
                facial_expression TEXT,
                other_relevant_details TEXT,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                triggered_by_message_id INTEGER,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id),
                FOREIGN KEY(triggered_by_message_id) REFERENCES messages(id)
            )
        ''')

        # character_prompts table
        c.execute('''
            CREATE TABLE IF NOT EXISTS character_prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                character_name TEXT NOT NULL,
                character_system_prompt TEXT NOT NULL,
                dynamic_prompt_template TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id),
                UNIQUE(session_id, character_name)
            )
        ''')

        # character_plans table
        c.execute('''
            CREATE TABLE IF NOT EXISTS character_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                character_name TEXT NOT NULL,
                goal TEXT,
                steps TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                why_new_plan_goal TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id),
                UNIQUE(session_id, character_name)
            )
        ''')

        # character_plans_history table
        c.execute('''
            CREATE TABLE IF NOT EXISTS character_plans_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                character_name TEXT NOT NULL,
                goal TEXT,
                steps TEXT,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                triggered_by_message_id INTEGER,
                change_summary TEXT,
                why_new_plan_goal TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id),
                FOREIGN KEY(triggered_by_message_id) REFERENCES messages(id)
            )
        ''')

        # NEW: character_metadata table for extra data like is_npc, role, etc.
        c.execute('''
            CREATE TABLE IF NOT EXISTS character_metadata (
                session_id TEXT NOT NULL,
                character_name TEXT NOT NULL,
                is_npc INTEGER DEFAULT 0,
                role TEXT DEFAULT '',
                PRIMARY KEY (session_id, character_name),
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        ''')

        conn.commit()
        conn.close()
        logger.info("Database schema initialized/updated (unified for PCs/NPCs).")

    # ----------------------------------------------------------------
    # Session queries
    # ----------------------------------------------------------------

    def create_session(self, session_id: str, name: str):
        conn = self._ensure_connection()
        c = conn.cursor()
        try:
            c.execute('INSERT INTO sessions (session_id, name) VALUES (?, ?)', (session_id, name))
            conn.commit()
            logger.info(f"Session '{name}' with ID '{session_id}' created.")
        except sqlite3.IntegrityError:
            logger.error(f"Session with ID '{session_id}' already exists.")
        finally:
            conn.close()

    def delete_session(self, session_id: str):
        conn = self._ensure_connection()
        c = conn.cursor()

        # Remove everything associated with this session
        c.execute('DELETE FROM summaries WHERE session_id = ?', (session_id,))
        c.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
        c.execute('DELETE FROM message_visibility WHERE session_id = ?', (session_id,))
        c.execute('DELETE FROM location_history WHERE session_id = ?', (session_id,))
        c.execute('DELETE FROM session_characters WHERE session_id = ?', (session_id,))
        c.execute('DELETE FROM character_prompts WHERE session_id = ?', (session_id,))
        c.execute('DELETE FROM appearance_history WHERE session_id = ?', (session_id,))
        c.execute('DELETE FROM character_plans WHERE session_id = ?', (session_id,))
        c.execute('DELETE FROM character_plans_history WHERE session_id = ?', (session_id,))
        c.execute('DELETE FROM character_metadata WHERE session_id = ?', (session_id,))
        c.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))

        conn.commit()
        conn.close()
        logger.info(f"Session with ID '{session_id}' and all related data deleted.")

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('SELECT session_id, name FROM sessions')
        rows = c.fetchall()
        sessions = [{'session_id': row[0], 'name': row[1]} for row in rows]
        conn.close()
        return sessions

    def get_current_setting(self, session_id: str) -> Optional[str]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('SELECT current_setting FROM sessions WHERE session_id = ?', (session_id,))
        row = c.fetchone()
        conn.close()
        if row and row[0]:
            return row[0]
        return None

    def update_current_setting(self, session_id: str, setting_name: str):
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('UPDATE sessions SET current_setting = ? WHERE session_id = ?', (setting_name, session_id))
        conn.commit()
        conn.close()

    def get_current_setting_description(self, session_id: str) -> Optional[str]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('SELECT setting_description FROM sessions WHERE session_id = ?', (session_id,))
        row = c.fetchone()
        conn.close()
        if row and row[0]:
            return row[0]
        return None

    def update_current_setting_description(self, session_id: str, description: str):
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('UPDATE sessions SET setting_description = ? WHERE session_id = ?', (description, session_id))
        conn.commit()
        conn.close()

    def get_current_location(self, session_id: str) -> Optional[str]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('SELECT current_location FROM sessions WHERE session_id = ?', (session_id,))
        row = c.fetchone()
        conn.close()
        if row and row[0]:
            return row[0]
        return None

    def update_current_location(self, session_id: str, location: str, triggered_by_message_id: Optional[int] = None):
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('UPDATE sessions SET current_location = ? WHERE session_id = ?', (location, session_id))
        if triggered_by_message_id is not None:
            c.execute('''
                INSERT INTO location_history (session_id, location, triggered_by_message_id)
                VALUES (?, ?, ?)
            ''', (session_id, location, triggered_by_message_id))
        conn.commit()
        conn.close()

    def get_location_history(self, session_id: str) -> List[Dict[str, Any]]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            SELECT location, changed_at, triggered_by_message_id
            FROM location_history
            WHERE session_id = ?
            ORDER BY changed_at ASC
        ''', (session_id,))
        rows = c.fetchall()
        conn.close()
        history = []
        for row in rows:
            history.append({
                'location': row[0],
                'changed_at': row[1],
                'triggered_by_message_id': row[2]
            })
        return history

    # ----------------------------------------------------------------
    # Character management (session_characters + metadata)
    # ----------------------------------------------------------------

    def add_character_to_session(self, session_id: str, character_name: str,
                                 initial_location: str = "", initial_appearance: str = ""):
        """Creates a row in session_characters if not present."""
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            INSERT OR IGNORE INTO session_characters 
            (session_id, character_name, current_location, current_appearance)
            VALUES (?, ?, ?, ?)
        ''', (session_id, character_name, initial_location, initial_appearance))
        conn.commit()
        conn.close()

    def remove_character_from_session(self, session_id: str, character_name: str):
        """Removes the character from session_characters and associated metadata."""
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('DELETE FROM session_characters WHERE session_id=? AND character_name=?',
                  (session_id, character_name))
        c.execute('DELETE FROM character_metadata WHERE session_id=? AND character_name=?',
                  (session_id, character_name))
        conn.commit()
        conn.close()

    def get_session_characters(self, session_id: str) -> List[str]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('SELECT character_name FROM session_characters WHERE session_id=?', (session_id,))
        rows = c.fetchall()
        conn.close()
        return [r[0] for r in rows]

    def get_character_location(self, session_id: str, character_name: str) -> Optional[str]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            SELECT current_location FROM session_characters
            WHERE session_id=? AND character_name=?
        ''', (session_id, character_name))
        row = c.fetchone()
        conn.close()
        if row and row[0]:
            return row[0]
        return ""

    def get_character_appearance(self, session_id: str, character_name: str) -> str:
        """
        Return a user-facing text that combines the subfields if present.
        """
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            SELECT current_appearance, hair, clothing, accessories_and_held_items,
                   posture_and_body_language, facial_expression, other_relevant_details
            FROM session_characters
            WHERE session_id=? AND character_name=?
        ''', (session_id, character_name))
        row = c.fetchone()
        conn.close()
        if not row:
            return ""
        legacy_app = row[0] or ""
        hair = row[1] or ""
        cloth = row[2] or ""
        acc = row[3] or ""
        posture = row[4] or ""
        face = row[5] or ""
        other = row[6] or ""
        combined = []
        if hair.strip():
            combined.append(f"Hair: {hair}")
        if cloth.strip():
            combined.append(f"Clothing: {cloth}")
        if acc.strip():
            combined.append(f"Accessories: {acc}")
        if posture.strip():
            combined.append(f"Posture: {posture}")
        if face.strip():
            combined.append(f"Facial Expression: {face}")
        if other.strip():
            combined.append(f"Other: {other}")
        if not combined:
            return legacy_app
        return " | ".join(combined)

    def get_current_appearance_segments(self, session_id: str, character_name: str) -> Dict[str, str]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            SELECT hair, clothing, accessories_and_held_items, posture_and_body_language,
                   facial_expression, other_relevant_details
            FROM session_characters
            WHERE session_id=? AND character_name=?
        ''', (session_id, character_name))
        row = c.fetchone()
        conn.close()
        if not row:
            return {}
        return {
            'hair': row[0] or "",
            'clothing': row[1] or "",
            'accessories_and_held_items': row[2] or "",
            'posture_and_body_language': row[3] or "",
            'facial_expression': row[4] or "",
            'other_relevant_details': row[5] or ""
        }

    def get_all_character_locations(self, session_id: str) -> Dict[str, str]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            SELECT character_name, current_location
            FROM session_characters
            WHERE session_id=?
        ''', (session_id,))
        rows = c.fetchall()
        conn.close()
        return {row[0]: (row[1] if row[1] else "") for row in rows}

    def get_all_character_appearances(self, session_id: str) -> Dict[str, str]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            SELECT character_name, current_appearance, hair, clothing, accessories_and_held_items,
                   posture_and_body_language, facial_expression, other_relevant_details
            FROM session_characters
            WHERE session_id=?
        ''', (session_id,))
        rows = c.fetchall()
        conn.close()

        results = {}
        for row in rows:
            c_name = row[0]
            legacy_app = row[1] or ""
            hair = row[2] or ""
            cloth = row[3] or ""
            acc = row[4] or ""
            posture = row[5] or ""
            face = row[6] or ""
            other = row[7] or ""
            combined = []
            if hair.strip():
                combined.append(f"Hair: {hair}")
            if cloth.strip():
                combined.append(f"Clothing: {cloth}")
            if acc.strip():
                combined.append(f"Accessories: {acc}")
            if posture.strip():
                combined.append(f"Posture: {posture}")
            if face.strip():
                combined.append(f"Facial Expression: {face}")
            if other.strip():
                combined.append(f"Other: {other}")
            if not combined:
                results[c_name] = legacy_app
            else:
                results[c_name] = " | ".join(combined)
        return results

    def update_character_location(self, session_id: str, character_name: str,
                                  new_location: str, triggered_by_message_id: Optional[int] = None) -> bool:
        old_location = self.get_character_location(session_id, character_name)
        updated = merge_location_update(old_location, new_location)
        if updated == old_location:
            return False

        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            UPDATE session_characters
            SET current_location=?
            WHERE session_id=? AND character_name=?
        ''', (updated, session_id, character_name))
        conn.commit()
        conn.close()

        if triggered_by_message_id:
            conn2 = self._ensure_connection()
            c2 = conn2.cursor()
            c2.execute('''
                INSERT INTO location_history (session_id, location, triggered_by_message_id)
                VALUES (?, ?, ?)
            ''', (session_id, updated, triggered_by_message_id))
            conn2.commit()
            conn2.close()

        return True

    def update_character_appearance(self, session_id: str, character_name: str,
                                    new_appearance: AppearanceSegments,
                                    triggered_by_message_id: Optional[int] = None) -> bool:
        old_segments = self.get_current_appearance_segments(session_id, character_name)
        if not old_segments:
            return False

        merged_hair = merge_appearance_subfield(old_segments['hair'], new_appearance.hair or "")
        merged_cloth = merge_appearance_subfield(old_segments['clothing'], new_appearance.clothing or "")
        merged_acc = merge_appearance_subfield(old_segments['accessories_and_held_items'], new_appearance.accessories_and_held_items or "")
        merged_posture = merge_appearance_subfield(old_segments['posture_and_body_language'], new_appearance.posture_and_body_language or "")
        merged_face = merge_appearance_subfield(old_segments['facial_expression'], new_appearance.facial_expression or "")
        merged_other = merge_appearance_subfield(old_segments['other_relevant_details'], new_appearance.other_relevant_details or "")

        nothing_changed = (
            merged_hair == old_segments['hair'] and
            merged_cloth == old_segments['clothing'] and
            merged_acc == old_segments['accessories_and_held_items'] and
            merged_posture == old_segments['posture_and_body_language'] and
            merged_face == old_segments['facial_expression'] and
            merged_other == old_segments['other_relevant_details']
        )
        if nothing_changed:
            return False

        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            UPDATE session_characters
            SET hair=?, clothing=?, accessories_and_held_items=?,
                posture_and_body_language=?, facial_expression=?, other_relevant_details=?
            WHERE session_id=? AND character_name=?
        ''', (
            merged_hair, merged_cloth, merged_acc,
            merged_posture, merged_face, merged_other,
            session_id, character_name
        ))
        conn.commit()
        conn.close()

        # appearance_history
        conn2 = self._ensure_connection()
        c2 = conn2.cursor()
        c2.execute('''
            INSERT INTO appearance_history (
                session_id, character_name,
                hair, clothing, accessories_and_held_items,
                posture_and_body_language, facial_expression, other_relevant_details,
                triggered_by_message_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, character_name,
            merged_hair, merged_cloth, merged_acc, merged_posture, merged_face, merged_other, triggered_by_message_id )) 
        conn2.commit()
        conn2.close()
        return True
    
    def get_characters_appearance_except_one(self, session_id: str, exclude_character: str) -> Dict[str, str]:
        all_aps = self.get_all_character_appearances(session_id)
        return {name: ap for name, ap in all_aps.items() if name != exclude_character}

    def get_character_names(self, session_id: str) -> List[str]:
        """Retrieve a list of all characters in the session."""
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('SELECT character_name FROM session_characters WHERE session_id=?', (session_id,))
        rows = c.fetchall()
        conn.close()
        return [row[0] for row in rows]

    # ----------------------------------------------------------------
    # Character metadata queries (new table)
    # ----------------------------------------------------------------

    def save_character_metadata(self, session_id: str, character_name: str, meta: CharacterMetadata):
        """
        Insert or update the character's metadata record: is_npc, role, etc.
        """
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            INSERT INTO character_metadata (session_id, character_name, is_npc, role)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id, character_name) 
            DO UPDATE SET is_npc=excluded.is_npc, role=excluded.role
        ''', (
            session_id,
            character_name,
            1 if meta.is_npc else 0,
            meta.role
        ))
        conn.commit()
        conn.close()

    def get_character_metadata(self, session_id: str, character_name: str) -> Optional[CharacterMetadata]:
        """
        Retrieve the character’s metadata. Returns None if no record found.
        """
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            SELECT is_npc, role
            FROM character_metadata
            WHERE session_id=? AND character_name=?
        ''', (session_id, character_name))
        row = c.fetchone()
        conn.close()
        if not row:
            return None
        is_npc_val = bool(row[0])
        role_val = row[1] or ""
        return CharacterMetadata(is_npc=is_npc_val, role=role_val)

    # ----------------------------------------------------------------
    # Messages
    # ----------------------------------------------------------------

    def save_message(self,
                    session_id: str,
                    sender: str,
                    message: str,
                    visible: int,
                    message_type: str,
                    emotion: Optional[str],
                    thoughts: Optional[str]) -> int:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            INSERT INTO messages (
                session_id, sender, message, visible, message_type, emotion, thoughts
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            sender,
            message,
            visible,
            message_type,
            emotion,
            thoughts
        ))
        message_id = c.lastrowid
        conn.commit()
        conn.close()
        return message_id

    def add_message_visibility_for_session_characters(self, session_id: str, message_id: int):
        """
        For every character in the session, record a row in message_visibility marking
        this new message as visible=1 for them.
        """
        chars = self.get_session_characters(session_id)
        conn = self._ensure_connection()
        c = conn.cursor()
        for char in chars:
            c.execute('''
                INSERT INTO message_visibility (session_id, character_name, message_id, visible)
                VALUES (?, ?, ?, 1)
            ''', (session_id, char, message_id))
        conn.commit()
        conn.close()

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            SELECT id, sender, message, visible, message_type, emotion, thoughts, created_at
            FROM messages
            WHERE session_id=?
            ORDER BY id ASC
        ''', (session_id,))
        rows = c.fetchall()
        conn.close()

        messages = []
        for row in rows:
            messages.append({
                'id': row[0],
                'sender': row[1],
                'message': row[2],
                'visible': bool(row[3]),
                'message_type': row[4],
                'emotion': row[5],
                'thoughts': row[6],
                'created_at': row[7]
            })
        return messages

    def get_visible_messages_for_character(self, session_id: str, character_name: str) -> List[Dict[str, Any]]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            SELECT m.id, m.sender, m.message, mv.visible, m.message_type,
                m.emotion, m.thoughts, m.created_at
            FROM messages m
            JOIN message_visibility mv ON m.id = mv.message_id
            WHERE mv.session_id=?
            AND mv.character_name=?
            AND mv.visible=1
            ORDER BY m.id ASC
        ''', (session_id, character_name))
        rows = c.fetchall()
        conn.close()

        messages = []
        for row in rows:
            messages.append({
                'id': row[0],
                'sender': row[1],
                'message': row[2],
                'visible': bool(row[3]),
                'message_type': row[4],
                'emotion': row[5],
                'thoughts': row[6],
                'created_at': row[7],
            })
        return messages

    def hide_messages_for_character(self, session_id: str, character_name: str, message_ids: List[int]):
        if not message_ids:
            return
        conn = self._ensure_connection()
        c = conn.cursor()
        placeholders = ",".join("?" * len(message_ids))
        params = [session_id, character_name] + message_ids
        c.execute(f'''
            UPDATE message_visibility
            SET visible=0
            WHERE session_id=? AND character_name=? AND message_id IN ({placeholders})
        ''', params)
        conn.commit()
        conn.close()

    # ----------------------------------------------------------------
    # Summaries
    # ----------------------------------------------------------------

    def save_new_summary(self, session_id: str, character_name: str, summary: str, covered_up_to_message_id: int):
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            INSERT INTO summaries (session_id, character_name, summary, covered_up_to_message_id)
            VALUES (?, ?, ?, ?)
        ''', (session_id, character_name, summary, covered_up_to_message_id))
        conn.commit()
        conn.close()

    def get_all_summaries(self, session_id: str, character_name: Optional[str]) -> List[str]:
        conn = self._ensure_connection()
        c = conn.cursor()
        if character_name:
            c.execute('''
                SELECT summary FROM summaries
                WHERE session_id=? AND character_name=?
                ORDER BY id ASC
            ''', (session_id, character_name))
        else:
            c.execute('''
                SELECT summary FROM summaries
                WHERE session_id=?
                ORDER BY id ASC
            ''', (session_id,))
        rows = c.fetchall()
        conn.close()
        return [row[0] for row in rows]

    def get_all_summaries_records(self, session_id: str, character_name: str) -> List[Dict[str, Any]]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            SELECT id, summary, covered_up_to_message_id
            FROM summaries
            WHERE session_id=? AND character_name=?
            ORDER BY id ASC
        ''', (session_id, character_name))
        rows = c.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "summary": row[1],
                "covered_up_to_message_id": row[2] if row[2] else 0
            })
        return results

    def delete_summaries_by_ids(self, session_id: str, character_name: str, summary_ids: List[int]):
        if not summary_ids:
            return
        conn = self._ensure_connection()
        c = conn.cursor()
        placeholders = ",".join("?" * len(summary_ids))
        params = [session_id, character_name] + summary_ids
        c.execute(f'''
            DELETE FROM summaries
            WHERE session_id=? AND character_name=? AND id IN ({placeholders})
        ''', params)
        conn.commit()
        conn.close()

    def get_latest_covered_message_id(self, session_id: str, character_name: str) -> int:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            SELECT covered_up_to_message_id
            FROM summaries
            WHERE session_id=? AND character_name=?
            ORDER BY covered_up_to_message_id DESC
            LIMIT 1
        ''', (session_id, character_name))
        row = c.fetchone()
        conn.close()
        if row and row[0]:
            return row[0]
        return 0

    # ----------------------------------------------------------------
    # Character prompts
    # ----------------------------------------------------------------

    def get_character_prompts(self, session_id: str, character_name: str) -> Optional[Dict[str, str]]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            SELECT character_system_prompt, dynamic_prompt_template
            FROM character_prompts
            WHERE session_id=? AND character_name=?
        ''', (session_id, character_name))
        row = c.fetchone()
        conn.close()
        if row:
            return {
                'character_system_prompt': row[0],
                'dynamic_prompt_template': row[1]
            }
        return None

    def save_character_prompts(self, session_id: str, character_name: str,
                            character_system_prompt: str,
                            dynamic_prompt_template: str):
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            INSERT INTO character_prompts (session_id, character_name, character_system_prompt, dynamic_prompt_template)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id, character_name)
            DO UPDATE SET character_system_prompt=excluded.character_system_prompt,
                        dynamic_prompt_template=excluded.dynamic_prompt_template
        ''', (session_id, character_name, character_system_prompt, dynamic_prompt_template))
        conn.commit()
        conn.close()

    # ----------------------------------------------------------------
    # Character plans
    # ----------------------------------------------------------------

    def get_character_plan(self, session_id: str, character_name: str) -> Optional[Dict[str, Any]]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            SELECT goal, steps, updated_at, why_new_plan_goal
            FROM character_plans
            WHERE session_id=? AND character_name=?
        ''', (session_id, character_name))
        row = c.fetchone()
        conn.close()
        if row:
            goal_str = row[0] or ""
            steps_str = row[1] or ""
            updated_at = row[2]
            why_new = row[3] or ""
            steps_list = []
            if steps_str:
                try:
                    steps_list = json.loads(steps_str)
                except:
                    logger.warning(f"Failed to parse steps JSON for {character_name}.")
            return {
                'goal': goal_str,
                'steps': steps_list,
                'updated_at': updated_at,
                'why_new_plan_goal': why_new
            }
        return None

    def save_character_plan(self, session_id: str, character_name: str,
                            goal: str, steps: List[str], why_new_plan_goal: str):
        steps_str = json.dumps(steps)
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            INSERT INTO character_plans (session_id, character_name, goal, steps, why_new_plan_goal)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id, character_name)
            DO UPDATE SET goal=excluded.goal,
                        steps=excluded.steps,
                        why_new_plan_goal=excluded.why_new_plan_goal,
                        updated_at=CURRENT_TIMESTAMP
        ''', (session_id, character_name, goal, steps_str, why_new_plan_goal))
        conn.commit()
        conn.close()

    def save_character_plan_with_history(self,
                                        session_id: str,
                                        character_name: str,
                                        goal: str,
                                        steps: List[str],
                                        why_new_plan_goal: str,
                                        triggered_by_message_id: Optional[int],
                                        change_summary: str):
        steps_str = json.dumps(steps)
        conn = self._ensure_connection()
        c = conn.cursor()
        # main plan row
        c.execute('''
            INSERT INTO character_plans (session_id, character_name, goal, steps, why_new_plan_goal)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id, character_name)
            DO UPDATE SET goal=excluded.goal,
                        steps=excluded.steps,
                        why_new_plan_goal=excluded.why_new_plan_goal,
                        updated_at=CURRENT_TIMESTAMP
        ''', (session_id, character_name, goal, steps_str, why_new_plan_goal))

        # plan history
        c.execute('''
            INSERT INTO character_plans_history (
                session_id, character_name, goal, steps, triggered_by_message_id,
                change_summary, why_new_plan_goal
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            character_name,
            goal,
            steps_str,
            triggered_by_message_id,
            change_summary,
            why_new_plan_goal
        ))
        conn.commit()
        conn.close()

    def get_plan_changes_for_range(self, session_id: str, character_name: str,
                                after_message_id: int, up_to_message_id: int) -> List[Dict[str, Any]]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            SELECT triggered_by_message_id, change_summary, why_new_plan_goal
            FROM character_plans_history
            WHERE session_id=?
            AND character_name=?
            AND triggered_by_message_id IS NOT NULL
            AND triggered_by_message_id>?
            AND triggered_by_message_id<=?
            ORDER BY id ASC
        ''', (session_id, character_name, after_message_id, up_to_message_id))
        rows = c.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                "triggered_by_message_id": row[0],
                "change_summary": row[1],
                "why_new_plan_goal": row[2] or ""
            })
        return results

    def get_latest_nonempty_plan_in_history(self, session_id: str, character_name: str) -> Optional[Dict[str, Any]]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            SELECT goal, steps, changed_at, why_new_plan_goal
            FROM character_plans_history
            WHERE session_id=? AND character_name=?
            ORDER BY id DESC
        ''', (session_id, character_name))
        rows = c.fetchall()
        conn.close()
        for row in rows:
            g = row[0] or ""
            s_str = row[1] or ""
            try:
                s_list = json.loads(s_str)
            except:
                s_list = []
            if g.strip() or any(step.strip() for step in s_list):
                return {
                    'goal': g,
                    'steps': s_list,
                    'changed_at': row[2],
                    'why_new_plan_goal': row[3] or ""
                }
        return None

    def get_session_settings(self, session_id: str) -> Dict[str, Any]:
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            SELECT model_selection, npc_manager_active, auto_chat, show_private_info
            FROM sessions
            WHERE session_id = ?
        ''', (session_id,))
        row = c.fetchone()
        conn.close()
        if row:
            return {
                'model_selection': row[0] or '',
                'npc_manager_active': bool(row[1]),
                'auto_chat': bool(row[2]),
                'show_private_info': bool(row[3])
            }
        return {'model_selection': '', 'npc_manager_active': False, 'auto_chat': False, 'show_private_info': True}

    def update_session_settings(self, session_id: str, settings: Dict[str, Any]):
        conn = self._ensure_connection()
        c = conn.cursor()
        c.execute('''
            UPDATE sessions
            SET model_selection = ?,
                npc_manager_active = ?,
                auto_chat = ?,
                show_private_info = ?
            WHERE session_id = ?
        ''', (
            settings.get('model_selection', ''),
            1 if settings.get('npc_manager_active', False) else 0,
            1 if settings.get('auto_chat', False) else 0,
            1 if settings.get('show_private_info', True) else 0,
            session_id
        ))
        conn.commit()
        conn.close()