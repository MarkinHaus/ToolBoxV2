# --- START OF FILE POA.py ---
import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, List, Optional, Dict, Literal
import uuid
import json

from pydantic import BaseModel, Field

from toolboxv2 import Code, Result, get_app, App, RequestData
from toolboxv2.utils.extras.base_widget import get_user_from_request

# Assuming isaa_module is accessible. If it's in the same directory or installed:
# from .isaa_module import Tools as IsaTools # If isaa is in a file named isaa_module.py
# For this combined example, let's assume ISAA is initialized elsewhere and we get it via app.get_mod("isaa")
# For a self-contained example, you might initialize a local ISAA Tools instance.

Name = "POA"
version = "1.0.0"
export = get_app(f"{Name}.Export").tb


# --- ISAA Integration Placeholder ---
# In a real setup, ISAA would be a separate module initialized by the app.
# For demonstration, we'll create a mock ISAA or assume it's available via get_mod.
# We need ISAA for parsing text to items and for suggestions.

# --- Pydantic Models ---

class ItemType(str, Enum):
    TASK = "task"
    NOTE = "note"


class Frequency(str, Enum):
    ONE_TIME = "one_time"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ANNUALLY = "annually"


class ActionStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"  # Added status


class ActionItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    item_type: ItemType = ItemType.TASK
    title: str
    description: Optional[str] = None  # For notes, this is the main content
    parent_id: Optional[str] = None

    # Task-specific fields (optional for notes)
    frequency: Optional[Frequency] = Frequency.ONE_TIME
    priority: int = Field(default=3, ge=1, le=5)  # 1 highest, 5 lowest
    fixed_time: Optional[datetime] = None  # Due date/time

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    status: ActionStatus = ActionStatus.NOT_STARTED
    last_completed: Optional[datetime] = None
    next_due: Optional[datetime] = None

    # AI related
    created_by_ai: bool = False

    def model_dump_json_safe(self, *args, **kwargs):
        data = self.model_dump(*args, **kwargs)
        for field_name, value in data.items():
            if isinstance(value, datetime):
                data[field_name] = value.isoformat()
            elif isinstance(value, Enum):
                data[field_name] = value.value
        return data

    @classmethod
    def model_validate_json_safe(cls, json_data: Dict[str, Any]):
        for field_name, value in json_data.items():
            if field_name in ['fixed_time', 'created_at', 'updated_at', 'last_completed', 'next_due'] and value:
                if isinstance(value, str):
                    try:
                        json_data[field_name] = datetime.fromisoformat(value)
                    except ValueError:
                        # Handle cases like "Z" at the end for UTC, which fromisoformat might not like directly without stripping
                        json_data[field_name] = datetime.fromisoformat(value.replace('Z', '+00:00'))


            elif field_name == 'item_type' and isinstance(value, str):
                json_data[field_name] = ItemType(value)
            elif field_name == 'frequency' and isinstance(value, str):
                json_data[field_name] = Frequency(value)
            elif field_name == 'status' and isinstance(value, str):
                json_data[field_name] = ActionStatus(value)
        return cls.model_validate(json_data)


class HistoryEntry(BaseModel):
    item_id: str
    item_title: str
    item_type: ItemType
    timestamp: datetime = Field(default_factory=datetime.now)
    status_changed_to: ActionStatus
    parent_id: Optional[str] = None
    notes: Optional[str] = None  # e.g., "AI generated"

    def model_dump_json_safe(self, *args, **kwargs):
        data = self.model_dump(*args, **kwargs)
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        if isinstance(data.get("item_type"), Enum):
            data["item_type"] = data["item_type"].value
        if isinstance(data.get("status_changed_to"), Enum):
            data["status_changed_to"] = data["status_changed_to"].value
        return data

    @classmethod
    def model_validate_json_safe(cls, json_data: Dict[str, Any]):
        if 'timestamp' in json_data and isinstance(json_data['timestamp'], str):
            json_data['timestamp'] = datetime.fromisoformat(json_data['timestamp'])
        if 'item_type' in json_data and isinstance(json_data['item_type'], str):
            json_data['item_type'] = ItemType(json_data['item_type'])
        if 'status_changed_to' in json_data and isinstance(json_data['status_changed_to'], str):
            json_data['status_changed_to'] = ActionStatus(json_data['status_changed_to'])
        return cls.model_validate(json_data)


class UndoLogEntry(BaseModel):
    action_type: Literal["ai_create_item", "ai_modify_item"]
    item_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    # For 'create', previous_data is None. For 'modify', it's the ActionItem before modification.
    previous_data_json: Optional[str] = None  # Store as JSON string

    def model_dump_json_safe(self, *args, **kwargs):
        data = self.model_dump(*args, **kwargs)
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        return data

    @classmethod
    def model_validate_json_safe(cls, json_data: Dict[str, Any]):
        if 'timestamp' in json_data and isinstance(json_data['timestamp'], str):
            json_data['timestamp'] = datetime.fromisoformat(json_data['timestamp'])
        return cls.model_validate(json_data)


# --- ActionManagerEnhanced ---
class ActionManagerEnhanced:
    DB_ITEMS_PREFIX = "donext_items"
    DB_HISTORY_PREFIX = "donext_history"
    DB_CURRENT_ITEM_PREFIX = "donext_current_item"
    DB_UNDO_LOG_PREFIX = "donext_undo_log"

    def __init__(self, app: App, user_id: str):
        self.app = app
        self.user_id = user_id
        self.db = app.get_mod("DB")  # Assumes DB module is loaded
        self.isaa = app.get_mod("isaa")  # Assumes ISAA module is loaded

        self.items: List[ActionItem] = []
        self.history: List[HistoryEntry] = []
        self.current_item: Optional[ActionItem] = None
        self.undo_log: List[UndoLogEntry] = []

        self._load_data()

    def _get_db_key(self, prefix: str) -> str:
        return f"{prefix}_{self.user_id}"

    def _load_data(self):
        items_key = self._get_db_key(self.DB_ITEMS_PREFIX)
        history_key = self._get_db_key(self.DB_HISTORY_PREFIX)
        current_item_key = self._get_db_key(self.DB_CURRENT_ITEM_PREFIX)
        undo_log_key = self._get_db_key(self.DB_UNDO_LOG_PREFIX)

        try:
            items_data = self.db.get(items_key)
            if items_data.is_data() and items_data.get():
                loaded_items = json.loads(items_data.get()[0]) if isinstance(items_data.get(), list) else json.loads(
                    items_data.get())
                self.items = [ActionItem.model_validate_json_safe(item_dict) for item_dict in loaded_items]

            history_data = self.db.get(history_key)
            if history_data.is_data() and history_data.get():
                loaded_history = json.loads(history_data.get()[0]) if isinstance(history_data.get(),
                                                                                 list) else json.loads(
                    history_data.get())
                self.history = [HistoryEntry.model_validate_json_safe(entry_dict) for entry_dict in loaded_history]

            current_item_data = self.db.get(current_item_key)
            if current_item_data.is_data() and current_item_data.get():
                current_item_dict = json.loads(current_item_data.get()[0]) if isinstance(current_item_data.get(),
                                                                                         list) else json.loads(
                    current_item_data.get())
                if current_item_dict:
                    self.current_item = ActionItem.model_validate_json_safe(current_item_dict)

            undo_log_data = self.db.get(undo_log_key)
            if undo_log_data.is_data() and undo_log_data.get():
                loaded_undo = json.loads(undo_log_data.get()[0]) if isinstance(undo_log_data.get(),
                                                                               list) else json.loads(
                    undo_log_data.get())
                self.undo_log = [UndoLogEntry.model_validate_json_safe(entry_dict) for entry_dict in loaded_undo]

        except Exception as e:
            self.app.logger.error(f"Error loading data for user {self.user_id}: {e}")
            self.items = []
            self.history = []
            self.current_item = None
            self.undo_log = []

        self._recalculate_next_due_for_all()

    def _save_data(self):
        try:
            self.db.set(self._get_db_key(self.DB_ITEMS_PREFIX),
                        json.dumps([item.model_dump_json_safe() for item in self.items]))
            self.db.set(self._get_db_key(self.DB_HISTORY_PREFIX),
                        json.dumps([entry.model_dump_json_safe() for entry in self.history]))
            self.db.set(self._get_db_key(self.DB_CURRENT_ITEM_PREFIX),
                        json.dumps(self.current_item.model_dump_json_safe() if self.current_item else None))
            self.db.set(self._get_db_key(self.DB_UNDO_LOG_PREFIX),
                        json.dumps([entry.model_dump_json_safe() for entry in self.undo_log]))
        except Exception as e:
            self.app.logger.error(f"Error saving data for user {self.user_id}: {e}")

    def _add_history_entry(self, item: ActionItem, status_override: Optional[ActionStatus] = None,
                           notes: Optional[str] = None):
        entry = HistoryEntry(
            item_id=item.id,
            item_title=item.title,
            item_type=item.item_type,
            status_changed_to=status_override or item.status,
            parent_id=item.parent_id,
            notes=notes
        )
        self.history.append(entry)

    def _recalculate_next_due(self, item: ActionItem):
        if item.status == ActionStatus.COMPLETED and item.item_type == ItemType.TASK:
            if item.frequency and item.frequency != Frequency.ONE_TIME:
                base_time = item.last_completed or datetime.now()
                if item.fixed_time:  # If there was an original fixed time, try to align with it
                    base_time = max(base_time, item.fixed_time)
                    # Align to original time of day if fixed_time was set
                    base_time = base_time.replace(hour=item.fixed_time.hour, minute=item.fixed_time.minute,
                                                  second=item.fixed_time.second,
                                                  microsecond=item.fixed_time.microsecond)

                if item.frequency == Frequency.DAILY:
                    item.next_due = base_time + timedelta(days=1)
                elif item.frequency == Frequency.WEEKLY:
                    item.next_due = base_time + timedelta(weeks=1)
                elif item.frequency == Frequency.MONTHLY:
                    # This is a simplification, for more accuracy use dateutil.relativedelta
                    item.next_due = base_time + timedelta(days=30)
                elif item.frequency == Frequency.ANNUALLY:
                    item.next_due = base_time + timedelta(days=365)

                # If next_due is in the past (e.g. completing an overdue daily task), advance it until it's in the future
                while item.next_due and item.next_due < datetime.now():
                    if item.frequency == Frequency.DAILY:
                        item.next_due += timedelta(days=1)
                    elif item.frequency == Frequency.WEEKLY:
                        item.next_due += timedelta(weeks=1)
                    elif item.frequency == Frequency.MONTHLY:
                        item.next_due += timedelta(days=30)
                    elif item.frequency == Frequency.ANNUALLY:
                        item.next_due += timedelta(days=365)
                    else:
                        break  # Should not happen for recurring tasks
                item.status = ActionStatus.NOT_STARTED  # Reset for next occurrence
            else:  # One-time task
                item.next_due = None
        elif item.status == ActionStatus.NOT_STARTED and item.fixed_time and not item.next_due:
            item.next_due = item.fixed_time
        # If a task is not completed and has a fixed time, that's its due date.
        # If it's recurring and not completed, next_due should reflect the upcoming due date.
        # This can get complex for overdue items. For now, this covers completion and initial setup.

    def _recalculate_next_due_for_all(self):
        for item in self.items:
            self._recalculate_next_due(item)

    def add_item(self, item_data: Dict[str, Any], by_ai: bool = False) -> ActionItem:
        # Ensure datetime fields are correctly parsed if coming from JSON
        for dt_field in ['fixed_time', 'created_at', 'updated_at', 'last_completed', 'next_due']:
            if dt_field in item_data and isinstance(item_data[dt_field], str):
                try:
                    item_data[dt_field] = datetime.fromisoformat(item_data[dt_field].replace('Z', '+00:00'))
                except ValueError:
                    item_data[dt_field] = None  # Or handle error

        item = ActionItem.model_validate(item_data)
        item.created_by_ai = by_ai

        if not item.next_due and item.fixed_time:  # Initial due date
            item.next_due = item.fixed_time

        self.items.append(item)
        self._add_history_entry(item, status_override=ActionStatus.NOT_STARTED,
                                notes="Item created" + (" by AI" if by_ai else ""))

        if by_ai:
            self._log_ai_action("ai_create_item", item.id)

        self._save_data()
        return item

    def get_item_by_id(self, item_id: str) -> Optional[ActionItem]:
        return next((item for item in self.items if item.id == item_id), None)

    def update_item(self, item_id: str, update_data: Dict[str, Any], by_ai: bool = False) -> Optional[ActionItem]:
        item = self.get_item_by_id(item_id)
        if not item:
            return None

        previous_data_json = item.model_dump_json() if by_ai else None

        for key, value in update_data.items():
            if hasattr(item, key):
                # Handle datetime conversion if value is string
                if key in ['fixed_time', 'last_completed', 'next_due'] and isinstance(value, str):
                    try:
                        value = datetime.fromisoformat(value.replace('Z', '+00:00')) if value else None
                    except ValueError:
                        value = None
                elif key in ['item_type', 'frequency', 'status'] and isinstance(value, str):
                    try:
                        if key == 'item_type':
                            value = ItemType(value)
                        elif key == 'frequency':
                            value = Frequency(value)
                        elif key == 'status':
                            value = ActionStatus(value)
                    except ValueError:
                        # Keep original value if parsing fails
                        self.app.logger.warning(f"Failed to parse enum value '{value}' for field '{key}'")
                        continue
                setattr(item, key, value)

        item.updated_at = datetime.now()
        item.created_by_ai = by_ai  # If AI modifies it, mark it

        self._recalculate_next_due(item)  # Recalculate if status/frequency/time changed
        self._add_history_entry(item, notes="Item updated" + (" by AI" if by_ai else ""))

        if by_ai:
            self._log_ai_action("ai_modify_item", item.id, previous_data_json)

        self._save_data()
        return item

    def remove_item(self, item_id: str) -> bool:
        item = self.get_item_by_id(item_id)
        if not item:
            return False

        # Also remove children recursively
        children_to_remove = [child.id for child in self.items if child.parent_id == item_id]
        for child_id in children_to_remove:
            self.remove_item(child_id)  # Recursive call

        self.items = [i for i in self.items if i.id != item_id]

        # If it was the current item, clear it
        if self.current_item and self.current_item.id == item_id:
            self.current_item = None

        self._add_history_entry(item, status_override=ActionStatus.CANCELLED, notes="Item removed")
        self._save_data()
        return True

    def set_current_item(self, item_id: str) -> Optional[ActionItem]:
        item = self.get_item_by_id(item_id)
        if not item:
            return None

        if item.status == ActionStatus.COMPLETED and item.item_type == ItemType.TASK and item.frequency == Frequency.ONE_TIME:
            # Don't set a completed one-time task as current
            return None

        # If it's a recurring task that was completed, its status would have been reset to NOT_STARTED by _recalculate_next_due

        self.current_item = item
        if item.status == ActionStatus.NOT_STARTED:
            item.status = ActionStatus.IN_PROGRESS
            self._add_history_entry(item, notes="Set as current, status changed to In Progress")
        else:
            self._add_history_entry(item, notes="Set as current")

        self._save_data()
        return item

    def complete_current_item(self) -> Optional[ActionItem]:
        if not self.current_item:
            return None

        item_to_complete = self.current_item
        item_to_complete.status = ActionStatus.COMPLETED
        item_to_complete.last_completed = datetime.now()

        self._recalculate_next_due(item_to_complete)  # This might reset status for recurring tasks

        self._add_history_entry(item_to_complete, status_override=ActionStatus.COMPLETED, notes="Marked as completed")

        # If it was a one-time task or note, it's done. Otherwise, it might have a new due date.
        # For now, clear current_item after completion. User can re-select if it's recurring.
        self.current_item = None
        self._save_data()
        return item_to_complete

    def get_suggestions(self, count: int = 2) -> List[ActionItem]:
        if not self.isaa:
            self.app.logger.warning("ISAA module not available for suggestions.")
            return self._get_basic_suggestions(count)

        # AI-powered suggestions
        # Prepare context for ISAA
        active_items = [
            item.model_dump_json_safe() for item in self.items
            if item.status != ActionStatus.COMPLETED and item.status != ActionStatus.CANCELLED
        ]

        # Limit context size
        MAX_ITEMS_FOR_CONTEXT = 20
        if len(active_items) > MAX_ITEMS_FOR_CONTEXT:
            # Sort by priority then by next_due (earliest first, None last)
            active_items.sort(key=lambda x: (x.get('priority', 3),
                                             x.get('next_due') or '9999-12-31T23:59:59'))  # Crude sort for Nones
            active_items = active_items[:MAX_ITEMS_FOR_CONTEXT]

        current_time_str = datetime.now().isoformat()

        prompt = (
            f"Given the current time {current_time_str} and the following active items (tasks/notes), "
            f"suggest the top {count} item IDs that should be focused on next. Consider priority, "
            f"due dates (next_due), and item type. Tasks are generally more actionable than notes unless specified. "
            f"If a current item is set (see current_item_id), prioritize its sub-items if any are pressing, "
            f"otherwise suggest other top-level items. Focus on items that are 'not_started' or 'in_progress'.\n\n"
            f"Active Items (JSON):\n{json.dumps(active_items, indent=2)}\n\n"
            f"Current Item ID: {self.current_item.id if self.current_item else 'None'}\n\n"
            f"Return a JSON list of strings, where each string is an item ID. For example: [\"id1\", \"id2\"]."
        )

        class SuggestedIds(BaseModel):
            suggested_item_ids: List[str]

        try:
            # Using run_agent with a specific agent designed for this, or format_class
            # For simplicity, let's assume format_class can handle this if the agent is powerful enough
            # In a real scenario, you'd have an agent like "SuggestionAgent"
            structured_response = asyncio.run(self.isaa.format_class(SuggestedIds, prompt,
                                                                     agent_name="TaskCompletion"))  # or a dedicated suggestion agent

            if structured_response and isinstance(structured_response, dict):  # format_class returns a dict
                suggested_ids_model = SuggestedIds(**structured_response)
                suggested_item_ids = suggested_ids_model.suggested_item_ids

                suggestions = []
                for item_id in suggested_item_ids:
                    item = self.get_item_by_id(item_id)
                    if item:
                        suggestions.append(item)
                    if len(suggestions) == count:
                        break
                if suggestions:
                    return suggestions

            self.app.logger.warning(
                "AI suggestion failed or returned empty/invalid data. Falling back to basic suggestions.")
            return self._get_basic_suggestions(count)

        except Exception as e:
            self.app.logger.error(f"Error getting AI suggestions: {e}")
            return self._get_basic_suggestions(count)

    def _get_basic_suggestions(self, count: int = 2) -> List[ActionItem]:
        # Basic suggestion logic (priority, due date)
        available_items = [
            item for item in self.items
            if item.status == ActionStatus.NOT_STARTED or item.status == ActionStatus.IN_PROGRESS
        ]

        # If current item exists, prioritize its direct sub-items
        if self.current_item:
            sub_items = [
                item for item in available_items if item.parent_id == self.current_item.id
            ]
            if sub_items:
                available_items = sub_items  # Focus only on sub-items

        # Sort by: 1. Priority (lower number is higher), 2. Next Due Date (earlier is better, None is last)
        def sort_key(item: ActionItem):
            due_date = item.next_due if item.next_due else datetime.max
            return (item.priority, due_date)

        available_items.sort(key=sort_key)
        return available_items[:count]

    def get_history(self, limit: int = 50) -> List[HistoryEntry]:
        return sorted(self.history, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_all_items_hierarchy(self) -> Dict[str, List[Dict[str, Any]]]:
        hierarchy = {"root": []}
        item_map = {item.id: item.model_dump_json_safe() for item in self.items}

        processed_ids = set()

        # Build the hierarchy: Place items under their parents or in root
        for item_id, item_dict in item_map.items():
            parent_id = item_dict.get("parent_id")
            if parent_id and parent_id in item_map:  # It's a child
                if "children" not in item_map[parent_id]:
                    item_map[parent_id]["children"] = []
                item_map[parent_id]["children"].append(item_dict)
            else:  # It's a root item or parent doesn't exist (treat as root)
                hierarchy["root"].append(item_dict)
            processed_ids.add(item_id)

        # Sort children within each node if needed, e.g., by creation date or priority
        def sort_children_recursive(node_list):
            for node_dict in node_list:
                if "children" in node_dict:
                    node_dict["children"].sort(key=lambda x: (x.get('priority', 3), x.get('created_at')))
                    sort_children_recursive(node_dict["children"])

        hierarchy["root"].sort(key=lambda x: (x.get('priority', 3), x.get('created_at')))
        sort_children_recursive(hierarchy["root"])

        return hierarchy

    # --- AI Specific Methods ---
    async def ai_create_item_from_text(self, text: str) -> Optional[ActionItem]:
        if not self.isaa:
            self.app.logger.warning("ISAA module not available for AI item creation.")
            return None

        # Pydantic model for ISAA to parse into
        class ParsedItemFromText(BaseModel):
            item_type: Literal["task", "note"] = "task"
            title: str
            description: Optional[str] = None
            priority: Optional[int] = Field(default=3, ge=1, le=5)
            # ISAA should try to parse date string like "tomorrow", "next monday", "2024-12-25"
            due_date_str: Optional[str] = None
            frequency_str: Optional[str] = Field(default="one_time",
                                                 description="e.g. 'daily', 'weekly', 'one_time', 'every friday'")

        prompt = (
            f"Parse the following user input into a structured item (task or note). "
            f"Determine the item_type, title, description. If it's a task, "
            f"also determine priority (1-5, 1 is highest), and a potential due_date_str. "
            f"For due_date_str, convert relative dates like 'tomorrow', 'next week' into specific "
            f"date strings like 'YYYY-MM-DD'. Also try to infer frequency_str from terms like 'daily', 'every monday'. "
            f"If no specific type is clear, assume 'task'. The current date is {datetime.now().strftime('%Y-%m-%d')}.\n"
            f"User input: \"{text}\"\n\n"
            f"Format the output as JSON matching the Pydantic model ParsedItemFromText."
        )

        try:
            # parsed_data = await self.isaa.format_class(ParsedItemFromText, prompt, agent_name="TaskCompletion") # or specific agent
            # Workaround if format_class returns raw string, common with some LLMs not strictly adhering to JSON
            raw_response = await self.isaa.mini_task_completion(prompt,
                                                                agent_name="TaskCompletion")  # or dedicated agent
            if not raw_response:
                self.app.logger.error("AI parsing returned empty response.")
                return None

            # Extract JSON from the raw_response (it might be wrapped in markdown or text)
            json_str = raw_response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()

            parsed_dict = json.loads(json_str)
            parsed_data_model = ParsedItemFromText(**parsed_dict)

            if parsed_data_model:
                item_constructor_data = {
                    "item_type": ItemType(parsed_data_model.item_type),
                    "title": parsed_data_model.title,
                    "description": parsed_data_model.description,
                    "priority": parsed_data_model.priority or 3,
                }

                # Date parsing logic (can be complex, ISAA should ideally give YYYY-MM-DD)
                if parsed_data_model.due_date_str:
                    try:
                        # This is a placeholder. Real date parsing from natural language is hard.
                        # ISAA should be prompted to return YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
                        # For now, assume ISAA gives a parseable date string.
                        item_constructor_data["fixed_time"] = datetime.fromisoformat(
                            parsed_data_model.due_date_str.replace('Z', '+00:00'))
                    except ValueError:
                        self.app.logger.warning(
                            f"AI returned unparseable due_date_str: {parsed_data_model.due_date_str}")
                        # Basic relative date parsing
                        today = datetime.now()
                        if "tomorrow" in parsed_data_model.due_date_str.lower():
                            item_constructor_data["fixed_time"] = (today + timedelta(days=1)).replace(hour=9, minute=0,
                                                                                                      second=0,
                                                                                                      microsecond=0)  # Default to 9 AM
                        # Add more rules for "next week", "monday", etc. This is complex.
                        # Best if LLM returns standard ISO format.

                # Frequency parsing
                if parsed_data_model.frequency_str:
                    freq_str_lower = parsed_data_model.frequency_str.lower()
                    if "daily" in freq_str_lower:
                        item_constructor_data["frequency"] = Frequency.DAILY
                    elif "weekly" in freq_str_lower:
                        item_constructor_data["frequency"] = Frequency.WEEKLY
                    elif "monthly" in freq_str_lower:
                        item_constructor_data["frequency"] = Frequency.MONTHLY
                    elif "annually" in freq_str_lower or "yearly" in freq_str_lower:
                        item_constructor_data["frequency"] = Frequency.ANNUALLY
                    else:
                        item_constructor_data["frequency"] = Frequency.ONE_TIME

                return self.add_item(item_constructor_data, by_ai=True)
            return None
        except Exception as e:
            self.app.logger.error(
                f"Error creating item with AI: {e}. Raw response: {raw_response if 'raw_response' in locals() else 'N/A'}")
            return None

    def _log_ai_action(self, action_type: Literal["ai_create_item", "ai_modify_item"], item_id: str,
                       previous_data_json: Optional[str] = None):
        entry = UndoLogEntry(
            action_type=action_type,
            item_id=item_id,
            previous_data_json=previous_data_json
        )
        self.undo_log.append(entry)
        # Keep undo log reasonably sized
        MAX_UNDO_LOG_SIZE = 20
        if len(self.undo_log) > MAX_UNDO_LOG_SIZE:
            self.undo_log = self.undo_log[-MAX_UNDO_LOG_SIZE:]
        # self._save_data() # Save data is called by the calling function (add_item/update_item)

    async def undo_last_ai_action(self) -> bool:
        if not self.undo_log:
            return False

        last_action = self.undo_log.pop()
        action_undone = False

        if last_action.action_type == "ai_create_item":
            if self.remove_item(last_action.item_id):
                action_undone = True
        elif last_action.action_type == "ai_modify_item":
            if last_action.previous_data_json:
                try:
                    previous_data = ActionItem.model_validate_json_safe(json.loads(last_action.previous_data_json))
                    # Replace the current item with the previous version
                    self.items = [item for item in self.items if item.id != last_action.item_id]
                    self.items.append(previous_data)

                    # If it was the current item, update it
                    if self.current_item and self.current_item.id == last_action.item_id:
                        self.current_item = previous_data

                    action_undone = True
                except Exception as e:
                    self.app.logger.error(f"Error restoring item during undo: {e}")
                    # Put the action back if undo failed, so it's not lost
                    self.undo_log.append(last_action)
            else:  # Should not happen for modify
                self.app.logger.warning(
                    f"Undo for modify action on item {last_action.item_id} had no previous_data_json.")

        if action_undone:
            self._add_history_entry(
                # Create a temporary HistoryEntry as the item might be gone or changed
                HistoryEntry(item_id=last_action.item_id, item_title="N/A (Undone AI Action)", item_type=ItemType.TASK,
                             status_changed_to=ActionStatus.CANCELLED),
                notes=f"Undid AI action: {last_action.action_type} on item {last_action.item_id}"
            )
            self._save_data()

        return action_undone


# --- Manager Cache ---
_managers: Dict[str, ActionManagerEnhanced] = {}


async def get_manager(app: App, request: RequestData) -> ActionManagerEnhanced:
    user = await get_user_from_request(app, request)  # from toolboxv2.utils.extras.base_widget
    user_id = user.uid if user and user.uid else "default_public_user"  # Fallback for unauthenticated

    if user_id not in _managers:
        _managers[user_id] = ActionManagerEnhanced(app, user_id)
    return _managers[user_id]


# --- API Endpoints ---

@export(mod_name=Name, name="init_config", initial=True)  # `name` changed to avoid conflict if module is reloaded
def init_POA_module(app: App):
    app.run_any(("CloudM", "add_ui"),
                name=Name,
                title="DoNext Enhanced",
                path=f"/api/{Name}/main_page",  # Changed path slightly for clarity
                description="Enhanced Task and Note Management with AI"
                )
    app.logger.info(f"{Name} module initialized and UI registered.")


@export(mod_name=Name, name="new-item", api=True, request_as_kwarg=True, api_methods=['POST'])
async def api_new_item(app: App, request: RequestData, data=Name):
    manager = await get_manager(app, request)
    try:
        item_data = data
        # Default to task if not specified, client should send item_type
        if 'item_type' not in item_data:
            item_data['item_type'] = 'task'
        item = manager.add_item(item_data)
        return Result.json(data=item.model_dump_json_safe())
    except json.JSONDecodeError:
        return Result.default_user_error("Invalid JSON payload", 400)
    except Exception as e:
        app.logger.error(f"Error in new-item: {e}", exc_info=True)
        return Result.default_internal_error(f"Could not create item: {e}")


@export(mod_name=Name, name="set-current-item", api=True, request_as_kwarg=True, api_methods=['POST'])
async def api_set_current_item(app: App, request: RequestData, item_id: Optional[str] = None, data=None):
    if not item_id:  # Try to get from body if not in query params
        try:
            body = data
            item_id = body.get('item_id')
        except:  # Ignore if body is not JSON or item_id not present
            pass
    if not item_id:
        return Result.default_user_error("item_id is required.", 400)

    manager = await get_manager(app, request)
    item = manager.set_current_item(item_id)
    if item:
        return Result.json(data=item.model_dump_json_safe())
    return Result.default_user_error("Item not found or cannot be set as current.", 404)


@export(mod_name=Name, name="complete-current-item", api=True, request_as_kwarg=True, api_methods=['POST'])
async def api_complete_current_item(app: App, request: RequestData):
    manager = await get_manager(app, request)
    item = manager.complete_current_item()
    if item:
        return Result.json(data=item.model_dump_json_safe())
    return Result.default_user_error("No current item to complete.", 400)


@export(mod_name=Name, name="get-current-item", api=True, request_as_kwarg=True)
async def api_get_current_item(app: App, request: RequestData):
    manager = await get_manager(app, request)
    if manager.current_item:
        return Result.json(data=manager.current_item.model_dump_json_safe())
    return Result.json(data=None)  # No current item


@export(mod_name=Name, name="suggestions", api=True, request_as_kwarg=True)
async def api_get_suggestions(app: App, request: RequestData):
    manager = await get_manager(app, request)
    suggestions = manager.get_suggestions(count=2)  # AI suggestions now
    return Result.json(data=[s.model_dump_json_safe() for s in suggestions])


@export(mod_name=Name, name="all-items-hierarchy", api=True, request_as_kwarg=True)
async def api_get_all_items_hierarchy(app: App, request: RequestData):
    manager = await get_manager(app, request)
    return Result.json(data=manager.get_all_items_hierarchy())


@export(mod_name=Name, name="history", api=True, request_as_kwarg=True)
async def api_get_history(app: App, request: RequestData):
    manager = await get_manager(app, request)
    history_entries = manager.get_history()
    return Result.json(data=[h.model_dump_json_safe() for h in history_entries])


@export(mod_name=Name, name="remove-item", api=True, request_as_kwarg=True,
        api_methods=['POST'])  # Changed to POST, common for delete actions
async def api_remove_item(app: App, request: RequestData, item_id: Optional[str] = None, data=None):
    if not item_id:  # Try to get from body if not in query params
        try:
            body = data
            item_id = body.get('item_id')
        except:
            pass
    if not item_id:
        return Result.default_user_error("item_id is required for removal.", 400)

    manager = await get_manager(app, request)
    if manager.remove_item(item_id):
        return Result.ok("Item removed successfully.")
    return Result.default_user_error("Item not found.", 404)


# --- AI Endpoints ---
@export(mod_name=Name, name="ai-process-text", api=True, request_as_kwarg=True, api_methods=['POST'])
async def api_ai_process_text(app: App, request: RequestData, data):
    manager = await get_manager(app, request)
    try:
        text_input = data.get("text")
        if not text_input:
            return Result.default_user_error("Text input is required.", 400)

        item = await manager.ai_create_item_from_text(text_input)
        if item:
            return Result.json(data=item.model_dump_json_safe(), info="Item created by AI.")
        return Result.default_user_error("AI could not process text into an item.",
                                         500)  # Or 422 if parsing failed client-side
    except json.JSONDecodeError:
        return Result.default_user_error("Invalid JSON payload", 400)
    except Exception as e:
        app.logger.error(f"Error in ai-process-text: {e}", exc_info=True)
        return Result.default_internal_error(f"Could not process text with AI: {e}")


@export(mod_name=Name, name="undo-ai-action", api=True, request_as_kwarg=True, api_methods=['POST'])
async def api_undo_ai_action(app: App, request: RequestData):
    manager = await get_manager(app, request)
    if await manager.undo_last_ai_action():
        return Result.ok("Last AI action undone.")
    return Result.default_user_error("No AI action to undo or undo failed.", 400)


# --- Frontend HTML (Adapted from DoNext) ---
# This is a large string. In a real app, it would be in a .html file.
# Key changes to make in the HTML/JS:
# - API endpoints need to be updated (e.g., /api/POA/new-item)
# - `generateId()` in JS should be removed; server generates IDs.
# - New Item Modal needs `item_type` (task/note) selector.
# - Display should distinguish tasks/notes (e.g., icons).
# - Add AI input field and "Undo AI" button.
# - JS functions to call the new AI endpoints.

# NOTE: Due to the character limit and focus on backend, the full HTML is omitted here.
#       It would be DoNext.py's `template` string, with JS updated to:
#       1. Call the new `/api/POA/*` endpoints.
#       2. Send `item_type` when creating new items.
#       3. Handle AI input and undo button.
#       4. Display tasks and notes distinctly.
#       5. Remove client-side ID generation.
#       6. Update `currentAction` to `currentItem`, `allActions` to `allItems`, etc.
#       7. Adapt badge classes and display logic for new fields/enums.

# Placeholder for the main page - you'd serve the adapted DoNext HTML here
@get_app().tb(mod_name=Name, version=version, level=0, api=True,
              name="main_page", row=True, state=False)
def POAPage(app_ref: Optional[App] = None):  # Renamed arg to avoid conflict with outer app
    app_instance = app_ref if app_ref else get_app(Name)

    # This is where you would load and adapt DoNext.py's HTML template
    # For now, a simple message:
    # return Result.html("<html><body><h1>DoNext Enhanced Page</h1><p>Full UI to be implemented by adapting DoNext.py template.</p></body></html>")

    # For a quick test, let's include a simplified version of DoNext's HTML template
    # with some key JS functions needing updates pointed out.
    # IMPORTANT: This HTML is NOT fully functional with the new backend without significant JS updates.
    # It's provided for structural context.

    # --- Start of Simplified DoNext HTML Template (for context) ---
    # (Imagine the full HTML from DoNext.py here, with comments on JS changes)
    # --- End of Simplified DoNext HTML Template ---

    # For the sake_of_this_example, let's return the original DoNext template string.
    # The user will need to update the JavaScript within this template.
    template_html_content = """
<div>
    <title>Action Manager Enhanced</title> <!-- Updated Title -->

    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        body {
            background: #f0f2f5;
            color: #1a1a1a;
            min-height: 100vh;
        }

        .app-container {
            max-width: 800px;
            margin: 0 auto;
            flex-direction: column;
            gap: 16px;
            padding: 20px;
        }

        /* ... (all other styles from DoNext.py's template) ... */
        .card {
            color: var(--text-color);
            background: var(--theme-bg);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        .item-icon { /* For distinguishing tasks and notes */
            margin-right: 8px;
            font-size: 1.2em;
        }

        .badge {
            background: #e3e8ef;
            padding: 2px 4px;
            border-radius: 6px;
            font-size: 0.9em;
            margin-left: 6px;
            max-width: 100px;
            color: #000000;
        }

        .badge.priority-1 { background: #ff4d4f; color: white; }
        .badge.priority-2 { background: #ff7a45; color: white; }
        .badge.priority-3 { background: #ffa940; color: white; }
        .badge.priority-4 { background: #bae637; color: black; }
        .badge.priority-5 { background: #73d13d; color: white; }

        .badge.status-in_progress { background: #1890ff; color: white; }
        .badge.status-completed { background: #52c41a; color: white; }
        .badge.status-not_started { background: #d9d9d9; color: black; }
        .badge.status-cancelled { background: #bfbfbf; color: black; }


        .action-content, .item-content { /* Renamed for clarity */
            margin: 16px 0;
            font-size: 1.1em;
        }

        .button-group {
            display: flex;
            gap: 8px;
            margin-top: 16px;
        }

        .btn {
            padding: 8px 16px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-weight: 500;
            transition: background-color 0.2s;
        }

        .btn-primary {
            background: #1890ff;
            color: white;
        }

        .btn-primary:hover {
            background: #096dd9;
        }

        .btn-secondary {
            background: #f0f0f0;
            color: #1a1a1a;
        }

        .btn-secondary:hover {
            background: #d9d9d9;
        }

        .btn-warning { /* For Undo AI */
            background: #faad14;
            color: white;
        }
        .btn-warning:hover {
            background: #d48806;
        }


        .btn-remove {
            background: #a81103;
            color: #1a1a1a;
        }

        .btn-remove:hover {
            background: #75150c;
        }

        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 1000;
        }

        .modal-content {
            background: white;
            border-radius: 16px;
            padding: 24px;
            width: 90%;
            max-width: 500px;
            margin: 50px auto;
            height: 90%;
            overflow-y: auto;
        }

        .input-group {
            margin-bottom: 16px;
        }

        .input-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
        }

        .input-group input,
        .input-group select,
        .input-group textarea {
            width: 100%;
            padding: 8px;
            border: 1px solid #d9d9d9;
            border-radius: 8px;
            font-size: 1em;
            background-color:  #e3e8ef;
            color: #000000;
        }
        .ai-input-section {
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            background-color: var(--theme-bg);
        }
        .ai-input-section input {
            flex-grow: 1;
        }

        .history-section {
            transition: max-height 0.3s ease-out;
            overflow: hidden;
        }

        .history-header {
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .history-list {
            margin-top: 16px;
        }

        .history-item {
            padding: 12px;
            border-bottom: 1px solid #f0f0f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .history-item:last-child {
            border-bottom: none;
        }

        .item-hierarchy { /* Renamed */
            margin-top: 16px;
        }

        .sub-item { /* Renamed */
            margin-left: 24px;
            padding-left: 12px;
            border-left: 2px solid #f0f0f0;
        }

        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }

        .tab {
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            background: #f0f0f0;
        }

        .tab.active {
            background: #1890ff;
            color: white;
        }
        .current-item-section { /* Renamed */
            background: #2C3E50;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

    </style>

    <div class="app-container">
        <div class="tabs">
            <div class="tab active" onclick="switchTab('main')">Current Focus</div>
            <div class="tab" onclick="switchTab('all')">All Items</div>
            <div class="tab" onclick="switchTab('history')">History</div>
        </div>

        <!-- AI Quick Add Section -->
        <section class="card ai-input-section">
            <input type="text" id="aiTextInput" placeholder="Quick add with AI (e.g., 'Task: Review report by Friday p1' or 'Note: Idea for project X')" class="tb-input">
            <button class="btn btn-primary" onclick="submitAiText()">Add</button>
            <button class="btn btn-warning" onclick="undoAiAction()">Undo AI</button>
        </section>

        <div id="main-tab">
            <section class="card current-item-section"> <!-- Renamed class -->
                <div class="card-header">
                    <h2>Current Item</h2>
                    <div>
                        <span class="badge" id="elapsed-time" style="background: #2C3E50">00:00:00</span>
                    </div>
                </div>
                <div class="item-content" id="current-item-content"> <!-- Renamed class -->
                    No current item
                </div>
                <div class="item-hierarchy" id="current-sub-items"> <!-- Renamed class & id -->
                    <!-- Sub-items will be rendered here -->
                </div>
                <div class="button-group">
                    <button class="btn btn-primary" onclick="openNewItemModal()">New Item</button>
                    <button class="btn btn-secondary" onclick="completeCurrentItem()">Complete</button>
                </div>
            </section>

            <section class="card suggestion" id="suggestion1">
                <div class="card-header">
                    <h3>Next Suggested Item</h3>
                    <span class="badge" id="suggestion1-priority"></span>
                </div>
                <div class="item-content" id="suggestion1-content"> <!-- Renamed class -->
                    Loading...
                </div>
                <div class="button-group">
                    <button class="btn btn-primary" onclick="startSuggestion(1)">Start</button>
                </div>
            </section>

            <section class="card suggestion" id="suggestion2">
                <div class="card-header">
                    <h3>Alternative Item</h3>
                    <span class="badge" id="suggestion2-priority"></span>
                </div>
                <div class="item-content" id="suggestion2-content"> <!-- Renamed class -->
                    Loading...
                </div>
                <div class="button-group">
                    <button class="btn btn-primary" onclick="startSuggestion(2)">Start</button>
                </div>
            </section>
        </div>

        <div id="all-items-tab" style="display: none;"> <!-- Renamed id -->
            <section class="card">
                <div class="card-header">
                    <h2>All Items</h2>
                    <button class="btn btn-primary" onclick="openNewItemModal()">New Item</button>
                </div>
                <div id="items-hierarchy"> <!-- Renamed id -->
                    <!-- Items hierarchy will be rendered here -->
                </div>
            </section>
        </div>

        <div id="history-tab" style="display: none;">
            <section class="card">
                <div class="card-header">
                    <h2>Item History</h2>
                </div>
                <div class="history-list" id="history-list">
                    <!-- History items will be rendered here -->
                </div>
            </section>
        </div>
    </div>

    <!-- New Item Modal -->
    <div class="modal" id="newItemModal"> <!-- Renamed id -->
        <div class="modal-content">
            <div class="card-header">
                <h2>Create New Item</h2>
                <button class="btn btn-secondary" onclick="closeNewItemModal()">×</button>
            </div>
            <form id="newItemForm" onsubmit="createNewItem(event)"> <!-- Renamed id -->
                <div class="input-group">
                    <label for="itemType">Item Type</label>
                    <select id="itemType" onchange="toggleTaskFields()">
                        <option value="task" selected>Task</option>
                        <option value="note">Note</option>
                    </select>
                </div>
                <div class="input-group">
                    <label for="itemTitle">Title</label>
                    <input type="text" id="itemTitle" required>
                </div>
                <div class="input-group">
                    <label for="itemDescription">Description / Note Content</label>
                    <textarea id="itemDescription" rows="3"></textarea>
                </div>
                <div class="input-group">
                    <label for="itemParent">Parent Item</label>
                    <select id="itemParent">
                        <option value="">None (Top-level item)</option>
                    </select>
                </div>
                <div class="input-group task-field"> <!-- Task specific -->
                    <label for="itemFrequency">Frequency</label>
                    <select id="itemFrequency" required>
                        <option value="one_time">One Time</option>
                        <option value="daily">Daily</option>
                        <option value="weekly">Weekly</option>
                        <option value="monthly">Monthly</option>
                        <option value="annually">Annually</option>
                    </select>
                </div>
                <div class="input-group task-field"> <!-- Task specific -->
                    <label for="itemPriority">Priority (1-5)</label>
                    <select id="itemPriority" required>
                        <option value="1">1 (Highest)</option>
                        <option value="2">2</option>
                        <option value="3" selected>3</option>
                        <option value="4">4</option>
                        <option value="5">5 (Lowest)</option>
                    </select>
                </div>
                <div class="input-group task-field"> <!-- Task specific -->
                    <label for="itemFixedTime">Fixed Time (Optional)</label>
                    <input type="datetime-local" id="itemFixedTime">
                </div>
                <div class="button-group">
                    <button type="submit" class="btn btn-primary">Create</button>
                </div>
            </form>
        </div>
    </div>

<script unSave="true">
if (window.history.state){ // Keep this conditional as per original
    // State management
    let currentItem = null;
    let suggestions = [];
    let allItems = {};
    let history = [];
    let elapsedTimeInterval;

    // Utility functions
    function formatDuration(ms) {
        const seconds = Math.floor((ms / 1000) % 60);
        const minutes = Math.floor((ms / (1000 * 60)) % 60);
        const hours = Math.floor(ms / (1000 * 60 * 60));
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }

    // UI functions
    function switchTabInternal(tab) {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelector(`.tab[onclick="switchTab('${tab}')"]`).classList.add('active');

        document.getElementById('main-tab').style.display = tab === 'main' ? 'block' : 'none';
        document.getElementById('all-items-tab').style.display = tab === 'all' ? 'block' : 'none';
        document.getElementById('history-tab').style.display = tab === 'history' ? 'block' : 'none';

        if (tab === 'all') {
            refreshItemsHierarchy();
        } else if (tab === 'history') {
            refreshHistory();
        }
    }

    function getItemIcon(itemType) {
        return itemType === 'task' ? '📝' : '🗒️';
    }

    function updateCurrentItemDisplay() {
        const content = document.getElementById('current-item-content');
        const subItems = document.getElementById('current-sub-items');

        if (currentItem) {
            let badges = `<div class="badge status-${currentItem.status.replace('_', '-')}">${currentItem.status.replace('_', ' ')}</div>`;
            if (currentItem.item_type === 'task') {
                 badges += `<div class="badge priority-${currentItem.priority}">Priority ${currentItem.priority}</div>`;
                 badges += `<div class="badge">${currentItem.frequency.replace('_', ' ')}</div>`;
            }

            content.innerHTML = `
                <h3><span class="item-icon">${getItemIcon(currentItem.item_type)}</span>${currentItem.title}</h3>
                ${currentItem.description ? `<p>${currentItem.description}</p>` : ''}
                ${badges}
            `;

            subItems.innerHTML = '';
            if (allItems && allItems.root) {
                const currentItemNode = findItemInHierarchy(allItems.root, currentItem.id);
                if (currentItemNode && currentItemNode.children) {
                     subItems.innerHTML = currentItemNode.children.map(subItem => `
                        <div class="sub-item card">
                            <h4><span class="item-icon">${getItemIcon(subItem.item_type)}</span>${subItem.title}</h4>
                            <div class="badge priority-${subItem.priority || 3}">Priority ${subItem.priority || 3}</div>
                            <button class="btn btn-secondary" onclick="startItem('${subItem.id}')">Start</button>
                        </div>
                    `).join('');
                }
            }

            let startTime = new Date(); // This should ideally be when item.status became IN_PROGRESS
            if (elapsedTimeInterval) clearInterval(elapsedTimeInterval);
            elapsedTimeInterval = setInterval(() => {
                const elapsed = Date.now() - startTime;
                document.getElementById('elapsed-time').textContent = formatDuration(elapsed);
            }, 1000);
        } else {
            content.innerHTML = 'No current item';
            subItems.innerHTML = '';
            if (elapsedTimeInterval) {
                clearInterval(elapsedTimeInterval);
                document.getElementById('elapsed-time').textContent = '00:00:00';
            }
        }
    }

    function findItemInHierarchy(nodes, itemId) {
        if (!nodes) return null;
        for (const node of nodes) {
            if (node.id === itemId) return node;
            if (node.children) {
                const found = findItemInHierarchy(node.children, itemId);
                if (found) return found;
            }
        }
        return null;
    }

    async function updateSuggestions() {
        try {
            const response = await window.TB.api.request('POA', 'suggestions', null, 'GET');
            if (response.error === window.TB.ToolBoxError.none) { // UPDATED
                const data = response.get();
                suggestions = data || [];
                (data || []).forEach((suggestion, index) => {
                    const suggestionNumber = index + 1;
                    if (suggestion) {
                        const contentEl = document.getElementById(`suggestion${suggestionNumber}-content`);
                        const priorityEl = document.getElementById(`suggestion${suggestionNumber}-priority`);

                        let badges = '';
                        if (suggestion.item_type === 'task') {
                             badges = `<div class="badge">${suggestion.frequency.replace('_', ' ')}</div>`;
                             priorityEl.textContent = `Priority ${suggestion.priority}`;
                             priorityEl.className = `badge priority-${suggestion.priority}`;
                        } else {
                            priorityEl.textContent = 'Note';
                            priorityEl.className = 'badge';
                        }

                        contentEl.innerHTML = `
                            <h3><span class="item-icon">${getItemIcon(suggestion.item_type)}</span>${suggestion.title}</h3>
                            ${suggestion.description ? `<p>${suggestion.description.substring(0,100)}...</p>` : ''}
                            ${badges}
                        `;
                    } else {
                        document.getElementById(`suggestion${suggestionNumber}-content`).innerHTML = 'No suggestion available';
                        document.getElementById(`suggestion${suggestionNumber}-priority`).textContent = '';
                    }
                 });
            } else {
                console.error("Error fetching suggestions:", response.info.help_text);
                [1, 2].forEach(index => { // Assuming only 2 suggestions based on HTML
                    document.getElementById(`suggestion${index}-content`).innerHTML = 'Error loading suggestions';
                    document.getElementById(`suggestion${index}-priority`).textContent = '';
                });
            }
        } catch (error) {
            console.error("Network error or critical issue in updateSuggestions:", error);
             [1, 2].forEach(index => {
                document.getElementById(`suggestion${index}-content`).innerHTML = 'Network error';
                document.getElementById(`suggestion${index}-priority`).textContent = '';
            });
        }
    }

    async function refreshItemsHierarchy() {
        try {
            const response = await window.TB.api.request('POA', 'all-items-hierarchy', null, 'GET');
            if (response.error === window.TB.ToolBoxError.none) { // UPDATED
                allItems = response.get() || { root: [] };
                const container = document.getElementById('items-hierarchy');
                container.innerHTML = (allItems.root || []).map(item => renderItemCard(item, 0)).join('');
            } else {
                console.error("Error fetching items hierarchy:", response.info.help_text);
                document.getElementById('items-hierarchy').innerHTML = '<p>Error loading items.</p>';
            }
        } catch (error) {
            console.error("Network error or critical issue in refreshItemsHierarchy:", error);
            document.getElementById('items-hierarchy').innerHTML = '<p>Network error loading items.</p>';
        }
    }

    function renderItemCard(item, depth) {
        let subItemsHtml = '';
        if (item.children && item.children.length > 0) {
            subItemsHtml = item.children.map(child => renderItemCard(child, depth + 1)).join('');
        }

        let itemBadges = `<span class="badge status-${item.status.replace('_','-')}">${item.status.replace('_',' ')}</span>`;
        if (item.item_type === 'task') {
            itemBadges += `<span class="badge priority-${item.priority}">Priority ${item.priority}</span>`;
        }
        if (item.fixed_time) {
             itemBadges += `<div class="badge">Due: ${new Date(item.fixed_time).toLocaleString()}</div>`;
        }

        return `
            <div class="card ${depth > 0 ? 'sub-item' : ''}" style="${depth > 0 ? 'margin-left: ' + (depth * 20) + 'px;' : ''}">
                <div class="card-header">
                    <h3><span class="item-icon">${getItemIcon(item.item_type)}</span>${item.title}</h3>
                    <div>${itemBadges}</div>
                </div>
                <div class="item-content">
                    ${item.description || ''}
                </div>
                <div class="item-hierarchy">
                    ${subItemsHtml}
                </div>
                <div class="button-group">
                    <button class="btn btn-primary" onclick="startItem('${item.id}')">Start</button>
                    ${item.item_type === 'task' ? `<button class="btn btn-secondary" onclick="openNewItemModal('${item.id}','task')">Add Sub-task</button>`: ''}
                     <button class="btn btn-secondary" onclick="openNewItemModal('${item.id}','note')">Add Sub-note</button>
                    <button class="btn btn-remove" onclick="removeItem('${item.id}')">Remove</button>
                </div>
            </div>
        `;
    }

    async function refreshHistory() {
        try {
            const response = await window.TB.api.request('POA', 'history', null, 'GET');
            if (response.error === window.TB.ToolBoxError.none) { // UPDATED
                history = response.get() || [];
                const container = document.getElementById('history-list');
                container.innerHTML = (history || []).map(entry => `
                    <div class="history-item">
                        <div>
                            <h4><span class="item-icon">${getItemIcon(entry.item_type)}</span>${entry.item_title}</h4>
                            <span class="badge status-${entry.status_changed_to.replace('_','-')}">${entry.status_changed_to.replace('_',' ')}</span>
                            ${entry.parent_id ? '<span class="badge">Sub-item</span>' : ''}
                            ${entry.notes ? `<span class="badge" style="background-color: lightblue;">${entry.notes}</span>` : ''}
                        </div>
                        <div>
                            ${new Date(entry.timestamp).toLocaleString()}
                        </div>
                    </div>
                `).join('');
            } else {
                console.error("Error fetching history:", response.info.help_text);
                document.getElementById('history-list').innerHTML = '<p>Error loading history.</p>';
            }
        } catch (error) {
            console.error("Network error or critical issue in refreshHistory:", error);
            document.getElementById('history-list').innerHTML = '<p>Network error loading history.</p>';
        }
    }

    // Item Management Functions
    async function removeItemInternal(itemId) {
     TB.ui.Modal.show({
        title: 'Remove item?',
        content:
          'Are you sure you want to remove this item and all its sub-items?',
        maxWidth: '24rem',                    // tweak as you like
        buttons: [
          {
            text: 'Cancel',
            variant: 'outline',               // neutral style
            action(modal) {
              modal.close();                  // simply dismiss
            },
          },
          {
            text: 'Remove',
            variant: 'danger',                // red / destructive style
            className: 'ml-2',
            async action(modal) {
              try {
                const response = await window.TB.api.request(
                  'POA',
                  'remove-item',
                  { item_id: itemId },
                  'POST'
                );

                if (response.error === window.TB.ToolBoxError.none) {
                  refreshItemsHierarchy();
                  updateSuggestions();

                  if (currentItem && currentItem.id === itemId) {
                    currentItem = null;
                    updateCurrentItemDisplay();
                  }

                  // optional success toast
                  TB.ui.Toast.showSuccess('Item removed.');
                } else {
                  TB.ui.Toast.showError(
                    `Error removing item: ${
                      response.info?.help_text ?? 'Unknown error.'
                    }`
                  );
                }
              } catch (error) {
                console.error(
                  'Network error or critical issue in removeItem:',
                  error
                );
                TB.ui.Toast.showError(
                  'Network error removing item. Please try again.'
                );
              } finally {
                modal.close();                // ensure modal closes either way
              }
            },
          },
        ],
      });
    }

    async function startItemInternal(itemId) {
        try {
            const response = await window.TB.api.request('POA', 'set-current-item', {item_id: itemId}, 'POST');
            if (response.error === window.TB.ToolBoxError.none && response.get()) { // UPDATED
                currentItem = response.get();
                updateCurrentItemDisplay();
                updateSuggestions();
                switchTabInternal('main'); // Use internal function
            } else {
                window.TB.ui.Toast.showError("Error starting item: " + (response.info.help_text || "Item data not found or unknown error.")); // UPDATED
            }
        } catch (error) {
            console.error("Network error or critical issue in startItem:", error);
            window.TB.ui.Toast.showError("Network error starting item. Please try again."); // UPDATED
        }
    }

    async function completeCurrentItemInternal() {
        if (!currentItem) return;
        try {
            const response = await window.TB.api.request('POA', 'complete-current-item', null, 'POST');
            if (response.error === window.TB.ToolBoxError.none) { // UPDATED
                currentItem = null;
                updateCurrentItemDisplay();
                updateSuggestions();
                refreshHistory();
                refreshItemsHierarchy();
            } else {
                 window.TB.ui.Toast.showError("Error completing item: " + (response.info.help_text || "Unknown error.")); // UPDATED
            }
        } catch (error) {
            console.error("Network error or critical issue in completeCurrentItem:", error);
            window.TB.ui.Toast.showError("Network error completing item. Please try again."); // UPDATED
        }
    }

    function startSuggestionInternal(index) {
        if (suggestions && suggestions[index - 1]) {
            startItemInternal(suggestions[index - 1].id); // Use internal function
        }
    }

    // Modal Management
    function openNewItemModalInternal(parentId = '', itemType = 'task') {
        const modal = document.getElementById('newItemModal');
        const parentSelect = document.getElementById('itemParent');
        const itemTypeSelect = document.getElementById('itemType');

        itemTypeSelect.value = itemType;
        toggleTaskFieldsInternal(); // Use internal function

        parentSelect.innerHTML = '<option value="">None (Top-level item)</option>';
        if (allItems && allItems.root) {
            function populateParentOptions(items, prefix = '') {
                items.forEach(item => {
                    parentSelect.innerHTML +=
                        `<option value="${item.id}" ${item.id === parentId ? 'selected' : ''}>${prefix}${item.title} (${item.item_type})</option>`;
                    if (item.children) {
                        populateParentOptions(item.children, prefix + '-- ');
                    }
                });
            }
            populateParentOptions(allItems.root);
        } else if (parentId) {
             parentSelect.innerHTML += `<option value="${parentId}" selected>Loading parent...</option>`;
        }

        modal.style.display = 'block';
        document.getElementById('itemTitle').focus();
    }

    function closeNewItemModalInternal() {
        document.getElementById('newItemModal').style.display = 'none';
        document.getElementById('newItemForm').reset();
        toggleTaskFieldsInternal(); // Use internal function
    }

    function toggleTaskFieldsInternal() {
        const itemType = document.getElementById('itemType').value;
        const taskFields = document.querySelectorAll('.task-field');
        const isTask = itemType === 'task';

        taskFields.forEach(field => {
            field.style.display = isTask ? 'block' : 'none';
            field.querySelectorAll('input, select').forEach(input => {
                if (isTask) input.setAttribute('required', 'required');
                else input.removeAttribute('required');
            });
        });
        document.querySelector('label[for="itemDescription"]').textContent = isTask ? 'Description' : 'Note Content';
    }

    async function init() {
        try {
            const currentItemResponse = await window.TB.api.request('POA', 'get-current-item', null, 'GET');
            if (currentItemResponse.error === window.TB.ToolBoxError.none) { // UPDATED
                currentItem = currentItemResponse.get();
            } else {
                console.error("Error fetching current item on init:", currentItemResponse.info.help_text);
            }

            await refreshItemsHierarchy();
            updateCurrentItemDisplay();
            await updateSuggestions();
            await refreshHistory();

        } catch (error) {
            console.error("Initialization error (network or critical issue):", error);
            window.TB.ui.Toast.showError("Failed to initialize application. Some features might not work. Please try refreshing."); // UPDATED
        }
    }

    async function createNewItemInternal(event) {
        event.preventDefault();

        const itemType = document.getElementById('itemType').value;
        const formData = {
            item_type: itemType,
            title: document.getElementById('itemTitle').value,
            description: document.getElementById('itemDescription').value,
            parent_id: document.getElementById('itemParent').value || null,
        };

        if (itemType === 'task') {
            formData.frequency = document.getElementById('itemFrequency').value;
            formData.priority = parseInt(document.getElementById('itemPriority').value);
            const fixedTimeVal = document.getElementById('itemFixedTime').value;
            formData.fixed_time = fixedTimeVal ? new Date(fixedTimeVal).toISOString() : null;
        }

        try {
            const response = await window.TB.api.request('POA', 'new-item', formData, 'POST');
            if (response.error === window.TB.ToolBoxError.none) { // UPDATED
                closeNewItemModalInternal(); // Use internal function
                refreshItemsHierarchy();
                updateSuggestions();
            } else {
                window.TB.ui.Toast.showError("Error creating item: " + (response.info.help_text || "Unknown error.")); // UPDATED
            }
        } catch (error) {
            console.error("Network error or critical issue in createNewItem:", error);
            window.TB.ui.Toast.showError("Network error creating item. Please try again."); // UPDATED
        }
    }

    // --- NEW AI Functions ---
    async function submitAiTextInternal() {
        const textInput = document.getElementById('aiTextInput').value;
        if (!textInput.trim()) return;

        try {
            const payload = {text: textInput};
            const response = await window.TB.api.request('POA', 'ai-process-text', payload, 'POST');

            if (response.error === window.TB.ToolBoxError.none && response.get()) { // UPDATED
                document.getElementById('aiTextInput').value = '';
                refreshItemsHierarchy();
                updateSuggestions();
            } else {
                window.TB.ui.Toast.showError("AI processing failed: " + (response.info.help_text || "Unknown error or no data returned.")); // UPDATED
            }
        } catch (error) {
            console.error("AI Process Error (network or critical):", error);
            window.TB.ui.Toast.showError("Error communicating with AI service."); // UPDATED
        }
    }

    async function undoAiActionInternal() {
        TB.ui.Modal.show({
            title: 'Undo last AI action?',
            content: 'Are you sure you want to undo the last AI action?',
            maxWidth: '24rem',
            buttons: [
              {
                text: 'Cancel',
                variant: 'outline',
                action(modal) {
                  modal.close();
                },
              },
              {
                text: 'Undo',
                variant: 'danger', // or 'primary' if not destructive
                className: 'ml-2',
                async action(modal) {
                  try {
                    const undoResponse = await window.TB.api.request(
                      'POA',
                      'undo-ai-action',
                      null,
                      'POST'
                    );

                    if (undoResponse.error === window.TB.ToolBoxError.none) {
                      TB.ui.Toast.showSuccess(
                        undoResponse.info?.help_text || 'AI action undone.'
                      );

                      refreshItemsHierarchy();
                      updateSuggestions();
                      refreshHistory();

                      const currentItemResponse = await window.TB.api.request(
                        'POA',
                        'get-current-item',
                        null,
                        'GET'
                      );

                      if (
                        currentItemResponse.error === window.TB.ToolBoxError.none
                      ) {
                        currentItem = currentItemResponse.get();
                      } else {
                        console.error(
                          'Failed to refresh current item after undo:',
                          currentItemResponse.info.help_text
                        );
                        currentItem = null;
                      }

                      updateCurrentItemDisplay();
                    } else {
                      TB.ui.Toast.showError(
                        'Undo AI action failed: ' +
                          (undoResponse.info?.help_text || 'Unknown error.')
                      );
                    }
                  } catch (error) {
                    console.error('Undo AI Error (network or critical):', error);
                    TB.ui.Toast.showError(
                      'An error occurred while undoing the AI action. Please try again.'
                    );
                  } finally {
                    modal.close();
                  }
                },
              },
            ],
          });
    }

    // Expose functions to global scope for HTML onclick/onsubmit handlers
    window.switchTab = switchTabInternal;
    window.submitAiText = submitAiTextInternal;
    window.undoAiAction = undoAiActionInternal;
    window.openNewItemModal = openNewItemModalInternal;
    window.completeCurrentItem = completeCurrentItemInternal;
    window.startSuggestion = startSuggestionInternal;
    window.closeNewItemModal = closeNewItemModalInternal;
    window.createNewItem = createNewItemInternal;
    window.toggleTaskFields = toggleTaskFieldsInternal;
    window.removeItem = removeItemInternal;
    window.startItem = startItemInternal;


    // Initialize
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

} // End of if (window.history.state)
</script></div>
"""
  # Assuming you put the HTML in a separate file


    full_html = template_html_content
    return Result.html(app_instance.web_context() + full_html)


if __name__ == "__main__":
    # This block is for local testing of ActionManagerEnhanced if needed,
    # but full functionality requires the toolboxv2 app environment.
    # To run this, you'd need to mock App, DB, and ISAA.

    print(f"{Name} module structure defined. Full testing requires toolboxv2 environment.")

    # Example: How to create an ActionItem
    # task_example = ActionItem(title="Test Task", item_type=ItemType.TASK, priority=1)
    # note_example = ActionItem(title="Test Note", item_type=ItemType.NOTE, description="This is a test note.")
    # print(task_example.model_dump_json_safe())
    # print(note_example.model_dump_json_safe())
