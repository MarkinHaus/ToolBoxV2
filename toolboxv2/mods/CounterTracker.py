# --- START OF FILE counter_tracker_api.py ---
import asyncio
import contextlib
import json
import uuid
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Any

import pytz
from dateutil.parser import isoparse
from dateutil.relativedelta import relativedelta  # For more accurate month/year additions
from pydantic import BaseModel, Field, field_validator, model_validator

from toolboxv2 import App, RequestData, Result, get_app
from toolboxv2.utils.extras.base_widget import get_user_from_request

# --- Constants ---
MODULE_NAME= Name = "CounterTracker"
VERSION = "1.0.0"
DB_COUNTERS_PREFIX = f"{MODULE_NAME.lower()}_counters"
DB_ENTRIES_PREFIX = f"{MODULE_NAME.lower()}_entries"
DB_SETTINGS_PREFIX = f"{MODULE_NAME.lower()}_settings"

export = get_app(f"{MODULE_NAME}.Export").tb


# --- Enums ---
class CounterFrequency(str, Enum):
    ONE_TIME = "one_time"  # Goal to reach by a deadline (target_deadline)
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    # YEARLY = "yearly" # Can be added if needed


class CounterStatus(str, Enum):  # Status of the current period for a repeating counter
    PENDING_START = "pending_start"  # For future-dated counters
    IN_PROGRESS = "in_progress"
    TARGET_MET = "target_met"
    TARGET_EXCEEDED = "target_exceeded"  # If current_count > target_count
    PERIOD_ENDED_MET = "period_ended_met"  # Historical status for a period
    PERIOD_ENDED_UNMET = "period_ended_unmet"  # Historical status for a period
    INACTIVE = "inactive"  # User-deactivated or series ended


# --- Pydantic Models ---
class CounterSettings(BaseModel):
    user_id: str
    timezone: str = "UTC"
    enable_poa_integration: bool = False
    poa_reminder_lead_time_hours: int = Field(default=2, ge=0,
                                              le=24 * 7)  # How many hours before period end to create POA task
    poa_default_priority: int = Field(default=3, ge=1, le=5)  # POA priority for created tasks

    @field_validator('timezone')
    def validate_timezone(cls, v):
        if v not in pytz.all_timezones_set:
            raise ValueError(f"Invalid timezone: {v}")
        return v

    def model_dump_json_safe(self):
        return self.model_dump(mode="json")

    @classmethod
    def model_validate_json_safe(cls, json_data: str | dict[str, Any]):
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        return cls.model_validate(json_data)


class CountEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    counter_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(pytz.utc))
    amount: int = Field(default=1, ge=1)
    notes: str | None = None

    def model_dump_json_safe(self):
        return self.model_dump(mode="json")

    @classmethod
    def model_validate_json_safe(cls, json_data: str | dict[str, Any]):
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        return cls.model_validate(json_data)


class CounterItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str | None = None
    unit_name: str = Field(default="times")  # e.g., "glasses", "km", "pages"

    target_count: int = Field(default=1, ge=1)
    frequency: CounterFrequency = CounterFrequency.DAILY

    # For ONE_TIME: this is the deadline. For repeating: this is when the series of repetitions ends.
    series_end_date: date | None = None

    is_active: bool = True  # User can pause/archive
    current_count_in_period: int = 0
    total_accumulated_count: int = 0  # Overall count, ignoring resets

    current_period_start_utc: datetime | None = None
    current_period_end_utc: datetime | None = None
    # last_period_status: Optional[CounterStatus] = None # For stats/streaks, or derived from entries

    created_at: datetime = Field(default_factory=lambda: datetime.now(pytz.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(pytz.utc))

    # Tracks when the counter itself was last meaningfully reset (e.g., period advanced)
    # Not just any update.
    last_period_advanced_at_utc: datetime | None = None

    @model_validator(mode='before')
    def _handle_datetime_and_date_str_input(cls, values: dict[str, Any]) -> dict[str, Any]:
        datetime_fields = ['current_period_start_utc', 'current_period_end_utc', 'created_at', 'updated_at',
                           'last_period_advanced_at_utc']
        date_fields = ['series_end_date']
        user_timezone_str = values.get('_user_timezone_str_context', 'UTC')  # Context from manager

        for field_name in datetime_fields:
            value = values.get(field_name)
            if isinstance(value, str):
                try:
                    dt_val = isoparse(value)
                    if dt_val.tzinfo is None:  # Naive datetime string implies user's local timezone
                        user_tz = pytz.timezone(user_timezone_str)
                        values[field_name] = user_tz.localize(dt_val).astimezone(pytz.utc)
                    else:  # Aware datetime string
                        values[field_name] = dt_val.astimezone(pytz.utc)
                except ValueError:
                    pass  # Let Pydantic handle main validation
            elif isinstance(value, datetime):  # Already a datetime object
                if value.tzinfo is None:  # Naive datetime object means UTC if from internal, user local if from raw input
                    # This case is tricky; assuming internal datetimes are UTC, user inputs are handled by string parsing above
                    user_tz = pytz.timezone(user_timezone_str)
                    values[field_name] = user_tz.localize(value).astimezone(pytz.utc)
                elif value.tzinfo != pytz.utc:
                    values[field_name] = value.astimezone(pytz.utc)

        for field_name in date_fields:
            value = values.get(field_name)
            if isinstance(value, str):
                with contextlib.suppress(ValueError):
                    values[field_name] = isoparse(value).date()

        if '_user_timezone_str_context' in values:
            del values['_user_timezone_str_context']

        return values

    def model_dump_json_safe(self):
        return self.model_dump(mode="json")

    @classmethod
    def model_validate_json_safe(cls, json_data: str | dict[str, Any], user_timezone_str: str = "UTC"):
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        json_data['_user_timezone_str_context'] = user_timezone_str
        return cls.model_validate(json_data)

    def get_status(self) -> CounterStatus:
        if not self.is_active:
            return CounterStatus.INACTIVE

        now_utc = datetime.now(pytz.utc)
        if self.current_period_start_utc and now_utc < self.current_period_start_utc:
            return CounterStatus.PENDING_START  # Current period hasn't started yet

        # Check if the entire series has ended
        if self.series_end_date:
            # Convert series_end_date to a datetime at the end of that day in UTC for comparison
            # Assuming user_timezone_str context is available if needed, but series_end_date is just a date.
            # For comparison, consider the end of that day in UTC.
            # A simpler approach: if today's date (UTC) is past series_end_date, it's ended.
            if date.today() > self.series_end_date:  # Using current system date, ideally in user's TZ for this logic
                return CounterStatus.INACTIVE  # Series ended

        if self.current_count_in_period >= self.target_count:
            return CounterStatus.TARGET_MET  # Could add TARGET_EXCEEDED if current > target
        return CounterStatus.IN_PROGRESS


# --- Counter Manager ---
class CounterManager:
    def __init__(self, app: App, user_id: str):
        self.app = app
        self.user_id = user_id
        self.db = app.get_mod("DB")
        self.settings: CounterSettings = self._load_settings()
        self.counters: list[CounterItem] = self._load_counters()
        self.entries: list[CountEntry] = self._load_entries()  # Load all, filter as needed

    def _get_db_key(self, prefix: str) -> str:
        return f"{prefix}_{self.user_id}"

    def _load_settings(self) -> CounterSettings:
        key = self._get_db_key(DB_SETTINGS_PREFIX)
        data = self.db.get(key)
        if data.is_data() and data.get():
            try:
                return CounterSettings.model_validate_json_safe(
                    data.get()[0] if isinstance(data.get(), list) else data.get())
            except Exception as e:
                self.app.logger.error(f"Error loading CounterSettings for {self.user_id}: {e}. Using defaults.")
        return CounterSettings(user_id=self.user_id)

    def _save_settings(self):
        self.db.set(self._get_db_key(DB_SETTINGS_PREFIX), self.settings.model_dump_json_safe())

    def update_settings(self, settings_data: dict[str, Any]) -> CounterSettings:
        # Ensure user_id isn't changed
        updated_data = {**self.settings.model_dump(), **settings_data, "user_id": self.user_id}
        self.settings = CounterSettings.model_validate(updated_data)
        self._save_settings()
        # Re-evaluate counter periods if timezone changed, as periods are TZ-dependent
        for counter in self.counters:
            self._advance_period_if_due(counter, force_recalculate=True)
        self._save_counters()
        return self.settings

    def _load_counters(self) -> list[CounterItem]:
        key = self._get_db_key(DB_COUNTERS_PREFIX)
        data = self.db.get(key)
        loaded_counters = []
        if data.is_data() and data.get():
            try:
                raw_list = json.loads(data.get()[0] if isinstance(data.get(), list) else data.get())
                loaded_counters = [CounterItem.model_validate_json_safe(c, user_timezone_str=self.settings.timezone) for
                                   c in raw_list]
            except Exception as e:
                self.app.logger.error(f"Error loading Counters for {self.user_id}: {e}")

        # Ensure all loaded counters have their periods checked/initialized
        for counter in loaded_counters:
            if counter.is_active:  # only if it's an active counter
                self._advance_period_if_due(counter, force_recalculate=(counter.current_period_start_utc is None))
        return loaded_counters

    def _save_counters(self):
        self.db.set(self._get_db_key(DB_COUNTERS_PREFIX), json.dumps([c.model_dump_json_safe() for c in self.counters]))

    def _load_entries(self) -> list[CountEntry]:
        key = self._get_db_key(DB_ENTRIES_PREFIX)
        data = self.db.get(key)
        if data.is_data() and data.get():
            try:
                raw_list = json.loads(data.get()[0] if isinstance(data.get(), list) else data.get())
                return [CountEntry.model_validate_json_safe(e) for e in raw_list]
            except Exception as e:
                self.app.logger.error(f"Error loading CountEntries for {self.user_id}: {e}")
        return []

    def _save_entries(self):
        # Limit entries stored to, e.g., last 1000 per user or by time, to prevent unbounded growth.
        # For simplicity, this example saves all. Production would need pruning.
        MAX_ENTRIES = 1000  # Example limit
        if len(self.entries) > MAX_ENTRIES:
            self.entries = sorted(self.entries, key=lambda x: x.timestamp, reverse=True)[:MAX_ENTRIES]
        self.db.set(self._get_db_key(DB_ENTRIES_PREFIX), json.dumps([e.model_dump_json_safe() for e in self.entries]))

    def get_user_timezone(self) -> pytz.BaseTzInfo:
        try:
            return pytz.timezone(self.settings.timezone)
        except pytz.UnknownTimeZoneError:
            return pytz.utc

    def _calculate_current_period_boundaries_utc(self, counter: CounterItem, ref_datetime_utc: datetime) -> tuple[
        datetime | None, datetime | None]:
        user_tz = self.get_user_timezone()
        ref_datetime_user = ref_datetime_utc.astimezone(user_tz)

        start_user, end_user = None, None

        if counter.frequency == CounterFrequency.DAILY:
            start_user = datetime.combine(ref_datetime_user.date(), time.min, tzinfo=user_tz)
            end_user = datetime.combine(ref_datetime_user.date(), time.max, tzinfo=user_tz)
        elif counter.frequency == CounterFrequency.WEEKLY:
            # Week starts on Monday (isoweekday() == 1)
            start_of_week_user = ref_datetime_user - timedelta(days=ref_datetime_user.isoweekday() - 1)
            start_user = datetime.combine(start_of_week_user.date(), time.min, tzinfo=user_tz)
            end_of_week_user = start_of_week_user + timedelta(days=6)
            end_user = datetime.combine(end_of_week_user.date(), time.max, tzinfo=user_tz)
        elif counter.frequency == CounterFrequency.MONTHLY:
            start_user = datetime.combine(ref_datetime_user.date().replace(day=1), time.min, tzinfo=user_tz)
            next_month_start_user = (start_user + relativedelta(months=1))
            end_user = next_month_start_user - timedelta(microseconds=1)
        elif counter.frequency == CounterFrequency.ONE_TIME:
            # For ONE_TIME, the "period" is from creation until deadline or indefinitely if no deadline
            start_user = counter.created_at.astimezone(user_tz)  # Period starts at creation
            if counter.series_end_date:  # This is the deadline for ONE_TIME
                end_user = datetime.combine(counter.series_end_date, time.max, tzinfo=user_tz)
            else:  # No deadline, period is effectively infinite or until manually marked inactive
                end_user = None
        else:
            return None, None

        start_utc = start_user.astimezone(pytz.utc) if start_user else None
        end_utc = end_user.astimezone(pytz.utc) if end_user else None
        return start_utc, end_utc

    def _advance_period_if_due(self, counter: CounterItem, force_recalculate: bool = False):
        if not counter.is_active:
            return

        now_utc = datetime.now(pytz.utc)

        # Initial setup or forced recalculation
        if force_recalculate or not counter.current_period_start_utc or not counter.current_period_end_utc:
            new_start_utc, new_end_utc = self._calculate_current_period_boundaries_utc(counter, now_utc)
            if new_start_utc and new_end_utc:  # For repeating counters
                if counter.current_period_start_utc != new_start_utc:  # If period genuinely changes or initializes
                    counter.current_count_in_period = 0  # Reset count for new period
                    counter.last_period_advanced_at_utc = now_utc
                counter.current_period_start_utc = new_start_utc
                counter.current_period_end_utc = new_end_utc
            elif counter.frequency == CounterFrequency.ONE_TIME:  # For one-time counters
                # Use creation as start, deadline as end. No "advancement" in the same way.
                new_start_utc, new_end_utc = self._calculate_current_period_boundaries_utc(counter, counter.created_at)
                counter.current_period_start_utc = new_start_utc
                counter.current_period_end_utc = new_end_utc  # Can be None if no deadline
            counter.updated_at = now_utc
            return

        # Check if current period has ended for repeating counters
        if counter.frequency != CounterFrequency.ONE_TIME and counter.current_period_end_utc and now_utc > counter.current_period_end_utc:
            # Period ended. Log previous period status (simplified, could be more detailed in stats)
            # counter.last_period_status = CounterStatus.PERIOD_ENDED_MET if counter.current_count_in_period >= counter.target_count else CounterStatus.PERIOD_ENDED_UNMET

            # Check if the series itself has ended
            if counter.series_end_date:
                # Convert series_end_date to datetime at end of day in user's timezone then UTC
                user_tz = self.get_user_timezone()
                series_end_dt_user_tz = datetime.combine(counter.series_end_date, time.max, tzinfo=user_tz)
                series_end_dt_utc = series_end_dt_user_tz.astimezone(pytz.utc)
                if counter.current_period_end_utc >= series_end_dt_utc:  # Current period ending is at or after series end
                    counter.is_active = False  # Deactivate counter
                    counter.updated_at = now_utc
                    # No new period calculation needed
                    return

            # Calculate and set the new period
            new_start_utc, new_end_utc = self._calculate_current_period_boundaries_utc(counter, now_utc)
            counter.current_period_start_utc = new_start_utc
            counter.current_period_end_utc = new_end_utc
            counter.current_count_in_period = 0  # Reset for new period
            counter.last_period_advanced_at_utc = now_utc
            counter.updated_at = now_utc

    def create_counter(self, data: dict[str, Any]) -> CounterItem:
        # Pass user's timezone to model_validate_json_safe for context
        counter = CounterItem.model_validate_json_safe(data, user_timezone_str=self.settings.timezone)
        counter.updated_at = datetime.now(pytz.utc)  # Ensure fresh timestamp

        # Initialize period for new counter
        self._advance_period_if_due(counter, force_recalculate=True)

        self.counters.append(counter)
        self._save_counters()
        return counter

    def get_counter(self, counter_id: str) -> CounterItem | None:
        for counter in self.counters:
            if counter.id == counter_id:
                if counter.is_active: self._advance_period_if_due(counter)  # Check period before returning
                return counter
        return None

    def update_counter(self, counter_id: str, data: dict[str, Any]) -> CounterItem | None:
        idx = next((i for i, c in enumerate(self.counters) if c.id == counter_id), None)
        if idx is None:
            return None

        original_counter = self.counters[idx]

        # Preserve fields that shouldn't be easily overwritten by partial update
        # like created_at, total_accumulated_count (unless specifically handled).
        # For simplicity, model_validate will handle merging.
        # We need to be careful if frequency or target_count changes, may need to reset period logic.
        update_payload = {**original_counter.model_dump(), **data}
        update_payload['_user_timezone_str_context'] = self.settings.timezone  # For validation context

        updated_counter = CounterItem.model_validate(update_payload)
        updated_counter.id = original_counter.id  # Ensure ID isn't changed
        updated_counter.created_at = original_counter.created_at  # Preserve creation
        updated_counter.total_accumulated_count = original_counter.total_accumulated_count  # Preserve total unless specifically changed

        updated_counter.updated_at = datetime.now(pytz.utc)

        # If frequency, series_end_date or crucial timing params changed, recalculate period
        recalculate_period = False
        if (original_counter.frequency != updated_counter.frequency or
            original_counter.series_end_date != updated_counter.series_end_date or
            original_counter.target_count != updated_counter.target_count or  # Target change could affect status
            original_counter.is_active != updated_counter.is_active):  # Activation status change
            recalculate_period = True

        if not updated_counter.is_active:  # If being deactivated
            updated_counter.current_period_start_utc = None
            updated_counter.current_period_end_utc = None
            # No current_count_in_period reset, it holds the last active period's count
        elif recalculate_period or not updated_counter.current_period_start_utc:
            self._advance_period_if_due(updated_counter, force_recalculate=True)

        self.counters[idx] = updated_counter
        self._save_counters()
        return updated_counter

    def delete_counter(self, counter_id: str) -> bool:
        original_len = len(self.counters)
        self.counters = [c for c in self.counters if c.id != counter_id]
        # Also delete associated entries
        self.entries = [e for e in self.entries if e.counter_id != counter_id]
        if len(self.counters) < original_len:
            self._save_counters()
            self._save_entries()
            return True
        return False

    def increment_counter(self, counter_id: str, amount: int = 1, notes: str | None = None) -> CounterItem | None:
        counter = self.get_counter(counter_id)
        if not counter or not counter.is_active:
            return None

        self._advance_period_if_due(counter)  # Ensure period is current before incrementing

        # If period advanced, counter.current_count_in_period would be 0
        # If counter became inactive (series ended), advance_period would handle it
        if not counter.is_active:  # Check again after advance_period_if_due
            self.app.logger.info(f"Counter {counter_id} became inactive (series ended). Cannot increment.")
            self._save_counters()  # Save potential deactivation
            return counter  # Return its final state

        now_utc = datetime.now(pytz.utc)
        # For ONE_TIME counters, check deadline
        if counter.frequency == CounterFrequency.ONE_TIME and counter.current_period_end_utc and now_utc > counter.current_period_end_utc:
            self.app.logger.info(f"Cannot increment ONE_TIME counter {counter_id}: deadline has passed.")
            counter.is_active = False  # Mark as inactive due to missed deadline
            self._save_counters()
            return counter

        counter.current_count_in_period += amount
        counter.total_accumulated_count += amount
        counter.updated_at = now_utc

        entry = CountEntry(counter_id=counter.id, amount=amount, notes=notes, timestamp=now_utc)
        self.entries.append(entry)

        self._save_counters()
        self._save_entries()

        if self.settings.enable_poa_integration:
            # Run POA check in background to not block API response
            asyncio.create_task(self._check_and_create_poa_reminder(counter))

        return counter

    def get_all_counters(self) -> list[CounterItem]:
        now_utc = datetime.now(pytz.utc)
        # Refresh periods for all active counters before returning list
        for counter in self.counters:
            if counter.is_active:
                self._advance_period_if_due(counter)
        self._save_counters()  # Save any changes from period advancements
        return sorted([c for c in self.counters if c.is_active or c.get_status() != CounterStatus.INACTIVE],
                      key=lambda x: (not x.is_active, x.name.lower()))  # Active first, then by name

    def get_counter_stats(self, counter_id: str) -> dict[str, Any] | None:
        counter = self.get_counter(counter_id)
        if not counter: return None

        stats = {
            "counter_info": counter.model_dump_json_safe(),
            "current_status": counter.get_status().value,
            "current_period_progress_percent": 0,
            "historical_period_summary": [],  # List of {period_start, period_end, count, target, met_target}
            "total_entries_logged": 0,
            "longest_streak_met_periods": 0,  # Days/weeks/months target was met consecutively
            "all_time_average_per_period": 0,
        }

        if counter.target_count > 0 and counter.current_period_start_utc:  # Avoid division by zero
            stats["current_period_progress_percent"] = round(
                (counter.current_count_in_period / counter.target_count) * 100, 1
            )

        counter_entries = sorted([e for e in self.entries if e.counter_id == counter_id], key=lambda x: x.timestamp)
        stats["total_entries_logged"] = sum(e.amount for e in counter_entries)

        # More complex stats (streaks, historical summary) require iterating through periods
        # This is a simplified version. A full implementation would:
        # 1. Determine all past periods from counter.created_at and frequency.
        # 2. For each period, sum entries within that period.
        # 3. Calculate streak and averages.
        # For now, these are placeholders or simple calculations.

        # Example: last 5 completed periods (simplified)
        # This requires robustly defining "period completion"
        # A true historical summary is non-trivial.
        # Here's a placeholder for the idea:
        # user_tz = self.get_user_timezone()
        # temp_period_end = counter.current_period_start_utc # Start from current period's start
        # for _ in range(5): # Look back 5 periods
        #     if not temp_period_end: break
        #     # Calculate previous period based on frequency
        #     # ... logic to find temp_period_start and sum entries ...
        #     # This is complex; omitting full logic for brevity

        return stats

    async def _check_and_create_poa_reminder(self, counter: CounterItem):
        if not self.settings.enable_poa_integration: return
        if not counter.is_active or counter.get_status() == CounterStatus.TARGET_MET: return

        # Avoid creating reminder if period is almost over or too far in future
        now_utc = datetime.now(pytz.utc)
        if not counter.current_period_end_utc: return  # Should not happen for active repeating counters

        time_to_period_end = counter.current_period_end_utc - now_utc
        reminder_lead_time = timedelta(hours=self.settings.poa_reminder_lead_time_hours)

        if timedelta(minutes=15) < time_to_period_end <= reminder_lead_time:  # Create reminder if within window
            # Only if target is not met
            if counter.current_count_in_period < counter.target_count:
                try:
                    poa_app = self.app.get_mod("POA")
                    if not poa_app:
                        self.app.logger.warning("POA module not found for CounterTracker integration.")
                        return

                    # Need a way to get POA manager for the current user_id
                    # Assuming POA's get_manager can be called like this:
                    # This is a synchronous call from an async task. POA's get_manager itself must be safe.
                    # Or, POA module should provide an async way to get manager or add tasks.
                    # For this example, assuming POA module has a globally accessible method or
                    # we can instantiate its manager if needed. Let's try to get the manager.

                    # HACK: To call async get_manager from sync context if needed, or refactor
                    # This part is tricky. For now, let's assume we can get POA manager.
                    # A robust solution might involve an async queue or a dedicated POA interaction service.
                    # The 'request' object is not available here. We only have user_id.
                    # This means get_manager in POA needs to be adaptable.
                    # Let's assume POA.get_manager(app, user_id_only_param) exists or can be made.
                    # For now, we can't directly call `await poa_app.get_manager(self.app, request)`
                    # This is a placeholder for actual POA interaction which needs careful design
                    # due to potential async/sync context issues and lack of 'request'.

                    # SIMPLIFIED: Let's assume a direct add method on poa_app module for demo
                    # if hasattr(poa_app, "direct_add_task_for_user"):
                    #    await poa_app.direct_add_task_for_user(self.user_id, task_data)

                    self.app.logger.info(f"POA Integration: Would create task for counter {counter.id} "
                                         f"for user {self.user_id} (details omitted).")

                    # Construct task data
                    task_title = f"[Counter] {counter.name} ({counter.current_count_in_period}/{counter.target_count} {counter.unit_name})"
                    task_description = (
                        f"Reminder from CounterTracker for counter: {counter.name}.\n"
                        f"Current progress: {counter.current_count_in_period}/{counter.target_count} {counter.unit_name}.\n"
                        f"Period ends: {counter.current_period_end_utc.astimezone(self.get_user_timezone()).strftime('%Y-%m-%d %H:%M')}.\n"
                        f"Counter ID: {counter.id}"
                    )
                    # POA task due time: e.g., 30 mins before period end, or now if closer
                    task_due_utc = counter.current_period_end_utc - timedelta(minutes=30)
                    if task_due_utc < now_utc + timedelta(minutes=5): task_due_utc = now_utc + timedelta(minutes=10)

                    # This is the part that needs a proper POA API callable without a full 'RequestData'
                    # For now, this is illustrative of the intent.
                    # One way is for POA to expose a utility function:
                    # async def create_poa_task_for_user(app: App, user_id: str, task_data: dict) -> Optional[ActionItem]:
                    #    manager = POA_ActionManager(app, user_id) # or however it gets instance
                    #    return manager.add_item(task_data)
                    #
                    # Then CounterManager can call:
                    # await poa_app_module.create_poa_task_for_user(self.app, self.user_id, poa_task_payload)

                    # Placeholder for actual POA call:
                    # poa_task_payload = {
                    #     "title": task_title, "description": task_description,
                    #     "fixed_time": task_due_utc.isoformat(), # POA expects UTC ISO
                    #     "priority": self.settings.poa_default_priority,
                    #     "item_type": "task" # Assuming POA.ItemType.TASK.value
                    # }
                    # self.app.logger.info(f"POA: Creating task: {poa_task_payload}")
                    # poa_item = await poa_app_module.some_async_add_item_function(self.app, self.user_id, poa_task_payload)
                    # if poa_item:
                    #    self.app.logger.info(f"POA task {poa_item.id} created for counter {counter.id}")
                    # else:
                    #    self.app.logger.warning(f"Failed to create POA task for counter {counter.id}")


                except Exception as e:
                    self.app.logger.error(f"Error in POA integration for counter {counter.id}: {e}", exc_info=True)


# --- Manager Cache & Getter ---
_managers: dict[str, CounterManager] = {}


async def get_manager(app: App, request: RequestData) -> CounterManager:
    user = await get_user_from_request(app, request)
    if request is None:
        app.logger.warning("No request provided to get CounterManager. Using default user ID.")
        user_id = "default_public_user"
    else:
        user_id = user.uid if user and user.uid else f"guest_ct_{request.session_id[:8]}"  # Guest ID specific to CounterTracker
    if user_id not in _managers:
        _managers[user_id] = CounterManager(app, user_id)
    return _managers[user_id]


# --- API Endpoints ---
@export(mod_name=MODULE_NAME, name="init_config", initial=True)
def init_counter_tracker_module(app: App):
    if app is None:
        app = get_app()
    app.run_any(("CloudM", "add_ui"), name=MODULE_NAME, title="Counter Tracker",
                path=f"/api/{MODULE_NAME}/ui", description="Track your recurring and one-time goals.")
    app.logger.info(f"{MODULE_NAME} module (v{VERSION}) initialized.")


@export(mod_name=MODULE_NAME, name="get-settings", api=True, request_as_kwarg=True)
async def api_get_settings(app: App, request: RequestData):
    manager = await get_manager(app, request)
    return Result.json(data=manager.settings.model_dump_json_safe())


@export(mod_name=MODULE_NAME, name="update-settings", api=True, request_as_kwarg=True, api_methods=['POST'])
async def api_update_settings(app: App, request: RequestData, data: dict[str, Any]):
    manager = await get_manager(app, request)
    try:
        updated_settings = manager.update_settings(data)
        return Result.json(data=updated_settings.model_dump_json_safe(), info="Settings updated.")
    except ValueError as ve:
        return Result.default_user_error(str(ve), 400)
    except Exception as e:
        app.logger.error(f"Error updating counter settings: {e}", exc_info=True)
        return Result.default_internal_error("Could not update settings.")


@export(mod_name=MODULE_NAME, name="create-counter", api=True, request_as_kwarg=True, api_methods=['POST'])
async def api_create_counter(app: App, request: RequestData, data: dict[str, Any]):
    manager = await get_manager(app, request)
    try:
        counter = manager.create_counter(data)
        return Result.json(data=counter.model_dump_json_safe())
    except ValueError as ve:  # Pydantic validation error
        return Result.default_user_error(f"Invalid counter data: {str(ve)}", 400)
    except Exception as e:
        app.logger.error(f"Error creating counter: {e}", exc_info=True)
        return Result.default_internal_error("Could not create counter.")


@export(mod_name=MODULE_NAME, name="list-counters", api=True, request_as_kwarg=True)
async def api_list_counters(app: App, request: RequestData):
    manager = await get_manager(app, request)
    counters = manager.get_all_counters()
    return Result.json(data=[c.model_dump_json_safe() for c in counters])


@export(mod_name=MODULE_NAME, name="get-counter", api=True,
        request_as_kwarg=True)  # Path: /api/CounterTracker/get-counter/{counter_id}
async def api_get_counter(app: App, request: RequestData, counter_id: str):
    manager = await get_manager(app, request)
    counter = manager.get_counter(counter_id)
    if counter:
        return Result.json(data=counter.model_dump_json_safe())
    return Result.default_user_error("Counter not found.", 404)


@export(mod_name=MODULE_NAME, name="update-counter", api=True, request_as_kwarg=True,
        api_methods=['POST'])  # Path: /api/CounterTracker/update-counter/{counter_id}
async def api_update_counter(app: App, request: RequestData, counter_id: str, data: dict[str, Any]):
    manager = await get_manager(app, request)
    try:
        counter = manager.update_counter(counter_id, data)
        if counter:
            return Result.json(data=counter.model_dump_json_safe())
        return Result.default_user_error("Counter not found.", 404)
    except ValueError as ve:
        return Result.default_user_error(f"Invalid counter data for update: {str(ve)}", 400)
    except Exception as e:
        app.logger.error(f"Error updating counter {counter_id}: {e}", exc_info=True)
        return Result.default_internal_error("Could not update counter.")


@export(mod_name=MODULE_NAME, name="delete-counter", api=True, request_as_kwarg=True,
        api_methods=['POST'])  # Path: /api/CounterTracker/delete-counter/{counter_id}
async def api_delete_counter(app: App, request: RequestData, counter_id: str,
                             data: dict | None = None):  # data can be None if ID is from path
    manager = await get_manager(app, request)
    if manager.delete_counter(counter_id):
        return Result.ok("Counter deleted.")
    return Result.default_user_error("Counter not found or delete failed.", 404)


@export(mod_name=MODULE_NAME, name="increment-counter", api=True, request_as_kwarg=True,
        api_methods=['POST'])  # Path: /api/CounterTracker/increment-counter/{counter_id}
async def api_increment_counter(app: App, request: RequestData, counter_id: str, data: dict[str, Any] | None = None):
    manager = await get_manager(app, request)
    amount = data.get("amount", 1) if data else 1
    notes = data.get("notes") if data else None
    try:
        amount = int(amount)
        if amount < 1: raise ValueError("Amount must be positive.")
    except ValueError:
        return Result.default_user_error("Invalid amount.", 400)

    counter = manager.increment_counter(counter_id, amount, notes)
    if counter:
        return Result.json(data=counter.model_dump_json_safe())
    return Result.default_user_error("Counter not found, inactive, or increment failed.", 404)


@export(mod_name=MODULE_NAME, name="get-counter-stats", api=True,
        request_as_kwarg=True)  # Path: /api/CounterTracker/get-counter-stats/{counter_id}
async def api_get_counter_stats(app: App, request: RequestData, counter_id: str):
    manager = await get_manager(app, request)
    stats = manager.get_counter_stats(counter_id)
    if stats:
        return Result.json(data=stats)
    return Result.default_user_error("Counter not found or stats unavailable.", 404)


@export(mod_name=MODULE_NAME, name="get-counter-entries", api=True,
        request_as_kwarg=True)  # Path: /api/CounterTracker/get-counter-entries/{counter_id}
async def api_get_counter_entries(app: App, request: RequestData, counter_id: str, limit: int | None = 50):
    manager = await get_manager(app, request)
    if limit is None:
        limit = 50
    try:
        limit = int(limit)
    except (ValueError, TypeError):
        limit = 50

    counter = manager.get_counter(counter_id)  # Ensures counter exists
    if not counter:
        return Result.default_user_error("Counter not found.", 404)

    entries = sorted([e for e in manager.entries if e.counter_id == counter_id], key=lambda x: x.timestamp,
                     reverse=True)
    return Result.json(data=[e.model_dump_json_safe() for e in entries[:limit]])


# --- UI Endpoint ---
@get_app().tb(mod_name=MODULE_NAME, version=VERSION, level=0, api=True, name="ui", state=False)
def counter_tracker_ui_page(app_ref: App | None = None):
    app_instance = app_ref if app_ref else get_app(MODULE_NAME)
    html_content = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Counter Tracker</title>
    <style>
    """+"""/* Basic styles - will be expanded */
body {
  font-family: var(--font-family-base);
  margin: 0;
  background-color: var(--theme-bg, #f8f9fa);
  color: var(--theme-text, #181823);
}

.container {
  max-width: 900px;
  margin: 20px auto;
  padding: 20px;
  background-color: var(--theme-bg, #ffffff);
  border-radius: var(--radius-md);
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

h1, h2 {
  color: var(--theme-header-color, var(--dark-primary-color, #2c3e50));
}

.hidden {
  display: none !important;
}

.counter-card {
  border: 1px solid var(--theme-border, #e5e7eb);
  padding: 15px;
  margin-bottom: 15px;
  border-radius: var(--radius-sm);
  background-color: var(--theme-bg, #fdfdfd);
}

.counter-card h3 {
  margin-top: 0;
}

.counter-card p {
  margin: 5px 0;
}

.progress-bar-container {
  background-color: #e0e0e0;
  border-radius: var(--radius-sm);
  height: 20px;
  margin: 10px 0;
}

.progress-bar {
  background-color: var(--color-success, #198754);
  height: 100%;
  border-radius: var(--radius-sm);
  text-align: center;
  color: var(--anti-text-clor, white);
  line-height: 20px;
  transition: width var(--transition-medium);
}

.btn {
  padding: 8px 12px;
  margin-right: 5px;
  border: none;
  border-radius: var(--radius-sm);
  cursor: pointer;
}

.btn-primary {
  background-color: var(--button-bg, var(--theme-primary));
  color: var(--button-text, white);
}

.btn-secondary {
  background-color: var(--theme-secondary, #537FE7);
  color: var(--theme-secondary-text-color, #2c3e50);
}

label {
  display: block;
  margin-top: 10px;
}

input[type="text"],
input[type="number"],
input[type="date"],
select,
textarea {
  width: calc(100% - 16px);
  padding: 8px;
  margin-top: 5px;
  border: 1px solid var(--input-border, #ccc);
  border-radius: var(--radius-sm);
  background-color: var(--input-bg, #fff);
  color: var(--theme-input-text-color, var(--theme-text));
}

.tabs button {
  padding: 10px 15px;
  border: none;
  background-color: var(--theme-tab-bg-color, #eee);
  cursor: pointer;
  color: var(--theme-tab-text-color, #333);
}

.tabs button.active {
  background-color: var(--theme-tab-active-bg-color, #ccc);
  font-weight: bold;
}

.tab-content {
  padding: 15px;
  border: 1px solid var(--theme-border, #ccc);
  border-top: none;
}

#overallStatsViewPlot,
#statsViewPlot {
  min-height: 300px;
  width: 100%;
}

.compact-counter-group {
  margin-bottom: 10px;
  padding: 10px;
  border: 1px solid var(--theme-border, #eee);
  border-radius: var(--radius-sm);
}

.compact-counter-group summary {
  font-weight: bold;
  cursor: pointer;
}

.compact-counter-group .counter-card {
  margin-left: 20px;
  margin-top: 10px;
}

/* Ensure TB Modals are styled nicely if default isn't enough */
/* .tb-modal-content {} */
"""+f"""
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3.3.0/build/global/luxon.min.js"></script>
</head>
<body>
    <div class="container">
        <header style="display: flex; justify-content: space-between; align-items: center;">
            <h1>{MODULE_NAME}</h1>
        </header>

        <div class="tabs">
            <button class="tab active" data-tab="dashboard">Dashboard</button>
            <button class="tab" data-tab="overall-stats">Overall Stats</button>
            <button class="tab" data-tab="settings">Settings</button>
        </div>

        <div id="dashboardTab" class="tab-content">
            <h2>My Counters</h2>
            <button id="showAddCounterModalBtn" class="btn btn-primary">Add New Counter</button>
            <div id="countersList"></div>
        </div>

        <div id="overallStatsTab" class="tab-content hidden">
            <h2>Overall Statistics</h2>
            <p>Summary of all your tracking efforts.</p>
            <div id="overallStatsSummary"></div>
            <div><canvas id="overallStatsViewPlot"></canvas></div>
            <div id="topCountersList"><h3>Most Active Counters:</h3><ul></ul></div>
        </div>

        <div id="settingsTab" class="tab-content hidden">
            <h2>Settings</h2>
            <form id="settingsForm">
                <div>
                    <label for="settingTimezone">Timezone:</label>
                    <select id="settingTimezone"></select>
                </div>
                <div>
                    <label for="settingEnablePoa">Enable POA Integration:</label>
                    <input type="checkbox" id="settingEnablePoa">
                </div>
                <div id="poaSpecificSettings" class="hidden">
                    <label for="settingPoaLeadTime">POA Reminder Lead Time (hours):</label>
                    <input type="number" id="settingPoaLeadTime" min="0" step="1">
                    <label for="settingPoaPriority">POA Default Task Priority (1-5):</label>
                    <input type="number" id="settingPoaPriority" min="1" max="5" step="1">
                </div>
                <button type="submit" class="btn btn-primary">Save Settings</button>
            </form>
        </div>
    </div>

    """ + """<script unsave="true" defer>
    function init () {
    setTimeout(_init, 1000);
    }
    function _init () {
        "use strict";

        const API_MODULE_NAME = "CounterTracker";
        let currentSettings = { timezone: 'UTC', enable_poa_integration: false };
        let currentCounters = [];
        let statsChartInstance = null;
        let overallStatsChartInstance = null;
        const DateTime = window.luxon.DateTime;

        const elements = {
            // Tabs
            tabsContainer: document.querySelector('.tabs'),
            dashboardTabContent: document.getElementById('dashboardTab'),
            overallStatsTabContent: document.getElementById('overallStatsTab'),
            settingsTabContent: document.getElementById('settingsTab'),
            // Dashboard
            showAddCounterModalBtn: document.getElementById('showAddCounterModalBtn'),
            countersListDiv: document.getElementById('countersList'),
            // Overall Stats
            overallStatsSummaryDiv: document.getElementById('overallStatsSummary'),
            overallStatsViewPlotCanvas: document.getElementById('overallStatsViewPlot'),
            topCountersListUl: document.querySelector('#topCountersList ul'),
            // Settings
            settingsForm: document.getElementById('settingsForm'),
            settingTimezoneSelect: document.getElementById('settingTimezone'),
            settingEnablePoaCheckbox: document.getElementById('settingEnablePoa'),
            poaSpecificSettingsDiv: document.getElementById('poaSpecificSettings'),
            settingPoaLeadTimeInput: document.getElementById('settingPoaLeadTime'),
            settingPoaPriorityInput: document.getElementById('settingPoaPriority'),
        };

        async function initializeApp() {
            if (!window.TB?.api?.request || !window.TB?.ui?.Modal) {
                console.error("Toolbox API or TB.ui.Modal not available. CounterTracker cannot run.");
                document.body.innerHTML = "<p style='color:red; font-size:1.2em; padding:20px;'>Error: Toolbox API/UI not available.</p>";
                return;
            }
            setupEventListeners();
            await loadSettings();
            await loadCounters(); // Initial load for dashboard
            showTab('dashboard');
            populateTimezoneSelect();
        }

        function setupEventListeners() {
            elements.tabsContainer.addEventListener('click', (e) => {
                if (e.target.classList.contains('tab')) {
                    showTab(e.target.dataset.tab);
                }
            });
            elements.showAddCounterModalBtn.addEventListener('click', () => openCounterModal());
            elements.settingsForm.addEventListener('submit', handleSaveSettings);
            elements.settingEnablePoaCheckbox.addEventListener('change', togglePoaSpecificSettingsVisibility);
        }

        async function apiRequest(endpoint, payload = null, method = 'GET', queryParams = {}) {
            if(window.TB?.ui?.Loader) TB.ui.Loader.show({text: "Working...", hideMainContent:false});
            try {
                const response = await window.TB.api.request(API_MODULE_NAME, endpoint, payload, method, { queryParams });
                if (response.error !== window.TB.ToolBoxError.none) {
                    const errorMsg = response.info?.help_text || response.data?.message || `API Error (${response.error})`;
                    console.error(`API Error [${endpoint}]:`, errorMsg, response);
                    if(window.TB?.ui?.Toast) TB.ui.Toast.showError(errorMsg.substring(0,150), {duration: 4000});
                    return { error: true, message: errorMsg, data: response.get() };
                }
                return { error: false, data: response.get() };
            } catch (err) {
                console.error(`Network/JS Error [${endpoint}]:`, err);
                if(window.TB?.ui?.Toast) TB.ui.Toast.showError("Network or application error.", {duration: 4000});
                return { error: true, message: "NETWORK_ERROR" };
            } finally {
                if(window.TB?.ui?.Loader) TB.ui.Loader.hide();
            }
        }

        function formatDate(isoString, format = DateTime.DATETIME_MED_WITH_SECONDS) {
            if (!isoString) return 'N/A';
            try {
                return DateTime.fromISO(isoString, { zone: 'utc' }).setZone(currentSettings.timezone).toLocaleString(format);
            } catch(e) {
                console.warn("Error formatting date:", isoString, currentSettings.timezone, e);
                return new Date(isoString).toLocaleString(); // Fallback
            }
        }
        function formatJustDate(isoString) {
             if (!isoString) return 'N/A';
             try {
                 return DateTime.fromISO(isoString).toLocaleString(DateTime.DATE_MED);
             } catch (e) { return isoString; }
        }

        function showTab(tabName) {
            document.querySelectorAll('.tabs .tab').forEach(tab => {
                tab.classList.toggle('active', tab.dataset.tab === tabName);
            });
            document.querySelectorAll('.tab-content').forEach(tc => {
                tc.classList.toggle('hidden', tc.id !== `${tabName}Tab`);
            });

            if (tabName === 'dashboard') {
                loadCounters();
            } else if (tabName === 'overall-stats') {
                loadOverallStats();
            } else if (tabName === 'settings') {
                loadSettings();
            }
        }

        async function loadSettings() {
            const response = await apiRequest('get-settings');
            if (!response.error && response.data) {
                currentSettings = response.data;
                elements.settingTimezoneSelect.value = currentSettings.timezone;
                elements.settingEnablePoaCheckbox.checked = currentSettings.enable_poa_integration;
                elements.settingPoaLeadTimeInput.value = currentSettings.poa_reminder_lead_time_hours;
                elements.settingPoaPriorityInput.value = currentSettings.poa_default_priority;
                togglePoaSpecificSettingsVisibility();
            } else {
                console.error("Failed to load settings, using defaults.");
                populateTimezoneSelect();
            }
        }

        async function handleSaveSettings(event) {
            event.preventDefault();
            const newSettings = {
                timezone: elements.settingTimezoneSelect.value,
                enable_poa_integration: elements.settingEnablePoaCheckbox.checked,
                poa_reminder_lead_time_hours: parseInt(elements.settingPoaLeadTimeInput.value),
                poa_default_priority: parseInt(elements.settingPoaPriorityInput.value)
            };
            const response = await apiRequest('update-settings', newSettings, 'POST');
            if (!response.error) {
                currentSettings = response.data;
                if(window.TB?.ui?.Toast) TB.ui.Toast.showSuccess("Settings saved!");
                await loadCounters(); // Reload counters as timezone might affect period display
            }
        }

        function populateTimezoneSelect() {
            const commonTimezones = ["UTC", "GMT", "US/Pacific", "US/Mountain", "US/Central", "US/Eastern", "America/New_York", "America/Los_Angeles", "America/Chicago", "America/Denver", "Europe/London", "Europe/Berlin", "Europe/Paris", "Europe/Moscow", "Europe/Madrid", "Asia/Tokyo", "Asia/Shanghai", "Asia/Hong_Kong", "Asia/Dubai", "Asia/Kolkata", "Australia/Sydney", "Australia/Melbourne", "Pacific/Auckland"];
            if (currentSettings.timezone && !commonTimezones.includes(currentSettings.timezone)) {
                commonTimezones.unshift(currentSettings.timezone);
            }
            elements.settingTimezoneSelect.innerHTML = commonTimezones.map(tz => `<option value="${tz}">${tz}</option>`).join('');
            elements.settingTimezoneSelect.value = currentSettings.timezone;
        }

        function togglePoaSpecificSettingsVisibility() {
            elements.poaSpecificSettingsDiv.classList.toggle('hidden', !elements.settingEnablePoaCheckbox.checked);
        }

        async function loadCounters() {
            const response = await apiRequest('list-counters');
            if (!response.error && response.data) {
                currentCounters = response.data;
                renderCounters();
            } else {
                elements.countersListDiv.innerHTML = "<p>Could not load counters.</p>";
            }
        }

        function renderCounters() {
            if (currentCounters.length === 0) {
                elements.countersListDiv.innerHTML = "<p>No counters yet. Add one!</p>";
                return;
            }

            // Group counters by name
            const groupedCounters = currentCounters.reduce((acc, counter) => {
                acc[counter.name] = acc[counter.name] || [];
                acc[counter.name].push(counter);
                return acc;
            }, {});

            let html = '';
            for (const name in groupedCounters) {
                const group = groupedCounters[name];
                if (group.length > 1) { // Multiple counters with the same name
                    html += `<details class="compact-counter-group"><summary>${name} (${group.length} instances)</summary>`;
                    group.forEach(counter => html += renderSingleCounterCard(counter));
                    html += `</details>`;
                } else { // Single counter
                    html += renderSingleCounterCard(group[0]);
                }
            }
            elements.countersListDiv.innerHTML = html;

            // Add event listeners using event delegation on countersListDiv
            elements.countersListDiv.removeEventListener('click', handleCounterCardActions); // Remove old if any
            elements.countersListDiv.addEventListener('click', handleCounterCardActions);
        }

        function handleCounterCardActions(event) {
            const target = event.target;
            const counterCard = target.closest('.counter-card');
            if (!counterCard) return;

            const counterId = counterCard.dataset.id;
            const counterName = counterCard.dataset.name; // Store name on card for modals

            if (target.classList.contains('increment-btn')) {
                openIncrementModal(counterId, counterName);
            } else if (target.classList.contains('edit-btn')) {
                openCounterModal(counterId);
            } else if (target.classList.contains('delete-btn')) {
                handleDeleteCounter(counterId);
            } else if (target.classList.contains('view-stats-btn')) {
                openStatsModal(counterId, counterName);
            }
        }

        function renderSingleCounterCard(counter) {
            const progressPercent = counter.target_count > 0 ? Math.min(100, (counter.current_count_in_period / counter.target_count) * 100) : 0;
            const statusText = counter.is_active ? (counter.status || "In Progress") : "Inactive"; // Assuming 'status' is on counter object from get_status
            return `
                <div class="counter-card" data-id="${counter.id}" data-name="${counter.name}">
                    <h3>${counter.name} <small>(${counter.unit_name})</small></h3>
                    ${counter.description ? `<p>${counter.description}</p>` : ''}
                    <p>
                        Target: ${counter.current_count_in_period} / ${counter.target_count} ${counter.unit_name}
                        (${counter.frequency.replace('_', ' ')}${counter.frequency !== 'one_time' ? 'ly' : ''})
                        ${counter.series_end_date ? ` (Ends: ${formatJustDate(counter.series_end_date)})` : ''}
                    </p>
                    <div class="progress-bar-container">
                        <div class="progress-bar" style="width: ${progressPercent}%;">
                            ${Math.round(progressPercent)}%
                        </div>
                    </div>
                    <p>Status: <span class="status-badge">${statusText}</span></p>
                    ${counter.current_period_start_utc ? `<p><small>Current Period: ${formatDate(counter.current_period_start_utc)} - ${formatDate(counter.current_period_end_utc)}</small></p>` : ''}
                    <p><small>Total Accumulated: ${counter.total_accumulated_count} ${counter.unit_name}</small></p>

                    <div class="actions">
                        <button class="btn btn-primary increment-btn">Increment</button>
                        <button class="btn btn-secondary edit-btn">Edit</button>
                        <button class="btn view-stats-btn">Stats</button>
                        <button class="btn delete-btn" style="background-color:#e74c3c; color:white;">Delete</button>
                    </div>
                </div>
            `;
        }

        function openCounterModal(counterIdToEdit = null) {
            const isEdit = !!counterIdToEdit;
            let counter = null;
            if (isEdit) {
                counter = currentCounters.find(c => c.id === counterIdToEdit);
                if (!counter) {
                    if(window.TB?.ui?.Toast) TB.ui.Toast.showError("Counter not found for editing.");
                    return;
                }
            }

            const formId = "counterFormInstance"; // Unique ID for the form
            const content = `
                <form id="${formId}">
                    <input type="hidden" id="modalCounterId" value="${isEdit ? counter.id : ''}">
                    <div><label for="modalCounterName">Name:</label><input type="text" id="modalCounterName" value="${isEdit ? counter.name : ''}" required></div>
                    <div><label for="modalCounterDescription">Description:</label><textarea id="modalCounterDescription">${isEdit && counter.description ? counter.description : ''}</textarea></div>
                    <div><label for="modalCounterUnitName">Unit Name:</label><input type="text" id="modalCounterUnitName" value="${isEdit ? counter.unit_name : 'times'}" required></div>
                    <div><label for="modalCounterTargetCount">Target Count:</label><input type="number" id="modalCounterTargetCount" value="${isEdit ? counter.target_count : 1}" min="1" required></div>
                    <div>
                        <label for="modalCounterFrequency">Frequency:</label>
                        <select id="modalCounterFrequency">
                            <option value="one_time" ${isEdit && counter.frequency === 'one_time' ? 'selected' : ''}>One Time</option>
                            <option value="daily" ${(!isEdit || (isEdit && counter.frequency === 'daily')) ? 'selected' : ''}>Daily</option>
                            <option value="weekly" ${isEdit && counter.frequency === 'weekly' ? 'selected' : ''}>Weekly</option>
                            <option value="monthly" ${isEdit && counter.frequency === 'monthly' ? 'selected' : ''}>Monthly</option>
                        </select>
                    </div>
                    <div><label for="modalCounterSeriesEndDate">End Date (Optional):</label><input type="date" id="modalCounterSeriesEndDate" value="${isEdit && counter.series_end_date ? counter.series_end_date : ''}"></div>
                    <div><label for="modalCounterIsActive">Is Active:</label><input type="checkbox" id="modalCounterIsActive" ${(!isEdit || (isEdit && counter.is_active)) ? 'checked' : ''}></div>
                </form>`;

            TB.ui.Modal.show({
                title: isEdit ? "Edit Counter" : "Add New Counter",
                content: content,
                buttons: [
                    { text: "Cancel", variant: "secondary", action: modal => modal.close() },
                    { text: isEdit ? "Save Changes" : "Create Counter", variant: "primary",
                      action: async (modal) => {
                        const formElement = document.getElementById(formId);
                        if (formElement && formElement.checkValidity()) {
                            await handleSaveCounterFromModal(formElement, modal);
                        } else {
                            if(window.TB?.ui?.Toast) TB.ui.Toast.showError("Please fill all required fields.");
                            formElement.reportValidity(); // Show native browser validation
                        }
                      }
                    }
                ]
            });
        }

        async function handleSaveCounterFromModal(formElement, modalInstance) {
            const id = formElement.querySelector('#modalCounterId').value;
            const counterData = {
                name: formElement.querySelector('#modalCounterName').value,
                description: formElement.querySelector('#modalCounterDescription').value,
                unit_name: formElement.querySelector('#modalCounterUnitName').value,
                target_count: parseInt(formElement.querySelector('#modalCounterTargetCount').value),
                frequency: formElement.querySelector('#modalCounterFrequency').value,
                series_end_date: formElement.querySelector('#modalCounterSeriesEndDate').value || null,
                is_active: formElement.querySelector('#modalCounterIsActive').checked
            };

            let response;
            if (id) {
                response = await apiRequest(`update-counter?counter_id=${id}`, counterData, 'POST');
            } else {
                response = await apiRequest('create-counter', counterData, 'POST');
            }

            if (!response.error) {
                if(window.TB?.ui?.Toast) TB.ui.Toast.showSuccess(`Counter ${id ? 'updated' : 'created'}!`);
                modalInstance.close();
                await loadCounters();
            }
        }

        async function handleDeleteCounter(counterId) {
            const confirmed = await TB.ui.Modal.confirm({
                title: "Delete Counter",
                content: "Are you sure you want to delete this counter and all its entries? This cannot be undone.",
                confirmButtonText: "Delete",
                confirmButtonVariant: "danger"
            });
            if (!confirmed) return;

            const response = await apiRequest(`delete-counter?counter_id=${counterId}`, null, 'POST');
            if (!response.error) {
                if(window.TB?.ui?.Toast) TB.ui.Toast.showSuccess("Counter deleted.");
                await loadCounters();
            }
        }

        function openIncrementModal(counterId, counterName) {
            const formId = "incrementFormInstance";
            const content = `
                <form id="${formId}">
                    <input type="hidden" id="modalIncrementCounterId" value="${counterId}">
                    <p>Incrementing: <strong>${counterName}</strong></p>
                    <div><label for="modalIncrementAmount">Amount:</label><input type="number" id="modalIncrementAmount" value="1" min="1"></div>
                    <div><label for="modalIncrementNotes">Notes (Optional):</label><textarea id="modalIncrementNotes"></textarea></div>
                </form>`;

            TB.ui.Modal.show({
                title: `Increment ${counterName}`,
                content: content,
                onOpen: () => document.getElementById('modalIncrementAmount').focus(),
                buttons: [
                    { text: "Cancel", variant: "secondary", action: modal => modal.close() },
                    { text: "Log Increment", variant: "primary",
                      action: async (modal) => {
                        const formElement = document.getElementById(formId);
                        const amount = parseInt(formElement.querySelector('#modalIncrementAmount').value);
                        const notes = formElement.querySelector('#modalIncrementNotes').value;
                        if (isNaN(amount) || amount < 1) {
                            if(window.TB?.ui?.Toast) TB.ui.Toast.showError("Invalid amount."); return;
                        }
                        const response = await apiRequest(`increment-counter?counter_id=${counterId}`, { amount, notes }, 'POST');
                        if (!response.error) {
                            if(window.TB?.ui?.Toast) TB.ui.Toast.showSuccess("Increment logged!");
                            modal.close();
                            await loadCounters();
                        }
                      }
                    }
                ]
            });
        }

        async function openStatsModal(counterId, counterName) {
            if (statsChartInstance) statsChartInstance.destroy();

            const content = `
                <div id="modalStatsViewDetails"><p>Loading stats...</p></div>
                <div><canvas id="modalStatsViewPlotCanvas" style="min-height: 250px; width:100%;"></canvas></div>
                <div id="modalStatsViewEntries"><h3>Recent Entries:</h3><ul id="modalStatsEntriesListUl"></ul></div>`;

            const modal = TB.ui.Modal.show({
                title: `Stats for: ${counterName}`,
                content: content,
                maxWidth: 'max-w-2xl', // Wider modal for stats
                buttons: [{ text: "Close", variant: "secondary", action: m => m.close() }],
                onClose: () => { if (statsChartInstance) statsChartInstance.destroy(); statsChartInstance = null; }
            });

            // Wait for modal to be in DOM to get canvas context
            await new Promise(resolve => setTimeout(resolve, 50));

            const statsDetailsDiv = document.getElementById('modalStatsViewDetails');
            const statsEntriesListUl = document.getElementById('modalStatsEntriesListUl');
            const statsPlotCanvas = document.getElementById('modalStatsViewPlotCanvas');

            const statsResponse = await apiRequest(`get-counter-stats?counter_id=${counterId}`);
            if (!statsResponse.error && statsResponse.data) {
                renderCounterStatsDetails(statsResponse.data, statsDetailsDiv, statsPlotCanvas);
            } else {
                statsDetailsDiv.innerHTML = "<p>Could not load stats.</p>";
            }

            const entriesResponse = await apiRequest(`get-counter-entries?counter_id=${counterId}&limit=10`);
            if (!entriesResponse.error && entriesResponse.data) {
                renderStatsEntries(entriesResponse.data, statsEntriesListUl, counterName);
            }
        }

        function renderCounterStatsDetails(statsData, detailsDiv, plotCanvas) {
            const counterObj = statsData.counter_info;
            detailsDiv.innerHTML = `
                <p><strong>Status:</strong> ${statsData.current_status}</p>
                <p><strong>Current Period Progress:</strong> ${statsData.current_period_progress_percent}% (${counterObj.current_count_in_period}/${counterObj.target_count} ${counterObj.unit_name})</p>
                <p><strong>Total Entries Logged:</strong> ${statsData.total_entries_logged} ${counterObj.unit_name}</p>
                <p><strong>Frequency:</strong> ${counterObj.frequency.replace('_', ' ')}${counterObj.frequency !== 'one_time' ? 'ly' : ''}</p>
                ${counterObj.current_period_start_utc ? `<p><strong>Current Period:</strong> ${formatDate(counterObj.current_period_start_utc)} to ${formatDate(counterObj.current_period_end_utc)}</p>` : ''}
                ${counterObj.series_end_date ? `<p><strong>Series Ends:</strong> ${formatJustDate(counterObj.series_end_date)}</p>` : ''}
            `;
            renderStatsChart(plotCanvas, counterObj);
        }

        function renderStatsEntries(entries, entriesUl, counterName) {
            if (entries.length === 0) {
                entriesUl.innerHTML = "<li>No recent entries.</li>"; return;
            }
            const unitName = currentCounters.find(c => c.name === counterName)?.unit_name || 'unit(s)';
            entriesUl.innerHTML = entries.map(entry => `
                <li><strong>${entry.amount} ${unitName}</strong> on ${formatDate(entry.timestamp)}
                    ${entry.notes ? ` <small>- <em>${entry.notes}</em></small>` : ''}</li>`).join('');
        }

        function renderStatsChart(canvasElement, counterData) {
            if (statsChartInstance) statsChartInstance.destroy();
            const ctx = canvasElement.getContext('2d');
            statsChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Current Progress'],
                    datasets: [{
                        label: 'Count ('+counterData.unit_name+')', data: [counterData.current_count_in_period],
                        backgroundColor: 'rgba(75, 192, 192, 0.6)', borderColor: 'rgba(75, 192, 192, 1)', borderWidth: 1
                    }, {
                        label: 'Target ('+counterData.unit_name+')', data: [counterData.target_count],
                        backgroundColor: 'rgba(255, 99, 132, 0.2)', borderColor: 'rgba(255, 99, 132, 1)', borderWidth: 1
                    }]
                },
                options: {
                    scales: { y: { beginAtZero: true, suggestedMax: counterData.target_count * 1.2 } },
                    responsive: true, maintainAspectRatio: false
                }
            });
        }

        // --- Overall Stats Tab ---
        async function loadOverallStats() {
            // For overall stats, we might need a new backend endpoint or aggregate client-side from list-counters
            // For simplicity, let's aggregate client-side from `currentCounters`
            // This will be called after `loadCounters` ensures `currentCounters` is fresh if needed.
            if (!currentCounters || currentCounters.length === 0) {
                elements.overallStatsSummaryDiv.innerHTML = "<p>No counters available to show overall stats.</p>";
                elements.topCountersListUl.innerHTML = "";
                if(overallStatsChartInstance) overallStatsChartInstance.destroy();
                return;
            }

            const totalCounters = currentCounters.length;
            const activeCounters = currentCounters.filter(c => c.is_active).length;
            const totalAccumulatedAll = currentCounters.reduce((sum, c) => sum + c.total_accumulated_count, 0);

            elements.overallStatsSummaryDiv.innerHTML = `
                <p><strong>Total Counters:</strong> ${totalCounters}</p>
                <p><strong>Active Counters:</strong> ${activeCounters}</p>
                <p><strong>Total Accumulated (all counters, all time):</strong> ${totalAccumulatedAll}</p>
            `;

            const topFiveCounters = [...currentCounters]
                .sort((a,b) => b.total_accumulated_count - a.total_accumulated_count)
                .slice(0, 5);

            elements.topCountersListUl.innerHTML = topFiveCounters.map(c =>
                `<li>${c.name}: ${c.total_accumulated_count} ${c.unit_name} (total)</li>`
            ).join('');

            renderOverallStatsChart(topFiveCounters);
        }

        function renderOverallStatsChart(topCounters) {
            if (overallStatsChartInstance) overallStatsChartInstance.destroy();
            const ctx = elements.overallStatsViewPlotCanvas.getContext('2d');

            overallStatsChartInstance = new Chart(ctx, {
                type: 'pie', // Or 'doughnut'
                data: {
                    labels: topCounters.map(c => `${c.name} (${c.unit_name})`),
                    datasets: [{
                        label: 'Total Accumulated Count',
                        data: topCounters.map(c => c.total_accumulated_count),
                        backgroundColor: [ // Add more colors if more than 5 top counters shown
                            'rgba(255, 99, 132, 0.7)', 'rgba(54, 162, 235, 0.7)',
                            'rgba(255, 206, 86, 0.7)', 'rgba(75, 192, 192, 0.7)',
                            'rgba(153, 102, 255, 0.7)'
                        ],
                        hoverOffset: 4
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' } } }
            });
        }

        if (window.TB?.events) {
            if (window.TB.config?.get('appRootId') || window.TB._isInitialized === true) initializeApp();
            else window.TB.events.on('tbjs:initialized', initializeApp, { once: true });
        } else {
            console.warn("Toolbox not fully loaded, attempting init on DOMContentLoaded.");
            document.addEventListener('DOMContentLoaded', () => {
                if (window.TB?.events?.on) window.TB.events.on('tbjs:initialized', initializeApp, { once: true });
                else if (window.TB?._isInitialized) initializeApp();
                else console.error("CRITICAL: TB not available after DOMContentLoaded for CounterTracker.");
            });
        }
    }

    // Wait for tbjs to be initialized
if (window.TB?.events) {
    if (window.TB.config?.get('appRootId')) { // A sign that TB.init might have run
         init();
    } else {
        window.TB.events.on('tbjs:initialized', init, { once: true });
    }
} else {
    // Fallback if TB is not even an object yet, very early load
    document.addEventListener('tbjs:initialized', init, { once: true }); // Custom event dispatch from TB.init
}

    </script>""" + """
</body>
</html>"""
    return Result.html(app_instance.web_context() + html_content)


# --- END OF FILE counter_tracker_api.py ---
