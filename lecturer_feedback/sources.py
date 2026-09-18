from __future__ import annotations

import gzip
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from .models import Conversation, ConversationEvent, Dataset, Message, SourceValue, Task, TaskRating


class SourceError(ValueError):
    pass


def _table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _optional(row: sqlite3.Row, columns: set[str], name: str):
    return row[name] if name in columns else None


def load_sqlite(path: Path) -> Dataset:
    if not path.is_file():
        raise SourceError(f"SQLite input does not exist: {path}")

    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as error:
        raise SourceError(f"Could not open SQLite input: {error}") from error
    connection.row_factory = sqlite3.Row

    try:
        tables = _table_names(connection)
        required = {"tasks", "conversations", "messages"}
        missing = sorted(required - tables)
        if missing:
            raise SourceError(f"SQLite input is missing tables: {', '.join(missing)}")

        task_columns = _columns(connection, "tasks")
        tasks: dict[str, Task] = {}
        for row in connection.execute("SELECT * FROM tasks ORDER BY id"):
            task = Task(
                id=str(row["id"]),
                title=str(row["title"]),
                question=str(row["question"]),
                topic=_optional(row, task_columns, "sidebar_label"),
                reference_answer=_optional(row, task_columns, "reference_answer"),
                instructor_context=_optional(row, task_columns, "task_prompt"),
            )
            tasks[task.id] = task

        message_columns = _columns(connection, "messages")
        messages: dict[str, list[Message]] = defaultdict(list)
        message_order = "id" if "id" in message_columns else "created_at"
        for row in connection.execute(f"SELECT * FROM messages ORDER BY conversation_id, {message_order}"):
            role = str(row["role"]).lower().strip()
            if role not in {"system", "developer", "user", "assistant"}:
                raise SourceError(f"Unsupported message role: {role!r}")
            messages[str(row["conversation_id"])].append(
                Message(
                    role=role,
                    content=str(row["content"] or ""),
                    created_at=_optional(row, message_columns, "created_at"),
                )
            )

        events: dict[str, list[ConversationEvent]] = defaultdict(list)
        if "conversation_events" in tables:
            event_columns = _columns(connection, "conversation_events")
            event_order = "id" if "id" in event_columns else "created_at"
            for row in connection.execute(
                f"SELECT * FROM conversation_events ORDER BY conversation_id, {event_order}"
            ):
                rating = _optional(row, event_columns, "rating")
                events[str(row["conversation_id"])].append(
                    ConversationEvent(
                        event_type=str(row["event_type"]),
                        rating=int(rating) if rating is not None else None,
                        created_at=_optional(row, event_columns, "created_at"),
                    )
                )

        conversation_columns = _columns(connection, "conversations")
        conversations = [
            Conversation(
                id=str(row["id"]),
                user_key=(
                    str(row["user_id"])
                    if _optional(row, conversation_columns, "user_id") is not None
                    else None
                ),
                task_id=str(row["task_id"]),
                messages=messages.get(str(row["id"]), []),
                events=events.get(str(row["id"]), []),
            )
            for row in connection.execute("SELECT * FROM conversations ORDER BY id")
        ]

        task_ratings: list[TaskRating] = []
        if "task_ratings" in tables:
            rating_columns = _columns(connection, "task_ratings")
            for row in connection.execute("SELECT * FROM task_ratings"):
                task_ratings.append(
                    TaskRating(
                        task_id=str(row["task_id"]),
                        user_key=(str(row["user_id"]) if "user_id" in rating_columns else None),
                        rating=int(row["rating"]),
                    )
                )
    except (sqlite3.Error, KeyError) as error:
        raise SourceError(f"Invalid SQLite export: {error}") from error
    finally:
        connection.close()

    return Dataset(
        tasks=tasks,
        conversations=conversations,
        task_ratings=task_ratings,
        source_name="SQLite export",
    )


def load_jsonl(path: Path) -> Dataset:
    if not path.is_file():
        raise SourceError(f"JSONL input does not exist: {path}")
    tasks: dict[str, Task] = {}
    conversations: list[Conversation] = []
    task_ratings: list[TaskRating] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                task = Task.model_validate(item["task"])
                conversation = Conversation.model_validate(item["conversation"])
                if conversation.task_id != task.id:
                    raise ValueError("conversation.task_id does not match task.id")
                existing_task = tasks.get(task.id)
                if existing_task and existing_task != task:
                    raise ValueError(f"conflicting definitions for task {task.id!r}")
                tasks[task.id] = task
                conversations.append(conversation)
                for rating in item.get("task_ratings", []):
                    task_ratings.append(TaskRating.model_validate(rating))
            except (json.JSONDecodeError, KeyError, ValueError) as error:
                raise SourceError(f"Invalid JSONL on line {line_number}: {error}") from error
    return Dataset(
        tasks=tasks,
        conversations=conversations,
        task_ratings=task_ratings,
        source_name="JSONL export",
    )


def load_beta_log_gzip(path: Path) -> Dataset:
    """Load the beta-ai-logs.json.gz export used by the demonstration dataset."""
    if not path.is_file():
        raise SourceError(f"Compressed JSON input does not exist: {path}")
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceError(f"Could not read compressed JSON export: {error}") from error

    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise SourceError("Compressed JSON must contain a top-level 'results' array.")

    results = payload["results"]
    task_details: dict[str, dict] = {}
    conversations: list[Conversation] = []
    role_map = {"student": "user", "tutor": "assistant"}

    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            raise SourceError(f"Invalid beta-log result at position {index}: expected an object.")
        try:
            exercise_id = str(item["beta_exercise_id"])
            result_id = str(item["beta_exercise_result_id"])
            title = str(item["exercise_title"]).strip()
            raw_messages = item["conversation"]
        except KeyError as error:
            raise SourceError(f"Beta-log result {index} is missing {error.args[0]!r}.") from error
        if not title or not isinstance(raw_messages, list):
            raise SourceError(f"Beta-log result {index} has an invalid title or conversation.")

        task_id = f"beta-exercise-{exercise_id}"
        details = task_details.setdefault(task_id, {"title": title, "concepts": {}})
        if details["title"] != title:
            raise SourceError(f"Exercise {exercise_id} has conflicting titles.")

        traces = item.get("trace_history") or []
        if not isinstance(traces, list):
            raise SourceError(f"Beta-log result {index} has invalid trace_history.")
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            label = str(trace.get("concept_label") or "").strip()
            description = str(trace.get("concept_description") or "").strip()
            if label:
                previous = details["concepts"].get(label, "")
                if len(description) > len(previous):
                    details["concepts"][label] = description

        messages: list[Message] = []
        for message_index, raw_message in enumerate(raw_messages, start=1):
            if not isinstance(raw_message, dict):
                raise SourceError(
                    f"Beta-log result {index}, message {message_index} is not an object."
                )
            raw_role = str(raw_message.get("role") or "").strip().casefold()
            if raw_role not in role_map:
                raise SourceError(
                    f"Unsupported beta-log role {raw_role!r} in result {index}."
                )
            content = raw_message.get("content")
            if not isinstance(content, str):
                raise SourceError(
                    f"Beta-log result {index}, message {message_index} has non-text content."
                )
            messages.append(Message(role=role_map[raw_role], content=content))

        user_value = item.get("user")
        conversations.append(
            Conversation(
                id=f"beta-result-{result_id}",
                user_key=str(user_value) if user_value is not None else None,
                task_id=task_id,
                messages=messages,
            )
        )

    tasks: dict[str, Task] = {}
    for task_id, details in task_details.items():
        concept_lines = [
            f"- {label}: {description}" if description else f"- {label}"
            for label, description in details["concepts"].items()
        ]
        instructor_context = (
            "Im Export dokumentierte Curriculum-Konzepte:\n" + "\n".join(concept_lines)
            if concept_lines
            else None
        )
        tasks[task_id] = Task(
            id=task_id,
            title=details["title"],
            topic=details["title"],
            question=(
                "Bewerte das im adaptiven Lerngespräch gezeigte Verständnis der dokumentierten "
                "Curriculum-Konzepte. Die konkreten Tutorfragen stehen im Gesprächsverlauf."
            ),
            instructor_context=instructor_context,
        )

    return Dataset(
        tasks=tasks,
        conversations=conversations,
        source_name="Compressed beta-log export",
    )


def load_source(source: SourceValue) -> Dataset:
    if isinstance(source, Dataset):
        return source
    path = Path(source)
    suffix = path.suffix.lower()
    if path.name.lower().endswith(".json.gz"):
        return load_beta_log_gzip(path)
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return load_sqlite(path)
    if suffix in {".jsonl", ".ndjson"}:
        return load_jsonl(path)
    raise SourceError(
        "Input must be SQLite (.db/.sqlite), JSONL (.jsonl), or a beta-log export (.json.gz)."
    )


def validate_dataset(dataset: Dataset) -> list[str]:
    warnings: list[str] = []
    task_ids = set(dataset.tasks)
    orphan_count = sum(1 for conversation in dataset.conversations if conversation.task_id not in task_ids)
    if orphan_count:
        warnings.append(f"{orphan_count} conversation(s) reference a missing task and will be skipped.")
    system_only = sum(
        1
        for conversation in dataset.conversations
        if not any(message.role == "user" and message.content.strip() for message in conversation.messages)
    )
    if system_only:
        warnings.append(f"{system_only} conversation(s) contain no student message and will not be assessed.")
    if not dataset.conversations:
        warnings.append("The input contains no conversations.")
    return warnings
