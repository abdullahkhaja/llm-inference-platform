"""Synthetic text-to-SQL workloads for serving experiments.

These prompts are synthetic. They are not Spider examples. Prompt shape matches
training: system instruction plus user `Database schema` / `Question` / `SQL:`,
applied later by the serving chat template.

Prefix caching reuses a matching token prefix. Repeating the same schema at the
start of each user message can hit. The same schema pasted in a different
position, or a different schema of similar length, should not.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkItem:
    question: str
    schema: str
    workload: str
    synthetic: bool = True


_SHARED_SCHEMA = """CREATE TABLE employees (
  id INTEGER PRIMARY KEY,
  name TEXT,
  dept_id INTEGER,
  salary INTEGER
);
CREATE TABLE departments (
  id INTEGER PRIMARY KEY,
  name TEXT,
  city TEXT
);"""

_OTHER_SCHEMAS = [
    "CREATE TABLE books (id INTEGER, title TEXT, author TEXT, year INTEGER);",
    "CREATE TABLE flights (id INTEGER, origin TEXT, dest TEXT, seats INTEGER);",
    "CREATE TABLE orders (id INTEGER, customer TEXT, total INTEGER, status TEXT);",
    "CREATE TABLE sensors (id INTEGER, site TEXT, reading REAL, ts TEXT);",
]

_QUESTIONS = [
    "List employee names.",
    "List employees earning more than 50000.",
    "How many employees are in each department?",
    "List departments in Boston.",
    "Which employees earn more than the average salary?",
]


def _long_schema(repeat: int = 8) -> str:
    block = _SHARED_SCHEMA.strip()
    return "\n".join(f"-- shard {i}\n{block}" for i in range(repeat))


def item(workload: str, index: int) -> WorkItem:
    """Return a deterministic synthetic item for this workload and index."""
    name = workload.strip().lower()
    if name in {"short", "default"}:
        return WorkItem(_QUESTIONS[index % len(_QUESTIONS)], _OTHER_SCHEMAS[0], "short")
    if name == "medium":
        return WorkItem(
            _QUESTIONS[index % len(_QUESTIONS)],
            _SHARED_SCHEMA,
            "medium",
        )
    if name == "long":
        return WorkItem(
            _QUESTIONS[index % len(_QUESTIONS)],
            _long_schema(),
            "long",
        )
    if name in {"repeated-schema", "repeated_schema"}:
        return WorkItem(
            _QUESTIONS[index % len(_QUESTIONS)],
            _SHARED_SCHEMA,
            "repeated-schema",
        )
    if name in {"unrelated-schema", "unrelated_schema"}:
        schema = _OTHER_SCHEMAS[index % len(_OTHER_SCHEMAS)]
        return WorkItem(
            f"List rows from the main table. variant {index}.",
            schema,
            "unrelated-schema",
        )
    raise ValueError(
        "workload must be short, medium, long, repeated-schema, or unrelated-schema"
    )
