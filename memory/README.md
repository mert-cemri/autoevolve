### Memory Module (pluggable)

Purpose
- Provide a simple, pluggable memory store that any evolution loop can use to add, update, and retrieve contextual information (programs, artifacts, insights, prompts, errors, run metadata) at runtime.
- Usable by both OpenEvolve and custom_search.

Key Ideas
- Namespace: separate logical domains (e.g., "openevolve", "custom_search", "general").
- Kind: record type (e.g., "program", "artifact", "insight", "run", "prompt", "error").
- Tags: quick filtering (e.g., "island:0", "iteration:5", "beam:2", "lineage:1").
- Payload: arbitrary JSON-serializable dict.
- Optional fields: `run_id`, `key` (stable id for upsert/dedup), `relations` (graph links like derived_from).

Components
- `schemas.py` → `MemoryRecord`: data model for memory entries.
- `base.py` → `MemoryStore`: abstract interface (CRUD + search).
- `in_memory.py` → `InMemoryMemoryStore`: default in-process implementation with basic indexes.

Usage
```python
from autoevolve.memory import InMemoryMemoryStore, MemoryRecord

mem = InMemoryMemoryStore()

# Add a record
rec = MemoryRecord(
    namespace="openevolve",
    kind="insight",
    payload={"message": "Fitness plateau at iteration 20", "best": 1.4992},
    tags=["iteration:20", "island:0"],
    run_id="run_2025_10_13",
    key="openevolve:island:0:iter:20:insight",
)
rec_id = mem.add(rec)

# Update tags/payload
mem.update(rec_id, updates={"best": 1.4994}, tags=["note:watch-migration"]) 

# Search by namespace/tags/text
hits = mem.search(namespace="openevolve", tags_any=["island:0"], text_query="plateau")
```

Direct Integration Patterns (no adapters)
- OpenEvolve (at merge time):
  - Write `program` record with metrics, island, iteration, parent/child ids, changes summary; tag with `island:{i}`, `iteration:{t}`, and set `run_id`.
  - Write `artifact` summary records (e.g., stderr summaries) for the parent/child; tag with `artifact:*` labels.
  - During prompt build, query recent `insight` and `artifact` records for the island (and run) to assemble a compact “Memory context” section.
- custom_search (in strategy loop):
  - Write `program` records per candidate with score/metrics and tags like `strategy:beam_search`, `beam:{k}` or `lineage:{i}`, `iteration:{t}`.
  - Retrieve last K insights (or best-per-lineage summaries) to feed into the mutation `prompt_context`.

Notes
- In-memory store is single-process; for multi-process use a process-safe backend (e.g., add a SQLite implementation) or run a single writer with IPC.
- Text search is naive substring over serialized payload; you can later add a vector or regex-capable backend behind the same interface.
- The schema is intentionally minimal + extensible via `payload`, `tags`, `run_id`, `key`, and `relations` so you won’t need breaking changes as needs grow.
