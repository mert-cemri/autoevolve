### Memory (clean, pluggable, live UI)

Purpose
- Minimal memory for generator–validator evolution/search. Store raw step data and auto-enriched insights to guide future steps.

Schema (schemas.py)
- MemoryEntry (user-provided fields)
  - parent_program_id, child_program_id
  - generator_input (free-form; e.g., {"code": "..."})
  - generator_output (free-form; e.g., {"code": "..."})
  - validator_output (metrics/errors)
  - diff_summary_user (optional short note)
  - generator_prompt (optional dict)
  - iteration (optional)
  - sampling, metadata (optional dicts)
- Auto-enriched (in_memory.py, async): synopsis_ai (dict)
  - overview
  - delta_summary_structured
  - validator_summary_structured
  - tags
  - causal_links [{change,effect,confidence}]
  - selectors [{key,op:'==',value}]
  - pitfalls

Store (in_memory.py)
- Non-blocking add(record):
  - Insert + index (parent/child/iteration + pre-tags)
  - Write live snapshot (JSON) for UI
  - Spawn background GPT summarizer; when done → set synopsis_ai, index tags/selectors/failures/buckets, write snapshot again
- Indexes supported in search(filter_eq):
  - parent, child, iteration, tag
  - sel:<key> (from synopsis selectors)
  - failure_signature, score_bucket, accuracy_bucket (also derived from validator_output if present)

API (base.py)
- add(entry) → id, get(id) → entry, delete(id) → bool
- search(filter_eq: dict, limit, offset) → List[entry]
- list_search_keys() → ["parent","child","iteration","tag","failure_signature","score_bucket","accuracy_bucket"]

Live UI (ui_app.py)
- Snapshot-driven; no backend coupling
- Sidebar filters: parent, child, iteration, tag, selector key/value, failure_signature, score/accuracy buckets
- Nodes table; detail panel shows structured synopsis + raw payload tabs

Run demo + UI
```bash
# UI
python -m streamlit run autoevolve/memory/ui_app.py

# Demo (in another terminal)
export MEMORY_SNAPSHOT_PATH=/tmp/memory_snapshot.json
export OPENAI_API_KEY=...    # optional for GPT synopsis
python autoevolve/memory/demo_step_memory.py
```

Notes
- GPT calls are async (non-blocking). Client timeout is set; you can set OPENAI_SUMMARY_MODEL (default gpt-4.1).
- Thread-safe updates via a lock; atomic snapshot writes.
- Pre-tags expose simple parent features immediately; richer tags/selectors appear after synopsis.
