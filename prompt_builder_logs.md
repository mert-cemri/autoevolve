### Prompt Builder: Context Engineering Overview

This document summarizes what gets captured in `prompt_builder_logs.jsonl` and how prompts are assembled per evolution step.

- Log source: `openevolve/prompt/sampler.py` in `PromptSampler.build_prompt`
- Templates: `openevolve/prompt/templates.py` via `TemplateManager`
- Log file: `/Users/cusgadmin/Documents/AutoEvolve_Math_MAS/autoevolve/prompt_builder_logs.jsonl`

### Log record types

Each evolution step emits three JSONL records (one per line):
- `type=inputs`: High‑level input summary before templating
- `type=details`: Computed context derived during prompt build
- `type=prompt`: Final `system` and `user` messages sent to the LLM

### Schema

- **inputs**
  - `evolution_round`: integer
  - `language`: string (e.g., "python")
  - `diff_based_evolution`: boolean
  - `feature_dimensions`: list of feature names (e.g., ["complexity", "diversity"]) used for MAP‑Elites bins
  - `program_metrics_keys`: list of metric keys available on the current program
  - `previous_programs_count`, `top_programs_count`, `inspirations_count`: integers
  - `artifacts_keys`: list of artifact names included (stdout, summaries, etc.)
  - `current_program_chars`, `parent_program_chars`: lengths of code strings
  - `previous_ids`, `top_ids`, `inspiration_ids`: program IDs selected for context

- **details**
  - `template_key`: chosen user template (e.g., `diff_user` vs `full_rewrite_user`)
  - `fitness_score`: numeric fitness from metrics (excludes feature dimensions)
  - `feature_coords`: formatted feature coordinates string (or "No feature coordinates")
  - `metrics_preview`: first lines of the formatted metrics block (safe, numeric-aware)
  - `improvement_areas_preview`: bullet list of guidance (fitness change, long code, region exploring, etc.)
  - `previous_programs_count`, `top_programs_count`, `inspirations_count`: as above
  - `artifacts_present`: boolean

- **prompt**
  - `system`: final system message (from config or template override)
  - `user`: final user message after filling the template placeholders

### How the prompt is assembled

1) Inputs gathered
- Current program code, its metrics, and prior attempts (`previous_programs`)
- Top programs (best by fitness), optional diverse picks, and inspirations
- Artifacts from the last evaluation (if enabled)
- Config toggles: diff vs full rewrite; include artifacts; template variations

2) Derived context computed
- `metrics_str`: formatted block of `program_metrics`
- `improvement_areas`: guidance from fitness deltas, region exploration, and code length checks
- `evolution_history`: composite section with:
  - Previous attempts (last up to 3)
  - Top programs (up to `prompt.num_top_programs`), plus optional diverse programs
  - Inspirations (with type and unique features)
- Fitness score and feature coordinates

3) Template filling
- Template key: `diff_user` if diff‑based, else `full_rewrite_user`
- Placeholders injected: `{metrics}`, `{improvement_areas}`, `{evolution_history}`, `{current_program}`, `{language}`, `{artifacts}`, `{fitness_score}`, `{feature_coords}`, `{feature_dimensions}`

### Example (from logs)

- **inputs** (abridged):
```json
{"type":"inputs","evolution_round":4,"feature_dimensions":["complexity","diversity"],"program_metrics_keys":["runs_successfully","value_score","distance_score","combined_score"],"previous_programs_count":3,"top_programs_count":3,"inspirations_count":2,"artifacts_keys":["stage1_result","distance_to_global", "solution_quality", "convergence_info", "best_position", "average_distance_to_global", "search_efficiency"]}
```

- **details** (abridged):
```json
{"type":"details","template_key":"diff_user","fitness_score":1.4992,"feature_coords":"No feature coordinates","metrics_preview":["- runs_successfully: 1.0000","- value_score: 0.9997"],"improvement_areas_preview":["- Fitness improved: 0.5003 → 1.4992","- Consider simplifying - code length exceeds 500 characters"],"artifacts_present":true}
```

- **prompt**: contains the full `system` string and the entire `user` message, which includes:
  - Current metrics summary and focus areas
  - Optional artifacts section (Last Execution Output)
  - Evolution history: previous attempts, top programs, and inspirations (with code blocks)
  - Current program (code block)
  - Task instructions and strict diff format requirements (for diff mode)

### Mapping from code to sections

- `PromptSampler.build_prompt`:
  - Gathers inputs and emits `inputs`
  - Computes `metrics_str`, `improvement_areas`, `evolution_history`, `fitness_score`, `feature_coords`, then emits `details`
  - Renders final template → emits `prompt`

- `TemplateManager` (`templates.py`):
  - `DIFF_USER_TEMPLATE` and `FULL_REWRITE_USER_TEMPLATE`
  - Subsections: `EVOLUTION_HISTORY_TEMPLATE`, `PREVIOUS_ATTEMPT_TEMPLATE`, `TOP_PROGRAM_TEMPLATE`, `INSPIRATIONS_SECTION_TEMPLATE`, `INSPIRATION_PROGRAM_TEMPLATE`

### What “context engineering” includes per step

- **Metrics signal**: numeric, safe‑formatted, fitness computed and tracked vs. last attempt
- **Guidance**: fitness change, feature region info, code length nudges
- **History**: last few attempts, top solutions, optional diverse examples
- **Inspirations**: labeled programs (High‑Performer/Alternative/etc.) with “unique features”
- **Artifacts**: evaluator outputs (stdout-like blocks), truncated to size limit
- **Program code**: the current code block the LLM must modify (diff) or rewrite
- **Strict instructions**: diff format or full rewrite instructions embedded in the template

### Tips for interpreting the logs

- Use `evolution_round` to align `inputs`, `details`, and `prompt` lines.
- `metrics_preview` and `improvement_areas_preview` show what the template will see.
- `feature_dimensions` is present even when `feature_coords` shows "No feature coordinates" (no feature metrics provided).
- When `artifacts_present` is true, look for the "Last Execution Output" section in the user prompt.

### Viewing

```bash
# Pretty-print all lines
cat /Users/cusgadmin/Documents/AutoEvolve_Math_MAS/autoevolve/prompt_builder_logs.jsonl | jq .

# Filter by type
jq 'select(.type=="details")' /Users/cusgadmin/Documents/AutoEvolve_Math_MAS/autoevolve/prompt_builder_logs.jsonl
jq 'select(.type=="prompt")'  /Users/cusgadmin/Documents/AutoEvolve_Math_MAS/autoevolve/prompt_builder_logs.jsonl
```

This file should give you a reliable, compact reference for exactly what enters the LLM’s context each step and how it’s structured.
