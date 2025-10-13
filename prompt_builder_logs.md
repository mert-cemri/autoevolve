### Prompt Builder: Context Engineering — Full Specification

This document explains exactly how the prompt is constructed each evolution step: what inputs are gathered, how they are processed, which templates are used, and what the final output looks like. It also notes current gaps and potential improvements.

- Orchestrator: `openevolve/prompt/sampler.py` in `PromptSampler.build_prompt`
- Templates & fragments: `openevolve/prompt/templates.py` via `TemplateManager`
- Logs: `/Users/cusgadmin/Documents/AutoEvolve_Math_MAS/autoevolve/prompt_builder_logs.jsonl`
  - `type=inputs` → raw inputs summary
  - `type=details` → derived context and previews
  - `type=prompt` → final `system` and `user` messages

### End-to-end pipeline (one evolution step)

1) Resolve templates
- Select user template: `diff_user` if `diff_based_evolution=True`, else `full_rewrite_user`. Respect `set_templates(...)` overrides if present.
- Resolve system message: use config’s `system_message`; if it names a template, resolve via `TemplateManager`.
- Optional: apply stochastic variations if `prompt.use_template_stochasticity=True` (simple placeholder replacement from `template_variations`).

2) Gather inputs (from controller/database)
- `current_program` (string) and `parent_program` (string)
- `program_metrics` (dict of metric name → value)
- `previous_programs` (list of program dicts; same-island history)
- `top_programs` (list; best by fitness score)
- `inspirations` (list; diverse/creative examples)
- `program_artifacts` (dict of artifact key → string or bytes)
- `feature_dimensions` (list of dimension names used by MAP‑Elites)
- `language`, `evolution_round`, `diff_based_evolution`, plus extra `**kwargs` for template fields if provided

3) Emit inputs log
- Write `type=inputs` with counts, ids, code lengths, keys, and flags. This is a compact snapshot to correlate later with the final prompt.

4) Compute derived context
- Format metrics block: `_format_metrics(program_metrics)` → numeric-aware pretty list: `- name: value` (with `:.4f` for numeric values).
- Compute fitness: `get_fitness_score(program_metrics, feature_dimensions)` (excludes feature dimensions from the fitness combination).
- Compute feature coordinates: `format_feature_coordinates(program_metrics, feature_dimensions)` → human-readable string or `"No feature coordinates"`.
- Identify improvement areas: `_identify_improvement_areas(...)`:
  - Compare current fitness vs previous attempt’s fitness (if any): emits a fragment for improved/declined/stable.
  - If `feature_dimensions` set and `feature_coords` available, note exploring region.
  - If code length exceeds threshold (`suggest_simplification_after_chars` or `code_length_threshold`), add a simplification hint.
  - If nothing else, add a default guidance fragment.
- Build evolution history: `_format_evolution_history(previous_programs, top_programs, inspirations, language, feature_dimensions)`:
  - Previous attempts: include up to last 3, newest last. For each, show `changes` (from `program.metadata.changes`), a safely formatted metric summary, and an outcome computed by numeric-only comparisons vs the parent’s metrics.
  - Top programs: include up to `prompt.num_top_programs`; compute score via `get_fitness_score`; derive “key features” heuristically from metrics if not provided.
  - Diverse programs: if `prompt.num_diverse_programs > 0` and more programs remain beyond tops, randomly sample from the remainder and render using the same `top_program` template under a “Diverse Programs” header.
  - Inspirations: `_format_inspirations_section(...)` lists programs with score, inferred `program_type` (`High-Performer`/`Alternative`/`Experimental`/`Exploratory` or metadata flags like `diverse`/`migrant`/`random`), and “unique features” from `_extract_unique_features` (short heuristics based on metadata, metrics, and code).
- Render artifacts (optional): `_render_artifacts(program_artifacts)`:
  - Convert bytes to UTF‑8 (replace on errors); apply optional `artifact_security_filter` (removes ANSI and masks tokens/passwords).
  - Truncate long content to `max_artifact_bytes` with a `(truncated)` suffix.
  - Produce a "Last Execution Output" section with fenced blocks per artifact key.

5) Emit details log
- Write `type=details` with: chosen `template_key`, `fitness_score`, `feature_coords`, previews of `metrics` and `improvement_areas`, counts, and `artifacts_present`.

6) Assemble final prompt
- Fill the selected user template with:
  - `{metrics}` → formatted metrics block
  - `{improvement_areas}` → bullet guidance
  - `{evolution_history}` → combined previous/top/diverse/inspirations section
  - `{current_program}` → the full code block to modify (diff mode) or rewrite
  - `{language}` → code block language tag
  - `{artifacts}` → optional rendered artifacts section
  - `{fitness_score}` → numeric with `:.4f`
  - `{feature_coords}` and `{feature_dimensions}` → diversity context
  - Any other `**kwargs` passed in
- System message is resolved as above.

7) Emit final prompt log
- Write `type=prompt` with the exact `system` and `user` strings sent to the LLM.

### Selection policies and heuristics (as implemented now)

- Previous attempts: include up to 3 most recent from `previous_programs` (island history).
- Top programs: take the first `num_top_programs` from the provided `top_programs` list.
- Diverse programs: uniformly sample (Python `random.sample`) up to `num_diverse_programs` from the remaining `top_programs` after the top set.
- Inspirations: included as provided; `program_type` inferred by metadata flags or fitness ranges.
- Artifacts: include all keys present, each truncated at `max_artifact_bytes` and optionally filtered for secrets.

### Templating inventory

- User templates: `diff_user`, `full_rewrite_user`
- History templates: `evolution_history`, `previous_attempt`, `top_program`
- Inspirations: `inspirations_section`, `inspiration_program`
- Evaluator system template (for LLM evaluators): `evaluator_system_message` (not used in generation prompts)

### Concrete example (from your logs, abridged)

- Inputs (R=4): includes metrics keys `[runs_successfully, value_score, distance_score, combined_score]`, 3 previous, 3 top, 2 inspirations, artifacts present.
- Details (R=4): `template_key=diff_user`, `fitness_score≈1.4992`, `feature_coords="No feature coordinates"`; improvement areas note fitness improvement and code length hint.
- Prompt (R=4): user message contains metrics, focus areas, a rich artifacts section, previous attempts, top programs (with code blocks), inspirations (with types/features), the current program, and strict diff instructions.

### What’s in the LLM context each step (checklist)

- System message: high-level evolution role/instruction.
- Metrics: safe-formatted numeric signals for the current program.
- Fitness: scalar summary used for guidance and ranking.
- Feature diversity: names and coordinates (if available) to support MAP‑Elites framing.
- Improvement areas: deltas vs previous, exploration notes, and code length nudges.
- Evolution history: last few attempts with outcomes; top programs; optional diverse examples.
- Inspirations: labeled examples emphasizing variety and “unique features.”
- Artifacts: evaluator outputs/errors (sanitized and truncated) from the last run.
- Current code: the exact block to modify (diff mode) or replace (full rewrite mode).
- Strict task instructions: search/replace diff format (diff mode) or complete rewrite guardrails.

### Configuration knobs that influence prompts

- `prompt.num_top_programs`, `prompt.num_diverse_programs`
- `prompt.use_template_stochasticity`, `prompt.template_variations`
- `prompt.include_artifacts`, `prompt.max_artifact_bytes`, `prompt.artifact_security_filter`
- `prompt.suggest_simplification_after_chars`, `prompt.code_length_threshold`
- `prompt.system_message` (or templates override)
- `diff_based_evolution` (runtime)

### Working with the logs

```bash
# Pretty-print everything
jq . /Users/cusgadmin/Documents/AutoEvolve_Math_MAS/autoevolve/prompt_builder_logs.jsonl

# Correlate a specific round
jq 'select(.evolution_round==4)' /Users/cusgadmin/Documents/AutoEvolve_Math_MAS/autoevolve/prompt_builder_logs.jsonl

# Inspect only the final prompts
jq 'select(.type=="prompt")' /Users/cusgadmin/Documents/AutoEvolve_Math_MAS/autoevolve/prompt_builder_logs.jsonl
```

This spec is synchronized with the current implementation in `PromptSampler.build_prompt` and the templates. Use it as the ground truth for what enters the LLM context and where each part comes from.

### Where do "changes" and artifacts come from?

- Changes (summary of code edits)
  - Source: Worker iteration after LLM response parsing
    - Diff mode (`diff_based_evolution=True`):
      - Parse response → `extract_diffs(...)`
      - Apply to parent code → `apply_diff(...)`
      - Summarize → `format_diff_summary(diff_blocks)`
    - Full rewrite mode: set to "Full rewrite"
  - Stored in the child program’s metadata:
    - `child_program.metadata["changes"] = changes_summary`
  - Code references:
    - `openevolve/process_parallel.py` inside `_run_iteration_worker` (see `changes_summary` assignment and metadata)
    - `openevolve/iteration.py` for the shared-DB path (same logic)
  - Used by prompt:
    - `PromptSampler._format_evolution_history` reads `program.metadata["changes"]` to render the "Previous Attempts" section.

- Artifacts (evaluation outputs/errors/telemetry)
  - Produced by evaluator per child program ID:
    - After evaluation: `artifacts = evaluator.get_pending_artifacts(child_id)`
  - Persisted in the database by the main process:
    - On merge: `database.store_artifacts(child_program.id, artifacts)` (see `process_parallel.py` integration flow)
  - Provided to prompt for the next round as parent artifacts:
    - The worker pulls `parent_artifacts = db_snapshot["artifacts"].get(parent_id)`
    - Passed into `build_prompt(..., program_artifacts=parent_artifacts)`
  - Rendered in prompt if enabled:
    - `PromptSampler._render_artifacts` builds the "Last Execution Output" section with fenced blocks per artifact key, applying optional sanitization and truncation per config.
  - Code references:
    - `openevolve/process_parallel.py` (collect, attach to result, store on merge, snapshot selection)
    - `openevolve/prompt/sampler.py` (render and include)

### Flow summary (changes + artifacts)

1) LLM produces a diff or full rewrite → parse and apply to parent → compute `changes_summary` → embed in child `metadata.changes`.
2) Evaluator runs child → `get_pending_artifacts(child_id)` returns artifacts dict → stored by DB on merge.
3) Next iteration uses the (now-parent) program:
   - `metadata.changes` shows up under "Previous Attempts"
   - Stored artifacts for that parent are passed as `program_artifacts` → rendered under "Last Execution Output" when `prompt.include_artifacts=True`.

These sources ensure the prompt’s history and execution outputs are grounded in actual prior generations and evaluations.
