"""
Shared helpers for building the coach diagnosis prompt and context.

These are intentionally kept in sync with scripts/coach_probe.py so both the
standalone probe and inline controller use the exact same logic.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def read_last_n_jsonl(path: str, n: int) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            rows.append(obj)
        except Exception:
            continue
    # Sort by iteration (oldest -> newest) in case file order is not chronological
    def _iter_key(obj: Dict[str, Any]) -> int:
        try:
            return int(obj.get("iteration", -1))
        except Exception:
            return -1
    rows.sort(key=_iter_key)
    return rows[-n:] if n > 0 else rows


def build_minimal_context(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    minimal: List[Dict[str, Any]] = []
    for r in rows:
        validator_output = r.get("validator_output", "")
        # Coerce validator_output to a compact string to keep prompt small and deterministic
        if not isinstance(validator_output, str):
            try:
                validator_output = json.dumps(validator_output, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                validator_output = str(validator_output)
        minimal.append({
            "iteration": r.get("iteration"),
            "parent_id": r.get("parent_id"),
            "child_id": r.get("child_id"),
            "intent": (r.get("llm_intent") or r.get("intent") or ""),
            "validator_output": validator_output,
            "primary_metric": r.get("primary_metric"),
            "primary_metric_delta": r.get("primary_metric_delta"),
        })
    return minimal


def read_best_metrics(output_dir: str) -> Dict[str, Any]:
    """
    Read best/best_program_info.json if present to give the coach the current best metrics.
    """
    best_path = os.path.join(output_dir, "best", "best_program_info.json")
    if not os.path.isfile(best_path):
        return {}
    try:
        with open(best_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        # Keep compact subset if available
        keys = ["combined_score", "performance_score", "correctness_score", "speedup_score", "best_program_id"]
        return {k: info.get(k) for k in keys if k in info}
    except Exception:
        return {}


def _extract_docstring_or_head(path: str, head_lines: int = 30) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Try simple triple-quote docstring extraction
        import re
        m = re.search(r'"""(.*?)"""', content, re.DOTALL)
        if not m:
            m = re.search(r"'''(.*?)'''", content, re.DOTALL)
        if m:
            text = m.group(1).strip()
        else:
            # Fallback to first N non-empty comment/code lines
            lines = content.splitlines()[:head_lines]
            short = []
            for ln in lines:
                s = ln.strip()
                if not s:
                    continue
                if len(s) > 200:
                    continue
                short.append(s)
            text = " ".join(short).strip()
        return text
    except Exception:
        return ""


def read_task_context(output_dir: str) -> str:
    """
    Read a compact problem context from the task directory:
    - Try initial_program.py and evaluator.py (docstrings or top comments)
    - Concatenate succinctly
    """
    task_dir = os.path.dirname(output_dir.rstrip(os.sep))
    candidates = [
        os.path.join(task_dir, "initial_program.py"),
        os.path.join(task_dir, "evaluator.py"),
    ]
    parts: List[str] = []
    for p in candidates:
        if os.path.isfile(p):
            snippet = _extract_docstring_or_head(p)
            if snippet:
                parts.append(snippet)
    context = "\n".join(parts).strip()
    return context


def make_prompt(
    context_rows: List[Dict[str, Any]],
    task_name: str = "",
    best_metrics: Optional[Dict[str, Any]] = None,
    task_context: str = "",
    best_plateau: Optional[bool] = None,
    best_stable_iters: Optional[int] = None,
    window_size: Optional[int] = None,
) -> str:
    # Keep prompt deterministic and compact; no derived stats, only what we have.
    header = (
        "You are a concise evolution coach. Read recent attempts (intent + validator result + scores and their deltas).\n"
        "Your primary job is to assess whether the search is genuinely stuck (local minimum or oscillation) and explain why.\n"
        "Decide if the search is stuck in a local minimum or oscillating without sustained progress:\n"
        "- Set stagnant=true when recent deltas are mostly non‑positive or very small AND intents repeat or alternate within a narrow set of ideas.\n"
        "- Otherwise return no_action.\n"
        "When stagnant=true: DO NOT propose code or fix suggestions. Produce a crisp diagnosis of why it is stuck, what themes are repeating, and a short call_to_action that urges breaking out of the rabbit hole while preserving correctness.\n"
        "- Be concrete for exploration: name rabbit_holes (what to stop), big_bets (plausible ideas for large gains), success_criteria (numeric targets for a win), and guardrails (constraints to keep correctness stable).\n\n"
    )
    if task_name:
        header += f"Task: {task_name}\n"
    if task_context:
        header += "Task_context:\n"
        header += task_context + "\n"
    header += (
        "Context (newest last):\n"
        "- Rows: JSON array of objects {iteration, parent_id, child_id, intent, validator_output, primary_metric, primary_metric_delta}\n\n"
        "Rules:\n"
        "- If unsure, return no_action.\n"
        "- First produce a one‑sentence stuck_summary describing the repeated intent themes and why scores are stalled or oscillating.\n"
        "- No code output. Provide strategy-level guidance only.\n"
        "- For call_to_action: be concrete and concise. Name 2–3 specific 'shoots' (targets to try next) and 1–2 'avoid' themes (what to stop doing). Shoots must be high-level levers (e.g., change search pattern, adjust constraint/objective, swap algorithmic subcomponent), not code.\n"
        "- Also provide: rabbit_holes (2–3 to stop), big_bets (3 ideas for large gains that differ materially from recent attempts), success_criteria (1–2 measurable numeric targets for a win this burst), guardrails (1–2 correctness/contract constraints).\n"
        "- Always preserve task contract and correctness. If correctness is not consistently 1.0, prioritize restoring correctness before performance tuning.\n"
        "- Also report: primary_metric_name (from rows, if consistent), window_size (N rows), nonpos_delta_fraction (fraction of deltas ≤ 0 or noise‑small), and a brief evidence list (1–2 short items like delta band or min/median/max examples).\n"
        "- Output strictly as JSON per schema. No extra text.\n\n"
        "Schema:\n"
        "{\n"
        '  "stagnant": true|false,\n'
        '  "no_action": true|false,\n'
        '  "stuck_summary": "one sentence",\n'
        '  "repeated_themes": ["short phrase", "short phrase"],\n'
        '  "delta_summary": "one sentence on deltas/oscillation range",\n'
        '  "primary_metric_name": "string or null",\n'
        '  "window_size": 10,\n'
        '  "nonpos_delta_fraction": 0.8,\n'
        '  "evidence": ["short item", "short item"],\n'
        '  "call_to_action": "one sentence urging to escape the rabbit hole while preserving correctness",\n'
        '  "shoots": ["concise target to try", "concise target to try"],\n'
        '  "avoid": ["concise theme to stop", "concise theme to stop"],\n'
        '  "rabbit_holes": ["theme to stop", "theme to stop"],\n'
        '  "big_bets": ["wild but plausible idea", "wild but plausible idea", "wild but plausible idea"],\n'
        '  "success_criteria": ["numeric target like \\"+0.01 combined_score in 10 iters\\"", "another precise target"],\n'
        '  "guardrails": ["constraint such as \\"correctness_score must stay 1.0\\"", "IO/API contract must be preserved"]\n'
        "}\n\n"
    )
    # Provide plateau metadata to make the decision more decisive
    if best_plateau is not None:
        plateau_info = {
            "best_plateau": bool(best_plateau),
            "best_stable_iters": int(best_stable_iters or 0),
            "window_size": int(window_size or 0),
        }
        header += "Plateau_state:\n"
        header += json.dumps(plateau_info, ensure_ascii=False, separators=(",", ":")) + "\n"
        header += (
            "- Decision guidance: If best_plateau=true (best solution unchanged for at least window_size iterations), "
            "then default to stagnant=true unless there is a clear improving trend across most of the last window rows.\n\n"
        )
    if best_metrics:
        header += "Best_metrics:\n"
        header += json.dumps(best_metrics, ensure_ascii=False, separators=(",", ":")) + "\n\n"
    header += "Rows:\n"
    rows_json = json.dumps(context_rows, ensure_ascii=False, separators=(",", ":"))
    return f"{header}{rows_json}"


