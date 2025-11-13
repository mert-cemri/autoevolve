#!/usr/bin/env python3
"""
Minimal LLM coach probe (standalone).

Purpose:
- Do NOT change or wire into evolution.
- Just read the latest N rows from intent_log.jsonl (as-is), send to an LLM with a tiny prompt,
  and append both input window and LLM JSON reply to coach_probe.jsonl so you can tail it.

Usage examples:
  python ogi/openevolve/scripts/coach_probe.py \
    --output_dir /Users/cusgadmin/Documents/AutoEvolve_Math_MAS/ogi/openevolve/examples/algotune/affine_transform_2d/openevolve_output_intent_test \
    --window 10 --model gpt-5 --once

  python ogi/openevolve/scripts/coach_probe.py \
    --output_dir /Users/cusgadmin/Documents/AutoEvolve_Math_MAS/ogi/openevolve/examples/algotune/eigenvectors_complex/openevolve_output_intent_test \
    --window 10 --model gpt-5 --watch --interval 30
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_last_n_jsonl(path: str, n: int) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    # Read efficiently even for larger files
    with open(path, "r", encoding="utf-8") as f:
        # Simple approach; file sizes here are manageable
        lines = f.readlines()
    # Parse all lines that are valid JSON
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            rows.append(obj)
        except Exception:
            # Skip malformed lines silently
            continue
    # Sort by iteration (oldest -> newest) in case file order is not chronological
    def _iter_key(obj: Dict[str, Any]) -> int:
        try:
            return int(obj.get("iteration", -1))
        except Exception:
            return -1
    rows.sort(key=_iter_key)
    return rows[-n:] if n > 0 else rows


def build_minimal_context(rows: List[Dict[str, Any]], truncate: int) -> List[Dict[str, Any]]:
    minimal: List[Dict[str, Any]] = []
    for r in rows:
        validator_output = r.get("validator_output", "")
        # Coerce validator_output to a compact string to keep prompt small and deterministic
        if not isinstance(validator_output, str):
            try:
                validator_output = json.dumps(validator_output, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                validator_output = str(validator_output)
        # No truncation applied; pass validator output as-is
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


def make_prompt(
    context_rows: List[Dict[str, Any]],
    task_name: str = "",
    best_metrics: Dict[str, Any] = None,
    task_context: str = "",
) -> str:
    # Keep prompt deterministic and compact; no derived stats, only what we have.
    header = (
        "You are a concise evolution coach. Read recent attempts (intent + validator result + scores and their deltas).\n"
        "Your primary job is to assess whether the search is genuinely stuck (local minimum or oscillation) and explain why.\n"
        "Decide if the search is stuck in a local minimum or oscillating without sustained progress:\n"
        "- Set stagnant=true when recent deltas are mostly non‑positive or very small AND intents repeat or alternate within a narrow set of ideas.\n"
        "- Otherwise return no_action.\n"
        "When stagnant=true: DO NOT propose code or fix suggestions. Produce a crisp diagnosis of why it is stuck, what themes are repeating, and a short call_to_action that urges breaking out of the rabbit hole while preserving correctness.\n"
        "- Be concrete: include 'shoots' (2–3 concise targets to try next) and 'avoid' (1–2 themes to stop), at a strategy level (no code).\n"
        "- Also provide: rabbit_holes (2–3 to stop), big_bets (3 ideas for large gains), success_criteria (1–2 measurable numeric targets), guardrails (1–2 constraints to preserve correctness/contract).\n\n"
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
        "- For call_to_action: be concrete and concise. Name 2–3 specific 'shoots' (targets to try next) and 1–2 'avoid' themes (what to stop doing). Shoots must be high-level levers, not code.\n"
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
    if best_metrics:
        header += "Best_metrics:\n"
        header += json.dumps(best_metrics, ensure_ascii=False, separators=(",", ":")) + "\n\n"
    header += "Rows:\n"
    rows_json = json.dumps(context_rows, ensure_ascii=False, separators=(",", ":"))
    return f"{header}{rows_json}"

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
            # Keep short non-empty lines; drop long code blocks
            short = []
            for ln in lines:
                s = ln.strip()
                if not s:
                    continue
                if len(s) > 200:
                    continue
                short.append(s)
            text = " ".join(short).strip()
        # Trim length
        if len(text) > 600:
            text = text[:600] + "...[truncated]"
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
    # Final clamp
    if len(context) > 1000:
        context = context[:1000] + "...[truncated]"
    return context


def call_llm(model: str, prompt: str, temperature: float, max_tokens: int) -> (str, Dict[str, Any]):
    """
    Call OpenAI Chat Completions in a minimal way.
    We avoid importing if not installed; provide clear error message.
    """
    try:
        # OpenAI SDK v1.x
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OpenAI SDK not installed. Please: pip install openai>=1.0.0"
        ) from e

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    client = OpenAI(api_key=api_key)
    # Some newer models (e.g., gpt-5) expect max_completion_tokens instead of max_tokens
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful, concise JSON-only assistant. Reply with valid JSON matching the schema."},
            {"role": "user", "content": prompt},
        ],
    }
    # gpt-5 uses max_completion_tokens and often ignores response_format; omit it and temperature
    # if str(model).startswith("gpt-5"):
        # kwargs["max_completion_tokens"] = max_tokens
        # Reduce hidden reasoning budget so content fits in the completion allowance
        # kwargs["reasoning"] = {"effort": "medium"}
    # else:
        # kwargs["max_tokens"] = max_tokens
        # kwargs["response_format"] = {"type": "json_object"}

    completion = client.chat.completions.create(**kwargs)
    text = completion.choices[0].message.content or ""
    import pdb; pdb.set_trace()

    # Extract minimal meta for debugging/visibility
    finish_reason = getattr(completion.choices[0], "finish_reason", None)
    usage = getattr(completion, "usage", None)
    meta = {
        "model": getattr(completion, "model", model),
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        },
    }
    return text.strip(), meta


def ensure_logs_dir(output_dir: str) -> str:
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def append_probe_log(log_path: str, record: Dict[str, Any]) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone LLM coach probe (no wiring).")
    parser.add_argument("--output_dir", required=True, help="Path to openevolve_output_* directory.")
    parser.add_argument("--window", type=int, default=10, help="Number of recent rows to send.")
    parser.add_argument("--truncate", type=int, default=300, help="Truncate validator_output to this many chars.")
    parser.add_argument("--model", type=str, default="gpt-5", help="LLM model for the coach.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max tokens for the coach reply.")
    parser.add_argument("--once", action="store_true", help="Run once and exit.")
    parser.add_argument("--watch", action="store_true", help="Keep watching for new rows and probe periodically.")
    parser.add_argument("--interval", type=int, default=30, help="Watch interval seconds.")
    args = parser.parse_args()

    logs_dir = ensure_logs_dir(args.output_dir)
    intent_path = os.path.join(logs_dir, "intent_log.jsonl")
    probe_path = os.path.join(logs_dir, "coach_probe.jsonl")

    if args.once and args.watch:
        print("Choose either --once or --watch, not both.", file=sys.stderr)
        sys.exit(2)

    def run_probe(last_count_signature: int) -> int:
        # Count current lines
        line_count = 0
        if os.path.isfile(intent_path):
            with open(intent_path, "r", encoding="utf-8") as f:
                for _ in f:
                    line_count += 1
        if line_count == 0:
            record = {
                "ts": _ts(),
                "event": "no_intent_log_yet",
                "intent_log_path": intent_path,
            }
            append_probe_log(probe_path, record)
            return last_count_signature

        # Only probe if there are new lines since last probe in watch mode
        if args.watch and line_count == last_count_signature:
            return last_count_signature

        rows = read_last_n_jsonl(intent_path, args.window)
        context_rows = build_minimal_context(rows, args.truncate)
        # Derive a short task name and read a light task context
        task_name = os.path.basename(os.path.dirname(args.output_dir.rstrip(os.sep)))
        best_metrics = read_best_metrics(args.output_dir)
        task_context = read_task_context(args.output_dir)
        prompt = make_prompt(
            context_rows,
            task_name=task_name,
            best_metrics=best_metrics,
            task_context=task_context,
        )

        try:
            reply, meta = call_llm(args.model, prompt, args.temperature, args.max_tokens)
            raw_reply = reply
            used_model = args.model
            used_meta = meta
            # If empty or cut off by length, one retry with larger budget
            if (not reply) or (meta.get("finish_reason") == "length"):
                second_tokens = max(args.max_tokens * 2, args.max_tokens + 200)
                reply, meta = call_llm(args.model, prompt, args.temperature, second_tokens)
                raw_reply = reply
                used_meta = meta
            try:
                reply_json = json.loads(reply)
            except Exception:
                # Fallback once with gpt-5-mini if initial model failed to produce JSON
                fallback_model = "gpt-5-mini" if args.model != "gpt-5-mini" else None
                if fallback_model:
                    try:
                        fb_reply, fb_meta = call_llm(fallback_model, prompt, args.temperature, args.max_tokens)
                        raw_reply = fb_reply
                        used_model = fallback_model
                        used_meta = fb_meta
                        reply_json = json.loads(fb_reply)
                    except Exception:
                        # Still invalid or failed
                        reply_json = {"stagnant": False, "no_action": True, "rationale": "Invalid JSON reply", "raw": raw_reply}
                else:
                    reply_json = {"stagnant": False, "no_action": True, "rationale": "Invalid JSON reply", "raw": raw_reply}
        except Exception as e:
            reply_json = {"stagnant": False, "no_action": True, "rationale": f"LLM call failed: {e.__class__.__name__}: {e}"}

        record = {
            "ts": _ts(),
            "intent_log_path": intent_path,
            "window_size": args.window,
            "model": used_model if 'used_model' in locals() else args.model,
            "context_rows": context_rows,
            "coach_reply": reply_json,
            # Always include the raw model reply for debugging (even if JSON was valid).
            "raw_reply": reply if 'reply' in locals() else None,
            "llm_meta": used_meta if 'used_meta' in locals() else None,
        }
        append_probe_log(probe_path, record)
        return line_count

    # Run once or watch loop
    last_signature = -1
    if args.once:
        last_signature = run_probe(last_signature)
    elif args.watch:
        while True:
            last_signature = run_probe(last_signature)
            time.sleep(args.interval)
    else:
        # Default to once if neither flag provided
        last_signature = run_probe(last_signature)


if __name__ == "__main__":
    main()


