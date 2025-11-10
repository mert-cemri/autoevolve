import os
import json
import threading
import difflib
from typing import Any, Dict, List, Optional

from .base import MemoryStore
from .schemas import MemoryEntry


class InMemoryMemoryStore(MemoryStore):
    """In-memory store with an id map and recency ordering."""

    def __init__(self) -> None:
        self._by_id: Dict[str, MemoryEntry] = {}
        # Secondary indexes (simple)
        self._idx_parent: Dict[str, set] = {}
        self._idx_child: Dict[str, set] = {}
        self._idx_iter: Dict[int, set] = {}
        self._idx_tag: Dict[str, set] = {}
        # Selectors: key -> value -> set(ids)
        self._idx_sel: Dict[str, Dict[str, set]] = {}
        # Buckets and failures
        self._idx_failure_sig: Dict[str, set] = {}
        self._idx_score_bucket: Dict[str, set] = {}
        self._idx_accuracy_bucket: Dict[str, set] = {}
        self._lock = threading.Lock()
        # Optional snapshot for UI (JSON file path). Set via env MEMORY_SNAPSHOT_PATH
        self._snapshot_path: Optional[str] = os.environ.get("MEMORY_SNAPSHOT_PATH")

    def add(self, record: MemoryEntry) -> str:
        # Insert immediately
        with self._lock:
            self._by_id[record.id] = record
            self._index_basic(record)
            self._pretag_parent_features(record)

        # Kick off non-blocking synopsis generation if applicable
        try:
            if record.synopsis_ai is None and isinstance(record.generator_input, dict) and isinstance(record.generator_output, dict):
                parent_code = record.generator_input.get("code") or record.generator_input
                child_code = record.generator_output.get("code") or record.generator_output
                t = threading.Thread(
                    target=self._summarize_async,
                    args=(
                        record.id,
                        parent_code,
                        child_code,
                        record.validator_output,
                        record.diff_summary_user,
                        record.generator_prompt,
                        record.metadata,
                    ),
                    daemon=True,
                )
                t.start()
        except Exception:
            pass

        # Write snapshot for UI
        self._snapshot_write()
        return record.id

    def get(self, record_id: str) -> Optional[MemoryEntry]:
        with self._lock:
            return self._by_id.get(record_id)

    def delete(self, record_id: str) -> bool:
        with self._lock:
            rec = self._by_id.pop(record_id, None)
            if rec is not None:
                self._deindex(rec)
        if rec is None:
            return False
        return True

    def search(
        self,
        filter_eq: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[MemoryEntry]:
        with self._lock:
            candidate_ids = set(self._by_id.keys())
        # Filter by exact-match on selected payload fields (index derived inside store)
        if filter_eq:
            for k, v in filter_eq.items():
                if k == "parent":
                    with self._lock:
                        candidate_ids &= self._idx_parent.get(str(v), set())
                elif k == "child":
                    with self._lock:
                        candidate_ids &= self._idx_child.get(str(v), set())
                elif k == "iteration":
                    with self._lock:
                        candidate_ids &= self._idx_iter.get(int(v), set())
                elif k == "tag":
                    with self._lock:
                        candidate_ids &= self._idx_tag.get(str(v), set())
                elif k.startswith("sel:"):
                    key = k.split(":", 1)[1]
                    with self._lock:
                        candidate_ids &= self._idx_sel.get(key, {}).get(str(v), set())
                elif k == "failure_signature":
                    with self._lock:
                        candidate_ids &= self._idx_failure_sig.get(str(v), set())
                elif k == "score_bucket":
                    with self._lock:
                        candidate_ids &= self._idx_score_bucket.get(str(v), set())
                elif k == "accuracy_bucket":
                    with self._lock:
                        candidate_ids &= self._idx_accuracy_bucket.get(str(v), set())
                else:
                    with self._lock:
                        candidate_ids = {rid for rid in candidate_ids if self._matches(self._by_id[rid], k, v)}

        with self._lock:
            recs = [self._by_id[rid] for rid in candidate_ids]
        # Order by recency
        recs.sort(key=lambda r: r.updated_at, reverse=True)
        # Page
        return recs[offset: offset + limit]

    # Internal: simple derivation rules for searchable fields
    def _matches(self, entry: MemoryEntry, key: str, value: Any) -> bool:
        # First check top-level canonical fields
        if key == "parent":
            return entry.parent_program_id == value
        if key == "child":
            return entry.child_program_id == value
        if key == "iteration":
            return entry.iteration == value
        return False

    # No upsert in minimal API; keep method for interface compatibility
    def upsert(self, record: MemoryEntry) -> str:
        return self.add(record)

    def list_search_keys(self) -> List[str]:
        # Note: selectors are addressed as keys "sel:<key>"
        return ["parent", "child", "iteration", "tag", "failure_signature", "score_bucket", "accuracy_bucket"]

    def _summarize_async(
        self,
        record_id: str,
        parent_code: Any,
        child_code: Any,
        validator_output: Any,
        diff_summary_user: Any,
        generator_prompt: Any,
        metadata: Any,
    ) -> None:
        try:
            result = self._summarize_with_gpt(
                parent_code=parent_code,
                child_code=child_code,
                validator_output=validator_output,
                diff_summary_user=diff_summary_user,
                generator_prompt=generator_prompt,
                metadata=metadata,
            )
            if result:
                with self._lock:
                    rec = self._by_id.get(record_id)
                    if rec is not None and rec.synopsis_ai is None:
                        rec.synopsis_ai = result
                        self._index_synopsis(rec)
                # Update snapshot after synopsis lands
                self._snapshot_write()
        except Exception:
            return

    # Internal: GPT-based synopsis (uses OpenAI if available). Returns None on failure.
    def _summarize_with_gpt(
        self,
        parent_code: Any,
        child_code: Any,
        validator_output: Any,
        diff_summary_user: Any,
        generator_prompt: Any,
        metadata: Any,
    ) -> Optional[Dict[str, Any]]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, timeout=20.0)

            def safe_json(obj: Any) -> str:
                try:
                    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))[:4000]
                except Exception:
                    return str(obj)[:4000]

            sys_msg = (
                "You are building a managed memory for code evolution/search using a generator–validator loop. "
                "Goal: create a compact, factual, reusable record of this step to guide future evolution. "
                "Output must be strictly based on provided content. No speculation."
            )
            # Prepare a compact unified diff to aid delta extraction
            def code_str(x: Any) -> str:
                return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False) if x is not None else ""

            parent_s = code_str(parent_code).splitlines(keepends=False)
            child_s = code_str(child_code).splitlines(keepends=False)
            diff_lines = list(
                difflib.unified_diff(parent_s, child_s, fromfile="parent", tofile="child", n=2)
            )
            diff_text = "\n".join(diff_lines)[:4000]
            user_payload = {
                "parent_code": parent_code,
                "child_code": child_code,
                "code_diff_unified": diff_text,
                "validator_output": validator_output,
                "diff_summary_user": diff_summary_user,
                "generator_prompt": generator_prompt,
                "metadata": metadata,
            }
            user_msg = (
                "Purpose: We are building a managed memory for generator–validator evolution/search agents solving hard problems. "
                "Your output should help future steps quickly retrieve what mattered and why.\n\n"
                "Return a single JSON object with a top-level key 'synopsis' only.\n"
                "synopsis must be an object that includes at least: \n"
                "  - overview: string (<= 3 sentences; what changed and observed effects)\n"
                "  - delta_summary_structured: [short, factual bullets] (what changed; use code_diff_unified + codes)\n"
                "  - validator_summary_structured: [short, factual bullets] (status, key metrics/errors)\n"
                "  - tags: [strings like 'unicode','api-swap','loop-to-len','perf']\n"
                "  - causal_links: [ { change: string, effect: string, confidence: number in [0,1] } ]\n"
                "  - selectors: [ { key: string, op: '==', value: string|number } ]\n"
                "  - pitfalls: [short bullets] (only if evidence from data)\n"
                "Optional (include if clearly supported by data, else omit): next_step_hints, status, error_code, failure_signature, score_bucket, accuracy_bucket.\n"
                "Strict rules: Use only provided content (no speculation), be concise, and keep JSON valid.\n\n"
                f"DATA: {safe_json(user_payload)}"
            )
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_SUMMARY_MODEL", "gpt-4.1"),
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=400,
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            try:
                data = json.loads(text)
                if isinstance(data, dict) and isinstance(data.get("synopsis"), dict):
                    return data["synopsis"]
                # Best-effort wrap
                return {"overview": text[:1000], "raw": text[:2000]}
            except Exception:
                return {"overview": text[:1000], "raw": text[:2000]}
        except Exception:
            return None

    # Index helpers
    def _index_basic(self, rec: MemoryEntry) -> None:
        pid = str(rec.parent_program_id)
        cid = str(rec.child_program_id)
        it = rec.iteration
        self._idx_parent.setdefault(pid, set()).add(rec.id)
        self._idx_child.setdefault(cid, set()).add(rec.id)
        if it is not None:
            self._idx_iter.setdefault(int(it), set()).add(rec.id)

    def _pretag_parent_features(self, rec: MemoryEntry) -> None:
        try:
            for t in self._compute_pre_tags(rec):
                self._idx_tag.setdefault(t, set()).add(rec.id)
        except Exception:
            pass

    def _index_synopsis(self, rec: MemoryEntry) -> None:
        # tags
        try:
            tags = rec.synopsis_ai.get("tags") if isinstance(rec.synopsis_ai, dict) else None
            if isinstance(tags, list):
                for t in tags:
                    self._idx_tag.setdefault(str(t), set()).add(rec.id)
        except Exception:
            pass
        # selectors (key=='value' only)
        try:
            sels = rec.synopsis_ai.get("selectors") if isinstance(rec.synopsis_ai, dict) else None
            if isinstance(sels, list):
                for sel in sels:
                    if not isinstance(sel, dict):
                        continue
                    k = str(sel.get("key"))
                    op = sel.get("op")
                    v = str(sel.get("value"))
                    if k and op == "==":
                        self._idx_sel.setdefault(k, {}).setdefault(v, set()).add(rec.id)
        except Exception:
            pass
        # failure signature and buckets
        try:
            fsig = rec.synopsis_ai.get("failure_signature") if isinstance(rec.synopsis_ai, dict) else None
            # derive from validator_output if missing
            if not fsig and isinstance(rec.validator_output, dict):
                err = rec.validator_output.get("error")
                if isinstance(err, str) and err:
                    fsig = err.split(":", 1)[0].strip()
            if isinstance(fsig, str) and fsig:
                self._idx_failure_sig.setdefault(fsig, set()).add(rec.id)
        except Exception:
            pass
        try:
            sb = rec.synopsis_ai.get("score_bucket") if isinstance(rec.synopsis_ai, dict) else None
            ab = rec.synopsis_ai.get("accuracy_bucket") if isinstance(rec.synopsis_ai, dict) else None
            # derive buckets from validator_output if missing
            if (not sb or not ab) and isinstance(rec.validator_output, dict):
                try:
                    if not sb and isinstance(rec.validator_output.get("score"), (int, float)):
                        sb = f"{round(float(rec.validator_output['score']), 1):.1f}"
                    if not ab and isinstance(rec.validator_output.get("accuracy"), (int, float)):
                        ab = f"{round(float(rec.validator_output['accuracy']), 1):.1f}"
                except Exception:
                    pass
            if isinstance(sb, str) and sb:
                self._idx_score_bucket.setdefault(sb, set()).add(rec.id)
            if isinstance(ab, str) and ab:
                self._idx_accuracy_bucket.setdefault(ab, set()).add(rec.id)
        except Exception:
            pass

    def _deindex(self, rec: MemoryEntry) -> None:
        try:
            pid = str(rec.parent_program_id)
            cid = str(rec.child_program_id)
            it = rec.iteration
            if pid in self._idx_parent:
                self._idx_parent[pid].discard(rec.id)
            if cid in self._idx_child:
                self._idx_child[cid].discard(rec.id)
            if it is not None and int(it) in self._idx_iter:
                self._idx_iter[int(it)].discard(rec.id)
            # remove pre-tags
            for t in self._compute_pre_tags(rec):
                if t in self._idx_tag:
                    self._idx_tag[t].discard(rec.id)
            # remove tag and selectors
            if isinstance(rec.synopsis_ai, dict):
                tags = rec.synopsis_ai.get("tags")
                if isinstance(tags, list):
                    for t in tags:
                        if t in self._idx_tag:
                            self._idx_tag[t].discard(rec.id)
                sels = rec.synopsis_ai.get("selectors")
                if isinstance(sels, list):
                    for sel in sels:
                        if not isinstance(sel, dict):
                            continue
                        k = str(sel.get("key"))
                        v = str(sel.get("value"))
                        if k in self._idx_sel and v in self._idx_sel[k]:
                            self._idx_sel[k][v].discard(rec.id)
                fsig = rec.synopsis_ai.get("failure_signature")
                if isinstance(fsig, str) and fsig in self._idx_failure_sig:
                    self._idx_failure_sig[fsig].discard(rec.id)
                sb = rec.synopsis_ai.get("score_bucket")
                if isinstance(sb, str) and sb in self._idx_score_bucket:
                    self._idx_score_bucket[sb].discard(rec.id)
                ab = rec.synopsis_ai.get("accuracy_bucket")
                if isinstance(ab, str) and ab in self._idx_accuracy_bucket:
                    self._idx_accuracy_bucket[ab].discard(rec.id)
        except Exception:
            pass

    def _compute_pre_tags(self, rec: MemoryEntry) -> List[str]:
        tags: List[str] = []
        try:
            code = rec.generator_input.get("code") if isinstance(rec.generator_input, dict) else None
            if isinstance(code, str):
                if "for " in code or "while " in code:
                    tags.append("parent:uses_loop")
                if "len(" in code:
                    tags.append("parent:uses_len")
                if "unicodedata" in code:
                    tags.append("parent:imports_unicode")
        except Exception:
            return tags
        return tags

    # Snapshot helper (for simple UI)
    def search_parents_by_code(self, code: str, topk: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar parent programs by code using semantic similarity.
        Falls back to recent programs if semantic search is not available.

        Args:
            code: The code to search for similar parents
            topk: Number of top results to return

        Returns:
            List of dictionaries with parent/child info and metadata
        """
        try:
            # Try semantic search with embeddings if OpenAI is available
            api_key = os.environ.get("OPENAI_API_KEY")
            embed_model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")

            if api_key and code:
                try:
                    from openai import OpenAI
                    import numpy as np
                    import logging

                    logger = logging.getLogger(__name__)

                    # Increased timeout to 30s for embedding API calls
                    client = OpenAI(api_key=api_key, timeout=30.0)

                    # Get embedding for query code
                    query_embedding = client.embeddings.create(
                        input=[code[:8000]],  # Limit code length
                        model=embed_model
                    ).data[0].embedding

                    # Compute similarity with all entries
                    results = []
                    with self._lock:
                        for rid, rec in self._by_id.items():
                            # Get parent code from generator_input
                            parent_code = None
                            if isinstance(rec.generator_input, dict):
                                parent_code = rec.generator_input.get("code")
                            elif isinstance(rec.generator_input, str):
                                parent_code = rec.generator_input

                            if not parent_code:
                                continue

                            # Get embedding for parent code (cache would be good here)
                            try:
                                parent_embedding = client.embeddings.create(
                                    input=[parent_code[:8000]],
                                    model=embed_model
                                ).data[0].embedding

                                # Compute cosine similarity
                                similarity = np.dot(query_embedding, parent_embedding) / (
                                    np.linalg.norm(query_embedding) * np.linalg.norm(parent_embedding)
                                )

                                results.append({
                                    "parent": rec.parent_program_id,
                                    "child": rec.child_program_id,
                                    "generator_input": rec.generator_input,
                                    "generator_output": rec.generator_output,
                                    "validator_output": rec.validator_output,
                                    "similarity": float(similarity),
                                    "iteration": rec.iteration,
                                })
                            except Exception:
                                continue

                    # Sort by similarity and return top-k
                    results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                    logger.debug(f"Memory semantic search: Found {len(results)} similar programs (returning top {topk})")
                    return results[:topk]

                except Exception as e:
                    # Log the error and fall back to simple search
                    logger.warning(f"Memory semantic search failed (falling back to recent): {str(e)}")
                    pass  # Fall back to simple search

        except Exception as e:
            # Outer exception handler
            pass

        # Fallback: return most recent entries
        with self._lock:
            recent = []
            for rid, rec in self._by_id.items():
                recent.append({
                    "parent": rec.parent_program_id,
                    "child": rec.child_program_id,
                    "generator_input": rec.generator_input,
                    "generator_output": rec.generator_output,
                    "validator_output": rec.validator_output,
                    "iteration": rec.iteration,
                })
            # Sort by iteration (most recent first)
            recent.sort(key=lambda x: x.get("iteration", 0) or 0, reverse=True)
            return recent[:topk]

    def _snapshot_write(self) -> None:
        try:
            if not self._snapshot_path:
                return
            with self._lock:
                data = []
                for rid, rec in self._by_id.items():
                    row: Dict[str, Any] = {
                        "id": rid,
                        "parent": rec.parent_program_id,
                        "child": rec.child_program_id,
                        "iteration": rec.iteration,
                        "created_at": rec.created_at,
                        "synopsis": rec.synopsis_ai,
                        "generator_input": rec.generator_input,
                        "generator_output": rec.generator_output,
                        "validator_output": rec.validator_output,
                        "diff_summary_user": rec.diff_summary_user,
                        "generator_prompt": rec.generator_prompt,
                        "metadata": rec.metadata,
                    }
                    # derive tags if available
                    try:
                        if isinstance(rec.synopsis_ai, dict) and isinstance(rec.synopsis_ai.get("tags"), list):
                            row["tags"] = rec.synopsis_ai["tags"]
                    except Exception:
                        pass
                    data.append(row)
            # Write atomically
            tmp = self._snapshot_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"entries": data}, f, ensure_ascii=False)
            os.replace(tmp, self._snapshot_path)
        except Exception:
            return
