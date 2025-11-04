import os
import json
import threading
import logging
from typing import Any, Dict, List, Optional

from .base import MemoryStore
from .schemas import MemoryEntry

logger = logging.getLogger(__name__)


class InMemoryMemoryStore(MemoryStore):
    """In-memory store with an id map and recency ordering.
    
    Simple mode only: indexes basic fields, status, and metric buckets.
    Supports embedding-based semantic search for similar parent programs.
    """

    def __init__(self) -> None:
        self._by_id: Dict[str, MemoryEntry] = {}
        # Secondary indexes (basic)
        self._idx_parent: Dict[str, set] = {}
        self._idx_child: Dict[str, set] = {}
        self._idx_iter: Dict[int, set] = {}
        # Selectors: key -> value -> set(ids)
        self._idx_sel: Dict[str, Dict[str, set]] = {}
        # Buckets and failures
        self._idx_failure_sig: Dict[str, set] = {}
        self._idx_score_bucket: Dict[str, set] = {}
        self._idx_accuracy_bucket: Dict[str, set] = {}
        # Generic metric buckets: key -> bucket_value -> set(ids)
        self._idx_metric_bucket: Dict[str, Dict[str, set]] = {}
        self._lock = threading.Lock()
        # Optional snapshot for UI (JSON file path). Set via env MEMORY_SNAPSHOT_PATH
        self._snapshot_path: Optional[str] = os.environ.get("MEMORY_SNAPSHOT_PATH")
        # Embeddings (per parent program). Non-blocking generation using OpenAI if key is present
        self._embed_enabled: bool = os.environ.get("OPENAI_API_KEY") is not None
        self._embed_model: str = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")
        self._parent_embed: Dict[str, List[float]] = {}
        self._parent_embed_pending: set = set()

    def add(self, record: MemoryEntry) -> str:
        # Insert immediately
        with self._lock:
            self._by_id[record.id] = record
            self._index_basic(record)
            self._index_status_and_metrics(record)

        # Kick off non-blocking parent embedding if applicable
        try:
            self._maybe_start_parent_embedding(record)
        except Exception:
            logger.error("Failed to start parent embedding thread", exc_info=True)

        # Write snapshot for UI
        self._snapshot_write()
        return record.id

    def get(self, record_id: str) -> Optional[MemoryEntry]:
        with self._lock:
            return self._by_id.get(record_id)

    def load_from_snapshot(self, snapshot_path: str) -> tuple[int, int]:
        """
        Load memory entries and embeddings from snapshot files.
        
        Args:
            snapshot_path: Path to memory_snapshot.json file.
                          Embeddings will be loaded from memory_embeddings.json in the same directory.
        
        Returns:
            Tuple of (entries_count, embeddings_count) loaded.
        
        Raises:
            FileNotFoundError: If snapshot_path does not exist.
            ValueError: If snapshot file format is invalid.
        """
        if not os.path.exists(snapshot_path):
            raise FileNotFoundError(f"Memory snapshot not found: {snapshot_path}")
        
        entries_count = 0
        embeddings_count = 0
        
        try:
            # Load entries from memory_snapshot.json
            with open(snapshot_path, "r", encoding="utf-8") as f:
                snapshot_data = json.load(f)
            
            if not isinstance(snapshot_data, dict) or "entries" not in snapshot_data:
                raise ValueError(f"Invalid snapshot format: expected dict with 'entries' key")
            
            entries = snapshot_data.get("entries", [])
            if not isinstance(entries, list):
                raise ValueError(f"Invalid snapshot format: 'entries' should be a list")
            
            logger.info(f"Loading {len(entries)} memory entries from snapshot: {snapshot_path}")
            
            # Reconstruct and index entries
            with self._lock:
                for entry_dict in entries:
                    try:
                        # Reconstruct MemoryEntry from snapshot data
                        # Snapshot format matches what we write in _snapshot_write
                        entry = MemoryEntry(
                            parent_program_id=str(entry_dict.get("parent", "")),
                            child_program_id=str(entry_dict.get("child", "")),
                            generator_input=entry_dict.get("generator_input"),
                            generator_output=entry_dict.get("generator_output"),
                            validator_output=entry_dict.get("validator_output"),
                            diff_summary_user=entry_dict.get("diff_summary_user"),
                            synopsis_ai=entry_dict.get("synopsis"),
                            generator_prompt=entry_dict.get("generator_prompt"),
                            iteration=entry_dict.get("iteration"),
                            metadata=entry_dict.get("metadata"),
                            id=str(entry_dict.get("id", "")),  # Preserve original ID
                            created_at=float(entry_dict.get("created_at", 0.0)),
                            updated_at=float(entry_dict.get("updated_at", entry_dict.get("created_at", 0.0))),
                        )
                        
                        # Add to store
                        self._by_id[entry.id] = entry
                        self._index_basic(entry)
                        self._index_status_and_metrics(entry)
                        entries_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load memory entry from snapshot: {e}", exc_info=True)
                        continue
            
            # Load embeddings from memory_embeddings.json (in same directory)
            index_dir = os.path.dirname(snapshot_path)
            embeddings_path = os.path.join(index_dir, "memory_embeddings.json")
            
            if os.path.exists(embeddings_path):
                try:
                    with open(embeddings_path, "r", encoding="utf-8") as f:
                        embeddings_data = json.load(f)
                    
                    if isinstance(embeddings_data, dict):
                        loaded_model = embeddings_data.get("model")
                        parents_list = embeddings_data.get("parents", [])
                        
                        if loaded_model and loaded_model != self._embed_model:
                            logger.warning(
                                f"Embedding model mismatch: snapshot uses '{loaded_model}', "
                                f"current config uses '{self._embed_model}'. "
                                f"Loaded embeddings may not be compatible with current searches."
                            )
                        
                        with self._lock:
                            for parent_data in parents_list:
                                if isinstance(parent_data, dict):
                                    parent_id = str(parent_data.get("parent", ""))
                                    vector = parent_data.get("vector")
                                    if parent_id and isinstance(vector, list):
                                        try:
                                            # Ensure vector is list of floats
                                            self._parent_embed[parent_id] = [float(v) for v in vector]
                                            embeddings_count += 1
                                        except (ValueError, TypeError) as e:
                                            logger.warning(
                                                f"Failed to load embedding for parent {parent_id}: {e}"
                                            )
                                            continue
                        
                        logger.info(f"Loaded {embeddings_count} parent embeddings from: {embeddings_path}")
                    else:
                        logger.warning(f"Invalid embeddings file format: {embeddings_path}")
                except Exception as e:
                    logger.warning(f"Failed to load embeddings from {embeddings_path}: {e}", exc_info=True)
            else:
                logger.info(f"Embeddings file not found (expected at {embeddings_path}), continuing without embeddings")
            
            logger.info(
                f"Successfully loaded memory snapshot: {entries_count} entries, "
                f"{embeddings_count} embeddings"
            )
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in snapshot file {snapshot_path}: {e}") from e
        except Exception as e:
            logger.error(f"Failed to load memory snapshot from {snapshot_path}: {e}", exc_info=True)
            raise
        
        return entries_count, embeddings_count

    # --- Simple similarity search (parent program) ---
    def search_parents_by_code(self, program_code: str, topk: int = 3) -> List[Dict[str, Any]]:
        """
        Given a program's code, return all memory entries whose parent program id
        is among the top-k most similar parents by embedding cosine similarity.

        Simple behavior:
        - If no embeddings or API key, return empty list.
        - Computes one embedding for the provided code using OPENAI_EMBED_MODEL.
        - Ranks against stored parent embeddings (self._parent_embed).
        - Selects top-k parent ids and returns ALL entries whose parent matches.
        """
        try:
            # Fast fail if no vectors available
            with self._lock:
                vectors = dict(self._parent_embed)
            if not vectors:
                return []

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return []

            # Embed query code
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, timeout=20.0)
                text = program_code if isinstance(program_code, str) else str(program_code)
                if len(text) > 200_000:
                    text = text[:200_000]
                r = client.embeddings.create(model=self._embed_model, input=text)
                qvec = r.data[0].embedding if getattr(r, "data", None) else None
            except Exception:
                logger.error("Parent search: failed to embed query code", exc_info=True)
                return []
            if not qvec:
                return []

            # Cosine similarity
            def _cos(a: List[float], b: List[float]) -> float:
                try:
                    import math
                    s = sum(x * y for x, y in zip(a, b))
                    na = math.sqrt(sum(x * x for x in a))
                    nb = math.sqrt(sum(y * y for y in b))
                    if na == 0.0 or nb == 0.0:
                        return 0.0
                    return s / (na * nb)
                except Exception:
                    return 0.0

            sims: List[tuple] = []
            for pid, vec in vectors.items():
                if isinstance(vec, list) and len(vec) == len(qvec):
                    sims.append((pid, _cos(qvec, vec)))
            sims.sort(key=lambda x: x[1], reverse=True)
            selected_parent_ids = [pid for pid, _ in sims[: max(0, int(topk))]]
            if not selected_parent_ids:
                return []

            # Collect entries for those parents, mirroring snapshot row shape simply
            with self._lock:
                results: List[Dict[str, Any]] = []
                for rid, rec in self._by_id.items():
                    if str(rec.parent_program_id) in selected_parent_ids:
                        results.append(
                            {
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
                        )
            return results
        except Exception:
            logger.error("Parent search failed", exc_info=True)
            return []

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
                elif k.startswith("sel:"):
                    key = k.split(":", 1)[1]
                    with self._lock:
                        # Support generic selectors and metric bucket selectors
                        if key.startswith("metric:"):
                            candidate_ids &= self._idx_metric_bucket.get(key, {}).get(str(v), set())
                        else:
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
        return [
            "parent",
            "child",
            "iteration",
            "failure_signature",
            "score_bucket",
            "accuracy_bucket",
            "status",
            "sel:metric:<name>_bucket",
        ]

    # ----- Embeddings -----
    def _maybe_start_parent_embedding(self, rec: MemoryEntry) -> None:
        if not self._embed_enabled:
            return
        # We embed per parent program, once.
        parent_id = str(rec.parent_program_id)
        with self._lock:
            if parent_id in self._parent_embed or parent_id in self._parent_embed_pending:
                return
            # Fetch code string from generator_input
            code_str = None
            try:
                if isinstance(rec.generator_input, dict):
                    code_str = rec.generator_input.get("code")
                elif isinstance(rec.generator_input, str):
                    code_str = rec.generator_input
            except Exception:
                code_str = None
            if not code_str or not isinstance(code_str, str) or len(code_str.strip()) == 0:
                return
            self._parent_embed_pending.add(parent_id)
        t = threading.Thread(target=self._embed_parent_worker, args=(parent_id, code_str), daemon=True)
        t.start()

    def _embed_parent_worker(self, parent_id: str, code_str: str) -> None:
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, timeout=20.0)
            except Exception:
                logger.error("OpenAI client import/init failed for embeddings", exc_info=True)
                return
            # Truncate to a safe length; embeddings have input limits
            text = code_str
            if len(text) > 200_000:
                text = text[:200_000]
            resp = client.embeddings.create(model=self._embed_model, input=text)
            vec = resp.data[0].embedding if getattr(resp, "data", None) else None
            if not vec:
                return
            with self._lock:
                self._parent_embed[parent_id] = list(vec)
                if parent_id in self._parent_embed_pending:
                    self._parent_embed_pending.remove(parent_id)
            self._snapshot_write_embeddings()
        except Exception:
            logger.error("Parent embedding computation failed", exc_info=True)
            with self._lock:
                if parent_id in self._parent_embed_pending:
                    self._parent_embed_pending.remove(parent_id)

    # Index helpers
    def _index_basic(self, rec: MemoryEntry) -> None:
        pid = str(rec.parent_program_id)
        cid = str(rec.child_program_id)
        it = rec.iteration
        self._idx_parent.setdefault(pid, set()).add(rec.id)
        self._idx_child.setdefault(cid, set()).add(rec.id)
        if it is not None:
            self._idx_iter.setdefault(int(it), set()).add(rec.id)

    def _index_status_and_metrics(self, rec: MemoryEntry) -> None:
        # status
        try:
            status = self._derive_status(rec)
            if status:
                self._idx_sel.setdefault("status", {}).setdefault(status, set()).add(rec.id)
        except Exception:
            logger.error("Indexing status failed", exc_info=True)
        # generic metric buckets from validator_output
        try:
            if isinstance(rec.validator_output, dict):
                self._index_metric_buckets_for(rec.id, rec.validator_output)
                # also mirror combined_score as score_bucket if present
                if isinstance(rec.validator_output.get("combined_score"), (int, float)):
                    sb = f"{round(float(rec.validator_output['combined_score']), 1):.1f}"
                    self._idx_score_bucket.setdefault(sb, set()).add(rec.id)
            # parent metrics and deltas (child - parent)
            parent_metrics = None
            try:
                if isinstance(rec.generator_input, dict) and isinstance(rec.generator_input.get("metrics"), dict):
                    parent_metrics = rec.generator_input.get("metrics")
            except Exception:
                parent_metrics = None
            if isinstance(parent_metrics, dict):
                # index parent buckets under metric_parent:<name>_bucket
                self._index_metric_buckets_for(rec.id, parent_metrics, prefix="metric_parent")
                # compute deltas when child metrics exist
                if isinstance(rec.validator_output, dict):
                    deltas: Dict[str, Any] = {}
                    for k, v in rec.validator_output.items():
                        pv = parent_metrics.get(k)
                        if isinstance(v, (int, float)) and isinstance(pv, (int, float)):
                            try:
                                deltas[k] = float(v) - float(pv)
                            except Exception:
                                pass
                    if deltas:
                        self._index_metric_buckets_for(rec.id, deltas, prefix="metric_delta")
            # failure signature from validator_output
            try:
                if isinstance(rec.validator_output, dict):
                    err = rec.validator_output.get("error")
                    if isinstance(err, str) and err:
                        fsig = err.split(":", 1)[0].strip()
                        if fsig:
                            self._idx_failure_sig.setdefault(fsig, set()).add(rec.id)
            except Exception:
                pass
            # score/accuracy buckets from validator_output
            try:
                if isinstance(rec.validator_output, dict):
                    if isinstance(rec.validator_output.get("score"), (int, float)):
                        sb = f"{round(float(rec.validator_output['score']), 1):.1f}"
                        self._idx_score_bucket.setdefault(sb, set()).add(rec.id)
                    if isinstance(rec.validator_output.get("accuracy"), (int, float)):
                        ab = f"{round(float(rec.validator_output['accuracy']), 1):.1f}"
                        self._idx_accuracy_bucket.setdefault(ab, set()).add(rec.id)
            except Exception:
                pass
        except Exception:
            logger.error("Indexing generic metric buckets failed", exc_info=True)

    def _index_metric_buckets_for(self, rec_id: str, metrics: Dict[str, Any], prefix: str = "metric") -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                key = f"{prefix}:{k}_bucket"
                bucket = f"{round(float(v), 1):.1f}"
                self._idx_metric_bucket.setdefault(key, {}).setdefault(bucket, set()).add(rec_id)

    def _derive_status(self, rec: MemoryEntry) -> Optional[str]:
        # success if no error present; else fail
        try:
            if isinstance(rec.validator_output, dict) and rec.validator_output.get("error"):
                return "fail"
            # also check synopsis status if present
            if isinstance(rec.synopsis_ai, dict) and rec.synopsis_ai.get("status"):
                val = str(rec.synopsis_ai.get("status")).lower()
                return "fail" if "fail" in val else "success"
            return "success"
        except Exception:
            return None

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
            # remove status
            try:
                status = self._derive_status(rec)
                if status and "status" in self._idx_sel and status in self._idx_sel["status"]:
                    self._idx_sel["status"][status].discard(rec.id)
            except Exception:
                pass
            # remove metric buckets (from validator_output, parent metrics, and deltas)
            try:
                if isinstance(rec.validator_output, dict):
                    # Remove validator_output metric buckets
                    for k, v in rec.validator_output.items():
                        if isinstance(v, (int, float)):
                            key = f"metric:{k}_bucket"
                            bucket = f"{round(float(v), 1):.1f}"
                            if key in self._idx_metric_bucket and bucket in self._idx_metric_bucket[key]:
                                self._idx_metric_bucket[key][bucket].discard(rec.id)
                # Remove parent metric buckets
                parent_metrics = None
                if isinstance(rec.generator_input, dict) and isinstance(rec.generator_input.get("metrics"), dict):
                    parent_metrics = rec.generator_input.get("metrics")
                if isinstance(parent_metrics, dict):
                    for k, v in parent_metrics.items():
                        if isinstance(v, (int, float)):
                            key = f"metric_parent:{k}_bucket"
                            bucket = f"{round(float(v), 1):.1f}"
                            if key in self._idx_metric_bucket and bucket in self._idx_metric_bucket[key]:
                                self._idx_metric_bucket[key][bucket].discard(rec.id)
                    # Remove delta metric buckets
                    if isinstance(rec.validator_output, dict):
                        for k, v in rec.validator_output.items():
                            pv = parent_metrics.get(k)
                            if isinstance(v, (int, float)) and isinstance(pv, (int, float)):
                                try:
                                    d = float(v) - float(pv)
                                    key = f"metric_delta:{k}_bucket"
                                    bucket = f"{round(d, 1):.1f}"
                                    if key in self._idx_metric_bucket and bucket in self._idx_metric_bucket[key]:
                                        self._idx_metric_bucket[key][bucket].discard(rec.id)
                                except Exception:
                                    pass
            except Exception:
                pass
            # remove failure signature, score/accuracy buckets
            try:
                if isinstance(rec.validator_output, dict):
                    err = rec.validator_output.get("error")
                    if isinstance(err, str) and err:
                        fsig = err.split(":", 1)[0].strip()
                        if fsig and fsig in self._idx_failure_sig:
                            self._idx_failure_sig[fsig].discard(rec.id)
                    if isinstance(rec.validator_output.get("score"), (int, float)):
                        sb = f"{round(float(rec.validator_output['score']), 1):.1f}"
                        if sb in self._idx_score_bucket:
                            self._idx_score_bucket[sb].discard(rec.id)
                    if isinstance(rec.validator_output.get("accuracy"), (int, float)):
                        ab = f"{round(float(rec.validator_output['accuracy']), 1):.1f}"
                        if ab in self._idx_accuracy_bucket:
                            self._idx_accuracy_bucket[ab].discard(rec.id)
                    if isinstance(rec.validator_output.get("combined_score"), (int, float)):
                        sb = f"{round(float(rec.validator_output['combined_score']), 1):.1f}"
                        if sb in self._idx_score_bucket:
                            self._idx_score_bucket[sb].discard(rec.id)
            except Exception:
                pass
        except Exception:
            pass

    # Snapshot helper (for simple UI)
    def _snapshot_write(self) -> None:
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
                # derive status if available
                try:
                    status = self._derive_status(rec)
                    if status:
                        row["status"] = status
                except Exception:
                    pass
                data.append(row)
        # Write atomically
        try:
            tmp = self._snapshot_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"entries": data}, f, ensure_ascii=False)
            os.replace(tmp, self._snapshot_path)
        except Exception:
            logger.error("Writing memory UI snapshot failed", exc_info=True)
            return

        # Try writing embeddings file as well
        try:
            self._snapshot_write_embeddings()
        except Exception:
            # Non-fatal
            pass

    def _snapshot_write_embeddings(self) -> None:
        if not self._snapshot_path:
            return
        try:
            index_dir = os.path.dirname(self._snapshot_path)
            emb_path = os.path.join(index_dir, "memory_embeddings.json")
            with self._lock:
                payload = {
                    "model": self._embed_model,
                    "parents": [
                        {"parent": pid, "vector": vec}
                        for pid, vec in self._parent_embed.items()
                    ],
                }
            tmp = emb_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, emb_path)
        except Exception:
            logger.error("Writing memory embeddings snapshot failed", exc_info=True)
