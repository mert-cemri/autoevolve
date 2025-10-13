import json
import time
from typing import Any, Dict, List, Optional, Set

from .base import MemoryStore
from .schemas import MemoryRecord


class InMemoryMemoryStore(MemoryStore):
    """Simple in-memory implementation with basic indexing.

    Indexes:
    - by id (dict)
    - by namespace (dict[str, set[id]])
    - by tag (dict[str, set[id]])
    Text search:
    - naive substring search over serialized payload (lowercased)
    Ordering:
    - recency (updated_at desc)

    Notes:
    - Intended for single-process use; for multiprocess scenarios, prefer a process-safe backend (e.g., SQLite).
    - Upsert pattern: search by a stable `key` in payload or MemoryRecord.key, then update that record id.
    """

    def __init__(self) -> None:
        self._by_id: Dict[str, MemoryRecord] = {}
        self._by_ns: Dict[str, Set[str]] = {}
        self._by_tag: Dict[str, Set[str]] = {}

    def add(self, record: MemoryRecord) -> str:
        self._by_id[record.id] = record
        self._by_ns.setdefault(record.namespace, set()).add(record.id)
        for t in record.tags:
            self._by_tag.setdefault(t, set()).add(record.id)
        return record.id

    def get(self, record_id: str) -> Optional[MemoryRecord]:
        return self._by_id.get(record_id)

    def update(self, record_id: str, updates: Dict[str, Any] = None, tags: List[str] = None) -> bool:
        rec = self._by_id.get(record_id)
        if rec is None:
            return False
        if updates:
            rec.update_payload(updates)
        if tags:
            old_tags = set(rec.tags)
            rec.add_tags(tags)
            # index any newly added tags
            for t in set(rec.tags) - old_tags:
                self._by_tag.setdefault(t, set()).add(rec.id)
        rec.updated_at = time.time()
        return True

    def delete(self, record_id: str) -> bool:
        rec = self._by_id.pop(record_id, None)
        if rec is None:
            return False
        # deindex
        if rec.namespace in self._by_ns:
            self._by_ns[rec.namespace].discard(rec.id)
            if not self._by_ns[rec.namespace]:
                del self._by_ns[rec.namespace]
        for t in rec.tags:
            if t in self._by_tag:
                self._by_tag[t].discard(rec.id)
                if not self._by_tag[t]:
                    del self._by_tag[t]
        return True

    def search(
        self,
        namespace: Optional[str] = None,
        kinds: Optional[List[str]] = None,
        tags_any: Optional[List[str]] = None,
        tags_all: Optional[List[str]] = None,
        text_query: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        # Candidate set
        if namespace and namespace in self._by_ns:
            candidate_ids = set(self._by_ns[namespace])
        else:
            candidate_ids = set(self._by_id.keys())

        # Filter by kinds
        if kinds:
            candidate_ids = {
                rid for rid in candidate_ids if self._by_id[rid].kind in kinds
            }

        # Filter by tags_any
        if tags_any:
            tagged_ids = set()
            for t in tags_any:
                tagged_ids |= self._by_tag.get(t, set())
            candidate_ids &= tagged_ids if tags_any else candidate_ids

        # Filter by tags_all
        if tags_all:
            for t in tags_all:
                candidate_ids &= self._by_tag.get(t, set())

        # Filter by text_query over payload
        recs = [self._by_id[rid] for rid in candidate_ids]
        if text_query:
            q = text_query.lower()

            def payload_text(rec: MemoryRecord) -> str:
                try:
                    return json.dumps(rec.payload, ensure_ascii=False).lower()
                except Exception:
                    return ""

            recs = [r for r in recs if q in payload_text(r)]

        # Order by recency
        recs.sort(key=lambda r: r.updated_at, reverse=True)

        # Page
        return recs[offset: offset + limit]
