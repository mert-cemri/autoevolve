from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid


@dataclass
class MemoryRecord:
    """Canonical memory unit stored in MemoryStore.

    Fields:
    - id: unique ID
    - namespace: logical grouping (e.g., "openevolve", "custom_search", "general")
    - kind: record type (e.g., "program", "artifact", "insight", "run", "error", "prompt")
    - tags: lightweight filtering (e.g., ["island:0", "beam:2", "iteration:5"]) 
    - payload: freeform JSON-serializable dict
    - created_at/updated_at: unix timestamps
    - parent_id: optional link for threading/causality (e.g., child of a program)
    - run_id: optional run/session identifier
    - key: optional stable key for upsert/dedup semantics
    - relations: optional graph-like edges (e.g., [{"type":"derived_from","id":"..."}])
    """

    namespace: str
    kind: str
    payload: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None

    # Optional, future-proofing
    run_id: Optional[str] = None
    key: Optional[str] = None
    relations: List[Dict[str, Any]] = field(default_factory=list)

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())

    def update_payload(self, updates: Dict[str, Any]) -> None:
        self.payload.update(updates)
        self.updated_at = time.time()

    def add_tags(self, new_tags: List[str]) -> None:
        existing = set(self.tags)
        for t in new_tags:
            if t not in existing:
                self.tags.append(t)
        self.updated_at = time.time()
