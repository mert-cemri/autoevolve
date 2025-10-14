from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .schemas import MemoryRecord


class MemoryStore(ABC):
    """Abstract interface for a pluggable memory store.

    Design goals:
    - Pluggable across OpenEvolve and custom_search
    - Simple CRUD and flexible search
    - Namespaces and tags for fast filtering
    - Optional lightweight indexing hooks (override on implementations)
    """

    @abstractmethod
    def add(self, record: MemoryRecord) -> str:
        """Add a record and return its id."""
        raise NotImplementedError

    @abstractmethod
    def get(self, record_id: str) -> Optional[MemoryRecord]:
        """Fetch a record by id."""
        raise NotImplementedError

    @abstractmethod
    def update(self, record_id: str, updates: Dict[str, Any] = None, tags: List[str] = None) -> bool:
        """Update payload and/or tags; return True if updated."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """Delete a record by id; return True if deleted."""
        raise NotImplementedError

    @abstractmethod
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
        """Search records by namespace/kinds/tags and optional text query over payload.
        Implementations may override scoring/ordering (default: recency).
        """
        raise NotImplementedError

    # Optional bulk ops
    def add_many(self, records: Iterable[MemoryRecord]) -> List[str]:
        return [self.add(r) for r in records]

    def get_many(self, record_ids: Iterable[str]) -> List[MemoryRecord]:
        return [r for rid in record_ids if (r := self.get(rid)) is not None]
