from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .schemas import MemoryEntry


class MemoryStore(ABC):
    """Minimal memory store: insert, get, delete, search by filters."""

    @abstractmethod
    def add(self, record: MemoryEntry) -> str:
        """Add a record and return its id."""
        raise NotImplementedError

    @abstractmethod
    def get(self, record_id: str) -> Optional[MemoryEntry]:
        """Fetch a record by id."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """Delete a record by id; return True if deleted."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        filter_eq: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[MemoryEntry]:
        """Search by exact-match on selected payload fields (store-defined)."""
        raise NotImplementedError

    @abstractmethod
    def list_search_keys(self) -> List[str]:
        """Return the supported exact-match filter keys."""
        raise NotImplementedError

    # Bulk helpers
    def upsert(self, record: MemoryEntry) -> str:
        """Not supported in minimal API; use add()."""
        return self.add(record)
