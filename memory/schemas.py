from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time
import uuid


@dataclass
class MemoryEntry:
    """Single schema for evolution/search memory.

    Minimal, semi-structured payload from the user program.
    The store derives searchable keys (no user-provided indexes).
    """

    parent_program_id: str
    child_program_id: str
    generator_input: Any
    generator_output: Any
    validator_output: Any
    diff_summary_user: Optional[str] = None  # optional user-provided diff summary
    synopsis_ai: Optional[Dict[str, Any]] = None  # AI-generated structured synopsis dict

    # Optional: capture the exact generation prompt used for child code
    # (model name should live under metadata)
    generator_prompt: Optional[Dict[str, Any]] = None  # e.g., {"system":..., "user":...}

    # Optional step metadata
    iteration: Optional[int] = None

    # Optional simple buckets for any extra details:
    # - sampling: how the example was sampled/chosen by the generator
    # - metadata: any run/context notes (strategy, island, hashes, etc.)
    sampling: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    # Gradient evolution fields (for gradient-based parent selection and memory retrieval)
    distance: Optional[float] = None  # Semantic distance: 1 - similarity(parent, child)
    gradient: Optional[float] = None  # Normalized improvement: delta_score / distance

    # System-managed identifiers and timestamps
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
