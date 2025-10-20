import os
import sys
import time

# Ensure project root is on sys.path when running directly
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from autoevolve.memory import InMemoryMemoryStore, MemoryEntry


def main() -> None:
    # Ensure snapshot path for UI (defaults if not provided)
    if not os.environ.get("MEMORY_SNAPSHOT_PATH"):
        os.environ["MEMORY_SNAPSHOT_PATH"] = "/tmp/memory_snapshot.json"
    mem = InMemoryMemoryStore()
    
    P0_CODE = (
        "def count_chars(s: str) -> int:\n"
        "    # baseline stub\n"
        "    return 0\n"
    )
    P1_CODE = (
        "def count_chars_loop(s: str) -> int:\n"
        "    c = 0\n"
        "    for _ in s:\n"
        "        c += 1\n"
        "    return c\n"
    )
    P2_CODE = (
        "def count_chars_len(s: str) -> int:\n"
        "    return len(s)\n"
    )
    P3_CODE = (
        "def count_chars_sum(s: str) -> int:\n"
        "    return sum(1 for _ in s)\n"
    )
    P4_CODE = (
        "def count_chars_loop_fast(s: str) -> int:\n"
        "    total = 0\n"
        "    it = s\n"
        "    for _ in it:\n"
        "        total += 1\n"
        "    return total\n"
    )
    P5_CODE = (
        "import unicodedata\n"
        "def count_chars_unicode(s: str) -> int:\n"
        "    n = unicodedata.normalize('NFC', s)\n"
        "    return len(n)\n"
    )

    # P0 -> P1: naive loop-based char count
    p0_p1 = MemoryEntry(
        parent_program_id="P0",
        child_program_id="P1",
        generator_input={"code": P0_CODE},
        generator_output={"code": P1_CODE},
        validator_output={"correctness": 1.0, "runtime_ms": 0.08, "complexity": "O(n)"},
        diff_summary_user="Baseline: explicit for-loop increments a counter.",
        iteration=1,
    )
    id_p1 = mem.add(p0_p1)
    print("added:", "P0 -> P1", id_p1)
    _tick()

    # P0 -> P2: built-in len
    p0_p2 = MemoryEntry(
        parent_program_id="P0",
        child_program_id="P2",
        generator_input={"code": P0_CODE},
        generator_output={"code": P2_CODE},
        validator_output={"correctness": 1.0, "runtime_ms": 0.01, "complexity": "O(1) API over O(n) impl"},
        diff_summary_user="Use Python's optimized len() for strings.",
        iteration=1,
    )
    id_p2 = mem.add(p0_p2)
    print("added:", "P0 -> P2", id_p2)
    _tick()

    # P0 -> P3: sum over generator
    p0_p3 = MemoryEntry(
        parent_program_id="P0",
        child_program_id="P3",
        generator_input={"code": P0_CODE},
        generator_output={"code": P3_CODE},
        validator_output={"correctness": 1.0, "runtime_ms": 0.03, "complexity": "O(n)"},
        diff_summary_user="Functional style using sum over a generator.",
        iteration=1,
    )
    id_p3 = mem.add(p0_p3)
    print("added:", "P0 -> P3", id_p3)
    _tick()

    # Second layer evolutions (two paths)
    # P1 -> P4: micro-opt on loop (local var caching)
    p1_p4 = MemoryEntry(
        parent_program_id="P1",
        child_program_id="P4",
        generator_input={"code": P1_CODE},
        generator_output={"code": P4_CODE},
        validator_output={"correctness": 1.0, "runtime_ms": 0.07, "complexity": "O(n)"},
        diff_summary_user="Minor loop micro-optimization; same complexity.",
        iteration=2,
    )
    id_p4 = mem.add(p1_p4)
    print("added:", "P1 -> P4", id_p4)
    _tick()

    # P2 -> P5: Unicode normalization + len (success with buckets)
    p2_p5 = MemoryEntry(
        parent_program_id="P2",
        child_program_id="P5",
        generator_input={"code": P2_CODE},
        generator_output={"code": P5_CODE},
        validator_output={"correctness": 1.0, "runtime_ms": 0.02, "handles_unicode": True, "score": 0.92, "accuracy": 0.91},
        diff_summary_user="Normalize to NFC before len() for robust Unicode handling.",
        iteration=2,
    )
    id_p5 = mem.add(p2_p5)
    print("added:", "P2 -> P5", id_p5)
    _tick()

    # P3 -> P6: introduce a failing change (to populate failure_signature)
    BAD_CODE = (
        "def count_chars_bad(s: str) -> int:\n"
        "    return len(x)  # NameError\n"
    )
    p3_p6 = MemoryEntry(
        parent_program_id="P3",
        child_program_id="P6",
        generator_input={"code": P3_CODE},
        generator_output={"code": BAD_CODE},
        validator_output={"error": "NameError: name 'x' is not defined"},
        diff_summary_user="Incorrect variable name leads to NameError.",
        iteration=2,
    )
    id_p6 = mem.add(p3_p6)
    print("added:", "P3 -> P6 (fail)", id_p6)
    _tick()

    # Queries: show evolution fan-out from P0, and next steps from P1/P2
    p0_children = mem.search(filter_eq={"parent": "P0"}, limit=10)
    print("children of P0:", [e.child_program_id for e in p0_children])

    p1_children = mem.search(filter_eq={"parent": "P1"}, limit=10)
    print("children of P1:", [e.child_program_id for e in p1_children])

    p2_children = mem.search(filter_eq={"parent": "P2"}, limit=10)
    print("children of P2:", [e.child_program_id for e in p2_children])

    iter1 = mem.search(filter_eq={"iteration": 1}, limit=10)
    iter2 = mem.search(filter_eq={"iteration": 2}, limit=10)
    print("iteration=1 steps:", [(e.parent_program_id, e.child_program_id) for e in iter1])
    print("iteration=2 steps:", [(e.parent_program_id, e.child_program_id) for e in iter2])

    # Show supported search keys
    print("supported search keys:", mem.list_search_keys())

    # Show synopses for a couple of steps
    print("synopsis (P1->P4):")
    _wait_and_print_synopsis(mem, id_p4)
    print("synopsis (P2->P5):")
    _wait_and_print_synopsis(mem, id_p5)
    print("synopsis (P3->P6):")
    _wait_and_print_synopsis(mem, id_p6)



def _wait_and_print_synopsis(mem: InMemoryMemoryStore, rid: str, timeout_s: float = 6.0, poll_s: float = 0.2) -> None:
    start = time.time()
    synopsis = mem.get(rid).synopsis_ai
    while synopsis is None and (time.time() - start) < timeout_s:
        time.sleep(poll_s)
        synopsis = mem.get(rid).synopsis_ai
    print("  synopsis:\n", synopsis if synopsis else "(pending)")


def _tick(delay: float = 0.4) -> None:
    # Small pause so the UI snapshot and async synopsis can catch up
    time.sleep(delay)

    
if __name__ == "__main__":
    main()




