import os
import sys

# Ensure project root is on sys.path when running this file directly
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from autoevolve.memory import InMemoryMemoryStore, MemoryRecord


def simulate_beam_loop(iterations: int = 2, beam_width: int = 2, branch_factor: int = 4):
    print("\n=== Beam Search Simulation (clean loop) ===")
    mem = InMemoryMemoryStore()
    run_id = "beam_run_loop"
    strategy_tag = "strategy:beam_search"

    def get_selected(iter_idx: int):
        sel = mem.search(
            namespace="custom_search",
            kinds=["beam_selected"],
            tags_all=[f"iteration:{iter_idx}", strategy_tag],
            limit=1,
        )
        if not sel:
            return []
        return sel[0].payload.get("selected", [])

    def label_for(rid: str) -> str:
        rec = mem.get(rid)
        return rec.payload.get("program_id", rid) if rec else rid

    # Iter 0: seed P0
    seed = MemoryRecord(
        namespace="custom_search",
        kind="program",
        payload={"program_id": "P0", "score": 0.50},
        tags=["strategy:beam_search", "iteration:0"],
        run_id=run_id,
        key="custom_search:program:P0",
    )
    id_P0 = mem.add(seed)
    print("iter0: seed = ['P0'] (written to memory as a program record)")

    # Beam loop
    for t in range(1, iterations + 1):
        # Determine parents to expand
        parents = [id_P0] if t == 1 else get_selected(t - 1)
        parent_labels = [label_for(rid) for rid in parents]
        branches_per_member = max(1, branch_factor // max(1, len(parents)))

        print(
            f"iter{t}: expand parents={parent_labels} · {branches_per_member} branches each → select top {beam_width}"
        )

        # Expand parents → candidates
        candidates = []  # list of (record_id, program_id, score)
        idx = 0
        for p_idx, parent_rid in enumerate(parents):
            parent_label = parent_labels[p_idx]
            for b in range(branches_per_member):
                # Deterministic score progression for demo
                score = 0.60 + 0.05 * t + 0.01 * (p_idx * branches_per_member + b)
                prog_id = f"I{t}_{parent_label}_C{idx+1}"
                rid = mem.add(
                    MemoryRecord(
                        namespace="custom_search",
                        kind="program",
                        payload={"program_id": prog_id, "score": score},
                        tags=["strategy:beam_search", f"iteration:{t}", f"parent:{parent_label}"],
                        run_id=run_id,
                        relations=[{"type": "derived_from", "id": parent_rid}],
                    )
                )
                candidates.append((rid, prog_id, score))
                idx += 1

        # Select top beam_width by score
        candidates.sort(key=lambda x: x[2], reverse=True)
        selected = candidates[:beam_width]
        selected_ids = [rid for rid, _, _ in selected]

        # Store selection summary
        mem.add(
            MemoryRecord(
                namespace="custom_search",
                kind="beam_selected",
                payload={
                    "iteration": t,
                    "selected": selected_ids,
                    "beam_width": beam_width,
                    "expanded": len(candidates),
                    "parents": parents,
                },
                tags=["strategy:beam_search", f"iteration:{t}"],
                run_id=run_id,
            )
        )

        # Pretty prints
        cands_pretty = [f"{pid}:{round(sc,3)}" for _, pid, sc in candidates]
        selected_pretty = [f"{pid}:{round(sc,3)}" for _, pid, sc in selected]
        print(f"       candidates=({len(candidates)}): {cands_pretty}")
        print(f"       selected=({beam_width}): {selected_pretty} (stored selection in memory)\n")

    # Final beam summary
    final_sel_ids = get_selected(iterations)
    final = [(label_for(rid), round(mem.get(rid).payload.get('score', 0.0), 3)) for rid in final_sel_ids if mem.get(rid)]
    print(f"final beam: {final}")


def simulate_insight_guided(iterations: int = 3, children_per_parent: int = 2):
    print("\n=== Insight-Guided Evolution (per-child guidance) ===")
    mem = InMemoryMemoryStore()
    run_id = "insight_run"

    def fetch_rules_for(label: str):
        recs = mem.search(
            namespace="custom_search",
            kinds=["insight"],
            tags_all=[f"for:{label}"],
            limit=10,
        )
        # Aggregate rules from all insights targeting this label
        rules = []
        for r in recs:
            rules.extend(r.payload.get("rules", []))
        return rules

    def child_bonus(child_label: str, rules):
        # Example rule: {"kind":"favor_child_idx","idx":2,"bonus":0.02}
        bonus = 0.0
        # Parse child index from suffix "..._Ck"
        idx = None
        if "_C" in child_label:
            try:
                idx = int(child_label.split("_C")[-1])
            except Exception:
                idx = None
        for rule in rules:
            if rule.get("kind") == "favor_child_idx" and idx is not None:
                if int(rule.get("idx", -1)) == idx:
                    bonus += float(rule.get("bonus", 0.0))
        return round(bonus, 3)

    # Seed parent program P0
    parent_label = "P0"
    parent_score = 0.50
    mem.add(
        MemoryRecord(
            namespace="custom_search",
            kind="program",
            payload={"program_id": parent_label, "score": parent_score},
            tags=["iteration:0"],
            run_id=run_id,
            key=f"custom_search:program:{parent_label}",
        )
    )
    print(f"iter0: seed parent='{parent_label}' score={parent_score} (program written)")

    for t in range(1, iterations + 1):
        # Retrieve structured rules from memory for current parent
        rules = fetch_rules_for(parent_label)
        print(f"iter{t}: parent='{parent_label}' base_score={round(parent_score,3)} · rules={rules}")

        # Generate children; apply per-child bonus based on rules
        children = []  # (label, base, bonus, adjusted)
        for i in range(children_per_parent):
            base = round(parent_score + 0.05 + 0.01 * i, 3)
            child_label = f"I{t}_{parent_label}_C{i+1}"
            bonus = child_bonus(child_label, rules)
            adjusted = round(base + bonus, 3)
            mem.add(
                MemoryRecord(
                    namespace="custom_search",
                    kind="program",
                    payload={
                        "program_id": child_label,
                        "score": adjusted,
                        "applied_rules": rules,
                        "bonus": bonus,
                    },
                    tags=[f"iteration:{t}", f"parent:{parent_label}"],
                    run_id=run_id,
                )
            )
            children.append((child_label, base, bonus, adjusted))
        child_str = [f"{lbl}: base={b} + bonus={bn} → {adj}" for (lbl, b, bn, adj) in children]
        print(f"       children=({len(children)}): {child_str}")

        # Select best child as next parent (greedy path)
        best_child = max(children, key=lambda x: x[3])
        next_parent_label, _, _, next_parent_score = best_child
        improvement = round(next_parent_score - parent_score, 3)

        # Emit a new insight with a rule that favors a specific child index next time
        # Simple heuristic: if improvement small, favor child 2 next; else favor child 1
        favored_idx = 2 if improvement < 0.06 else 1
        new_rules = [{"kind": "favor_child_idx", "idx": favored_idx, "bonus": 0.02}]
        mem.add(
            MemoryRecord(
                namespace="custom_search",
                kind="insight",
                payload={"rules": new_rules, "from_parent": parent_label, "improvement": improvement},
                tags=[f"for:{next_parent_label}", f"iteration:{t}"],
                run_id=run_id,
            )
        )
        print(
            f"       selected='{next_parent_label}' score={next_parent_score} · improvement={improvement} → wrote_insight rules={new_rules} for next parent"
        )

        # Move to next
        parent_label, parent_score = next_parent_label, next_parent_score

    print(f"final parent: '{parent_label}' score={round(parent_score,3)}")


if __name__ == "__main__":
    # Beam search demo
    simulate_beam_loop(iterations=2, beam_width=2, branch_factor=4)
    # Insight-guided demo
    simulate_insight_guided(iterations=3, children_per_parent=2)
    print("OK\n")
