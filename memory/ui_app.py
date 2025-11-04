import json
import os
import time
from typing import Any, Dict, List

import streamlit as st


def load_snapshot(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"entries": []}


def main() -> None:
    st.set_page_config(page_title="Memory Viewer", layout="wide")
    st.title("Live Memory Viewer (Simple Mode)")

    default_path = os.environ.get("MEMORY_SNAPSHOT_PATH", "/tmp/memory_snapshot.json")
    path = st.sidebar.text_input("Snapshot path", value=default_path)
    refresh_sec = st.sidebar.slider("Auto-refresh (sec)", 1, 10, 2)

    # Load
    data = load_snapshot(path)
    entries: List[Dict[str, Any]] = data.get("entries", [])

    # Derive filters
    parents = sorted({e.get("parent") for e in entries if e.get("parent")})
    children = sorted({e.get("child") for e in entries if e.get("child")})
    iterations = sorted({e.get("iteration") for e in entries if e.get("iteration") is not None})
    statuses = sorted({e.get("status") for e in entries if e.get("status")})

    # Failure signatures (from validator_output.error)
    failure_sigs = set()
    for e in entries:
        vo = e.get("validator_output") or {}
        err = vo.get("error")
        if isinstance(err, str) and err:
            fs = err.split(":", 1)[0].strip()
            if fs:
                failure_sigs.add(fs)
    failure_sigs = sorted(failure_sigs)

    # Score and accuracy buckets (from validator_output)
    score_buckets = set()
    accuracy_buckets = set()
    for e in entries:
        vo = e.get("validator_output") or {}
        # Prefer combined_score if present
        sc = vo.get("combined_score")
        if sc is None:
            sc = vo.get("score")
        if isinstance(sc, (int, float)):
            try:
                sb = f"{round(float(sc), 1):.1f}"
                score_buckets.add(sb)
            except Exception:
                pass
        ac = vo.get("accuracy")
        if isinstance(ac, (int, float)):
            try:
                ab = f"{round(float(ac), 1):.1f}"
                accuracy_buckets.add(ab)
            except Exception:
                pass

    # Dynamic metric bucket filters: child (validator_output), parent (generator_input.metrics), delta (child-parent)
    metric_bucket_map = {}
    parent_metric_bucket_map = {}
    delta_metric_bucket_map = {}
    for e in entries:
        vo = e.get("validator_output") or {}
        for k, v in vo.items():
            if isinstance(v, (int, float)):
                try:
                    b = f"{round(float(v), 1):.1f}"
                    metric_bucket_map.setdefault(k, set()).add(b)
                except Exception:
                    pass
        # parent metrics
        gi = e.get("generator_input") or {}
        pm = gi.get("metrics") if isinstance(gi, dict) else None
        if isinstance(pm, dict):
            for k, v in pm.items():
                if isinstance(v, (int, float)):
                    try:
                        b = f"{round(float(v), 1):.1f}"
                        parent_metric_bucket_map.setdefault(k, set()).add(b)
                    except Exception:
                        pass
            # delta (child - parent) for overlapping numeric keys
            for k, cv in vo.items():
                pv = pm.get(k)
                if isinstance(cv, (int, float)) and isinstance(pv, (int, float)):
                    try:
                        d = float(cv) - float(pv)
                        b = f"{round(d, 1):.1f}"
                        delta_metric_bucket_map.setdefault(k, set()).add(b)
                    except Exception:
                        pass

    # Filters
    sel_parent = st.sidebar.multiselect("Filter parent", parents)
    sel_child = st.sidebar.multiselect("Filter child", children)
    sel_iter = st.sidebar.multiselect("Filter iteration", iterations)
    sel_status = st.sidebar.multiselect("Filter status", statuses)
    sel_failure: List[str] = []
    if failure_sigs:
        sel_failure = st.sidebar.multiselect("Failure signature", failure_sigs)
    sel_score_bucket = []
    if score_buckets:
        sel_score_bucket = st.sidebar.multiselect("Combined score bucket", sorted(score_buckets))
    sel_accuracy_bucket = []
    if accuracy_buckets:
        sel_accuracy_bucket = st.sidebar.multiselect("Accuracy bucket", sorted(accuracy_buckets))

    # Child metric filter
    metric_names = sorted(metric_bucket_map.keys())
    metric_filter_name = st.sidebar.selectbox("Metric bucket (child: validator_output)", ["(none)"] + metric_names)
    metric_filter_vals: List[str] = []
    if metric_filter_name and metric_filter_name != "(none)":
        metric_filter_vals = st.sidebar.multiselect(
            f"Buckets for {metric_filter_name}", sorted(list(metric_bucket_map.get(metric_filter_name, set())))
        )

    # Parent metric filter (optional)
    parent_metric_filter_name = "(none)"
    parent_metric_filter_vals: List[str] = []
    parent_metric_names = sorted(parent_metric_bucket_map.keys())
    if parent_metric_names:
        parent_metric_filter_name = st.sidebar.selectbox("Parent metric bucket (generator_input.metrics)", ["(none)"] + parent_metric_names)
        if parent_metric_filter_name and parent_metric_filter_name != "(none)":
            parent_metric_filter_vals = st.sidebar.multiselect(
                f"Buckets for parent {parent_metric_filter_name}",
                sorted(list(parent_metric_bucket_map.get(parent_metric_filter_name, set())))
            )

    # Delta metric filter (optional)
    delta_metric_filter_name = "(none)"
    delta_metric_filter_vals: List[str] = []
    delta_metric_names = sorted(delta_metric_bucket_map.keys())
    if delta_metric_names:
        delta_metric_filter_name = st.sidebar.selectbox("Delta metric bucket (child - parent)", ["(none)"] + delta_metric_names)
        if delta_metric_filter_name and delta_metric_filter_name != "(none)":
            delta_metric_filter_vals = st.sidebar.multiselect(
                f"Buckets for delta {delta_metric_filter_name}",
                sorted(list(delta_metric_bucket_map.get(delta_metric_filter_name, set())))
            )

    def passes(e: Dict[str, Any]) -> bool:
        if sel_parent and e.get("parent") not in sel_parent:
            return False
        if sel_child and e.get("child") not in sel_child:
            return False
        if sel_iter and e.get("iteration") not in sel_iter:
            return False
        if sel_status and e.get("status") not in sel_status:
            return False
        # failure
        if sel_failure:
            vo = e.get("validator_output") or {}
            err = vo.get("error")
            fs = None
            if isinstance(err, str) and err:
                fs = err.split(":", 1)[0].strip()
            if fs not in sel_failure:
                return False
        # buckets
        if sel_score_bucket:
            vo = e.get("validator_output") or {}
            sc = vo.get("combined_score")
            if sc is None:
                sc = vo.get("score")
            bucket = None
            if isinstance(sc, (int, float)):
                try:
                    bucket = f"{round(float(sc), 1):.1f}"
                except Exception:
                    bucket = None
            if bucket not in sel_score_bucket:
                return False
        if sel_accuracy_bucket:
            vo = e.get("validator_output") or {}
            ac = vo.get("accuracy")
            bucket = None
            if isinstance(ac, (int, float)):
                try:
                    bucket = f"{round(float(ac), 1):.1f}"
                except Exception:
                    bucket = None
            if bucket not in sel_accuracy_bucket:
                return False
        # dynamic metric bucket filters (child, parent, delta)
        if metric_filter_name and metric_filter_name != "(none)" and metric_filter_vals:
            vo = e.get("validator_output") or {}
            val = vo.get(metric_filter_name)
            bucket = None
            if isinstance(val, (int, float)):
                try:
                    bucket = f"{round(float(val), 1):.1f}"
                except Exception:
                    bucket = None
            if bucket not in metric_filter_vals:
                return False
        if parent_metric_filter_name and parent_metric_filter_name != "(none)" and parent_metric_filter_vals:
            gi = e.get("generator_input") or {}
            pm = gi.get("metrics") if isinstance(gi, dict) else None
            bucket = None
            if isinstance(pm, dict):
                val = pm.get(parent_metric_filter_name)
                if isinstance(val, (int, float)):
                    try:
                        bucket = f"{round(float(val), 1):.1f}"
                    except Exception:
                        bucket = None
            if bucket not in parent_metric_filter_vals:
                return False
        if delta_metric_filter_name and delta_metric_filter_name != "(none)" and delta_metric_filter_vals:
            vo = e.get("validator_output") or {}
            gi = e.get("generator_input") or {}
            pm = gi.get("metrics") if isinstance(gi, dict) else None
            bucket = None
            if isinstance(pm, dict):
                cv = vo.get(delta_metric_filter_name)
                pv = pm.get(delta_metric_filter_name)
                if isinstance(cv, (int, float)) and isinstance(pv, (int, float)):
                    try:
                        d = float(cv) - float(pv)
                        bucket = f"{round(d, 1):.1f}"
                    except Exception:
                        bucket = None
            if bucket not in delta_metric_filter_vals:
                return False
        return True

    filt = [e for e in entries if passes(e)]

    # Load embeddings if available (same folder as snapshot)
    emb = None
    try:
        emb_path = os.path.join(os.path.dirname(path), "memory_embeddings.json")
        if os.path.exists(emb_path):
            with open(emb_path, "r", encoding="utf-8") as f:
                emb = json.load(f)
    except Exception:
        emb = None

    # Left: index lists; Right: nodes table
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Indexes")
        st.write({"parents": parents})
        st.write({"iterations": iterations})
        st.write({"statuses": statuses})

        with st.expander("Semantic search (parents)"):
            if not emb or not isinstance(emb.get("parents"), list):
                st.caption("No embeddings found yet. They are computed in the background when entries arrive.")
            else:
                q = st.text_area("Query (text or code)", height=80, placeholder="e.g., place small circles greedily, increase packing density by local swaps")
                api_key_override = st.text_input("OpenAI API key (optional)", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
                embed_model = st.text_input("Embedding model", value=os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large"))
                topk = st.slider("Top-K", 1, 20, 5)
                if st.button("Search", use_container_width=True):
                    try:
                        from openai import OpenAI
                        api_key_used = api_key_override or os.environ.get("OPENAI_API_KEY")
                        if not api_key_used:
                            raise RuntimeError("OPENAI_API_KEY not set; provide it above or export it before launching the UI")
                        client = OpenAI(api_key=api_key_used, timeout=20.0)
                        r = client.embeddings.create(model=embed_model, input=q or "")
                        qvec = r.data[0].embedding if getattr(r, "data", None) else None
                    except Exception as ex:
                        qvec = None
                        st.error(f"Embedding query failed: {ex}")
                    def _cos(a, b):
                        try:
                            import math
                            s = sum(x*y for x, y in zip(a, b))
                            na = math.sqrt(sum(x*x for x in a))
                            nb = math.sqrt(sum(y*y for y in b))
                            return s / (na * nb + 1e-12)
                        except Exception:
                            return 0.0
                    if qvec:
                        parent_vecs = [(p.get("parent"), p.get("vector")) for p in (emb.get("parents") or [])]
                        sims = []
                        for pid, vec in parent_vecs:
                            if isinstance(pid, str) and isinstance(vec, list) and len(vec) == len(qvec):
                                sims.append((pid, _cos(qvec, vec)))
                        sims.sort(key=lambda x: x[1], reverse=True)
                        sims = sims[:topk]
                        # summarize counts from current entries
                        counts = {}
                        for e in entries:
                            p = e.get("parent")
                            if p:
                                counts[p] = counts.get(p, 0) + 1
                        st.write([{"parent": pid, "similarity": round(score, 4), "entries": counts.get(pid, 0)} for pid, score in sims])

    # Table
    def _fmt_triplet(pm: Dict[str, Any], vo: Dict[str, Any], key: str) -> str:
        try:
            p = pm.get(key) if isinstance(pm, dict) else None
            c = vo.get(key)
            if isinstance(p, (int, float)) and isinstance(c, (int, float)):
                d = float(c) - float(p)
                return f"{p:.3f}, {c:.3f}, {d:+.3f}"
            if isinstance(c, (int, float)):
                return f"-, {c:.3f}, -"
            if isinstance(p, (int, float)):
                return f"{p:.3f}, -, -"
        except Exception:
            pass
        return "-"

    table = []
    for e in filt:
        vo = e.get("validator_output") or {}
        gi = e.get("generator_input") or {}
        pm = gi.get("metrics") if isinstance(gi, dict) else None
        row = {
            "id": e.get("id"),
            "parent": e.get("parent"),
            "child": e.get("child"),
            "iter": e.get("iteration"),
            "status": e.get("status"),
            "combined_score (p,c,Δ)": _fmt_triplet(pm or {}, vo, "combined_score"),
        }
        # Add common metrics if present
        if "sum_radii" in vo or (pm and "sum_radii" in pm):
            row["sum_radii (p,c,Δ)"] = _fmt_triplet(pm or {}, vo, "sum_radii")
        if "target_ratio" in vo or (pm and "target_ratio" in pm):
            row["target_ratio (p,c,Δ)"] = _fmt_triplet(pm or {}, vo, "target_ratio")
        table.append(row)
    with right:
        st.subheader("Nodes")
        st.dataframe(table, use_container_width=True, height=360)

    # Detail panel
    sel_id = st.selectbox("Select node id", [e.get("id") for e in filt]) if filt else None
    if sel_id:
        entry = next((e for e in entries if e.get("id") == sel_id), None)
        if entry:
            st.subheader(f"Node {sel_id}")
            st.markdown("**Raw Payload**")
            raw_tabs = st.tabs(["generator_input", "generator_output", "validator_output", "diff_summary_user", "generator_prompt", "metadata"])
            raw_tabs[0].write(entry.get("generator_input"))
            raw_tabs[1].write(entry.get("generator_output"))
            raw_tabs[2].write(entry.get("validator_output"))
            raw_tabs[3].write(entry.get("diff_summary_user"))
            raw_tabs[4].write(entry.get("generator_prompt"))
            raw_tabs[5].write(entry.get("metadata"))

            # Show derived fields
            st.markdown("**Derived Fields**")
            derived = {}
            if entry.get("status"):
                derived["status"] = entry.get("status")
            st.write(derived)

    # Auto-refresh
    st.caption(f"Auto-refreshing every {refresh_sec}s from {path}")
    time.sleep(refresh_sec)
    try:
        st.rerun()
    except Exception:
        pass


if __name__ == "__main__":
    main()
