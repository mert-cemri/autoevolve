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
    st.title("Live Memory Viewer")

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
    all_tags = sorted({t for e in entries for t in (e.get("tags") or [])})
    # Synopsis-derived facets
    all_selectors: Dict[str, set] = {}
    failure_sigs = set()
    score_buckets = set()
    accuracy_buckets = set()
    for e in entries:
        syn = e.get("synopsis") or {}
        sels = syn.get("selectors") or []
        if isinstance(sels, list):
            for sel in sels:
                if isinstance(sel, dict):
                    k = str(sel.get("key"))
                    v = sel.get("value")
                    if k:
                        all_selectors.setdefault(k, set()).add(v)
        fs = syn.get("failure_signature")
        if not fs:
            # Fallback from validator_output
            err = (e.get("validator_output") or {}).get("error")
            if isinstance(err, str) and err:
                fs = err.split(":", 1)[0].strip()
        if isinstance(fs, str) and fs:
            failure_sigs.add(fs)
        sb = syn.get("score_bucket")
        if not sb:
            sc = (e.get("validator_output") or {}).get("score")
            if isinstance(sc, (int, float)):
                try:
                    sb = f"{round(float(sc), 1):.1f}"
                except Exception:
                    sb = None
        if isinstance(sb, str) and sb:
            score_buckets.add(sb)
        ab = syn.get("accuracy_bucket")
        if not ab:
            ac = (e.get("validator_output") or {}).get("accuracy")
            if isinstance(ac, (int, float)):
                try:
                    ab = f"{round(float(ac), 1):.1f}"
                except Exception:
                    ab = None
        if isinstance(ab, str) and ab:
            accuracy_buckets.add(ab)

    sel_parent = st.sidebar.multiselect("Filter parent", parents)
    sel_child = st.sidebar.multiselect("Filter child", children)
    sel_iter = st.sidebar.multiselect("Filter iteration", iterations)
    sel_tag = st.sidebar.multiselect("Filter tag", all_tags)
    # Selector filters
    selector_key = st.sidebar.selectbox("Selector key", ["(none)"] + sorted(all_selectors.keys()))
    selector_vals: List[Any] = []
    if selector_key and selector_key != "(none)":
        selector_vals = st.sidebar.multiselect(
            f"Selector values for {selector_key}", sorted(list(all_selectors.get(selector_key, set())))
        )
    # Failure and buckets
    sel_failure = st.sidebar.multiselect("Failure signature", sorted(failure_sigs))
    sel_score_bucket = st.sidebar.multiselect("Score bucket", sorted(score_buckets))
    sel_accuracy_bucket = st.sidebar.multiselect("Accuracy bucket", sorted(accuracy_buckets))

    def passes(e: Dict[str, Any]) -> bool:
        if sel_parent and e.get("parent") not in sel_parent:
            return False
        if sel_child and e.get("child") not in sel_child:
            return False
        if sel_iter and e.get("iteration") not in sel_iter:
            return False
        if sel_tag:
            tags = set(e.get("tags") or [])
            if not any(t in tags for t in sel_tag):
                return False
        syn = e.get("synopsis") or {}
        # selectors
        if selector_key and selector_key != "(none)" and selector_vals:
            ok = False
            sels = syn.get("selectors") or []
            if isinstance(sels, list):
                for sel in sels:
                    if isinstance(sel, dict) and str(sel.get("key")) == selector_key and sel.get("value") in selector_vals:
                        ok = True
                        break
            if not ok:
                return False
        # failure
        if sel_failure:
            fs = syn.get("failure_signature")
            if not fs:
                err = (e.get("validator_output") or {}).get("error")
                if isinstance(err, str) and err:
                    fs = err.split(":", 1)[0].strip()
            if fs not in sel_failure:
                return False
        # buckets
        if sel_score_bucket:
            sb = syn.get("score_bucket")
            if not sb:
                sc = (e.get("validator_output") or {}).get("score")
                if isinstance(sc, (int, float)):
                    try:
                        sb = f"{round(float(sc), 1):.1f}"
                    except Exception:
                        sb = None
            if sb not in sel_score_bucket:
                return False
        if sel_accuracy_bucket:
            ab = syn.get("accuracy_bucket")
            if not ab:
                ac = (e.get("validator_output") or {}).get("accuracy")
                if isinstance(ac, (int, float)):
                    try:
                        ab = f"{round(float(ac), 1):.1f}"
                    except Exception:
                        ab = None
            if ab not in sel_accuracy_bucket:
                return False
        return True

    filt = [e for e in entries if passes(e)]

    # Left: index lists; Right: nodes table
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Indexes")
        st.write({"parents": parents})
        st.write({"iterations": iterations})
        st.write({"tags": all_tags})
        # Show selector keys/values present in dataset
        sel_summary = {}
        for e in entries:
            syn = e.get("synopsis") or {}
            sels = syn.get("selectors") or []
            if isinstance(sels, list):
                for sel in sels:
                    if isinstance(sel, dict):
                        k = str(sel.get("key"))
                        v = sel.get("value")
                        if k:
                            sel_summary.setdefault(k, set()).add(v)
        st.write({k: sorted(list(vs)) for k, vs in sel_summary.items()})

    # Table
    table = [
        {
            "id": e.get("id"),
            "parent": e.get("parent"),
            "child": e.get("child"),
            "iter": e.get("iteration"),
            "tags": ", ".join(e.get("tags") or []),
        }
        for e in filt
    ]
    with right:
        st.subheader("Nodes")
        st.dataframe(table, use_container_width=True, height=360)

    # Detail panel
    sel_id = st.selectbox("Select node id", [e.get("id") for e in filt]) if filt else None
    if sel_id:
        entry = next((e for e in entries if e.get("id") == sel_id), None)
        if entry:
            st.subheader(f"Node {sel_id}")
            synopsis = entry.get("synopsis") or {}
            st.markdown("**Overview**")
            st.write(synopsis.get("overview"))
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Delta**")
                st.write(synopsis.get("delta_summary_structured"))
                st.markdown("**Causal Links**")
                st.write(synopsis.get("causal_links"))
                st.markdown("**Pitfalls**")
                st.write(synopsis.get("pitfalls"))
            with c2:
                st.markdown("**Validator**")
                st.write(synopsis.get("validator_summary_structured"))
                st.markdown("**Tags**")
                st.write(synopsis.get("tags"))
                st.markdown("**Selectors**")
                st.write(synopsis.get("selectors"))

            st.markdown("**Raw Payload**")
            raw_tabs = st.tabs(["generator_input", "generator_output", "validator_output", "diff_summary_user", "generator_prompt", "metadata"])
            raw_tabs[0].write(entry.get("generator_input"))
            raw_tabs[1].write(entry.get("generator_output"))
            raw_tabs[2].write(entry.get("validator_output"))
            raw_tabs[3].write(entry.get("diff_summary_user"))
            raw_tabs[4].write(entry.get("generator_prompt"))
            raw_tabs[5].write(entry.get("metadata"))

    # Auto-refresh
    st.caption(f"Auto-refreshing every {refresh_sec}s from {path}")
    time.sleep(refresh_sec)
    try:
        st.rerun()
    except Exception:
        pass


if __name__ == "__main__":
    main()


