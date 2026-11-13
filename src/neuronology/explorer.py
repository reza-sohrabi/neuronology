"""Explorer-ready JSON shards (PLAN.md 12b, "Data for the explorer").

Layout (one directory per milestone and model, versioned path):

  <root>/v1/m3/<model>/index.json            list of neuron-poles with summary fields
  <root>/v1/m3/<model>/neurons/<key>.json    one file per neuron-pole

Public shards (committed under results/explorer/) contain only material that cannot
reveal the planted trigger or the unusual output before M4 is scored:
  - snippets from the openly licensed generic corpus (short, with per-token activations
    and the source dataset name for licence credit);
  - detection scores (numbers), attribution ranks and scores;
  - upstream drivers and readers as unit ids and numbers (no token strings);
and nothing from the transcripts. Descriptions are held back too, because a description
can name the trigger and that cannot be checked without the ground truth. Private shards
(written under artifacts/, which is gitignored, and on the volume) carry everything:
descriptions, transcript snippets (redacted), embedding tokens, logit-reader token ids.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neuronology.runinfo import write_json

PUBLIC_SNIPPET_BEFORE = 24
PUBLIC_SNIPPET_AFTER = 4
SCHEMA_VERSION = 1


def short_snippet(
    s: dict[str, Any], before: int = PUBLIC_SNIPPET_BEFORE, after: int = PUBLIC_SNIPPET_AFTER
) -> dict[str, Any]:
    c = s["center"]
    lo, hi = max(0, c - before), min(len(s["tokens"]), c + after + 1)
    return {
        "dataset": s.get("dataset"),
        "band": s.get("band"),
        "center": c - lo,
        "tokens": s["tokens"][lo:hi],
        "acts": [round(float(a), 3) for a in s["acts"][lo:hi]],
    }


def _strip_units(rows: list[dict[str, Any]] | None, n: int) -> list[dict[str, Any]]:
    keep = ("type", "layer", "neuron", "head", "mixer", "excess", "effect")
    return [{k: r[k] for k in keep if k in r} for r in (rows or [])[:n]]


def public_record(
    pole: dict[str, Any], desc: dict[str, Any] | None, n_snippets: int = 12
) -> dict[str, Any]:
    corpus = [s for s in pole.get("describe_corpus", []) if s.get("source") == "corpus"]
    corpus = sorted(corpus, key=lambda s: -s["max_act"])[:n_snippets]
    up = pole.get("upstream") or {}
    rd = pole.get("readers") or {}
    rec = {
        "schema": SCHEMA_VERSION,
        "key": pole["key"],
        "layer": pole["layer"],
        "neuron": pole["neuron"],
        "pole": pole["pole"],
        "ranks": pole.get("ranks", {}),
        "scores": pole.get("scores", {}),
        "scale": pole.get("scale"),
        "snippets": [short_snippet(s) for s in corpus],
        "upstream": {
            lane: _strip_units(up.get(lane), 20)
            for lane in ("gate_drivers", "gate_suppressors", "up_drivers", "up_suppressors")
            if up.get(lane)
        },
        "readers": _strip_units(rd.get("neurons"), 20),
        "description": None,
        "description_status": "withheld until M4 is scored",
    }
    if desc:
        rec["detection"] = {
            k: desc.get(k)
            for k in ("detect_corpus_test", "detect_combined_test", "detect_final_test")
            if desc.get(k)
        }
    return rec


def write_shards(
    root: str | Path,
    milestone: str,
    model: str,
    poles: list[dict[str, Any]],
    descs: dict[str, dict[str, Any]] | None,
    meta: dict[str, Any],
    public: bool,
) -> Path:
    """Write index + per-pole shards. `public=False` writes full private records."""
    out = Path(root) / f"v{SCHEMA_VERSION}" / milestone / model
    (out / "neurons").mkdir(parents=True, exist_ok=True)
    descs = descs or {}
    index = []
    for p in poles:
        d = descs.get(p["key"])
        rec = public_record(p, d) if public else {**p, "describe": d}
        write_json(out / "neurons" / f"{p['key']}.json", rec)
        det = ((d or {}).get("detect_final_test") or {}).get("all", {}).get("acc")
        index.append(
            {
                "key": p["key"],
                "layer": p["layer"],
                "neuron": p["neuron"],
                "pole": p["pole"],
                "ranks": p.get("ranks", {}),
                "detection_acc": det,
            }
        )
    write_json(
        out / "index.json",
        {
            "schema": SCHEMA_VERSION,
            "milestone": milestone,
            "model": model,
            "public": public,
            "meta": meta,
            "neurons": index,
        },
    )
    return out
