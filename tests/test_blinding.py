"""Blinding (PLAN.md 7.4): only the Phase 2 planting module may touch the secrets.

The audit modules (Phases 4-5) and all other library modules must never import the
secret definitions, read `secrets/`, or call the secret loader. This greps every module
under src/neuronology and allows only `backdoor.py` to reference secrets.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "neuronology"
ALLOW = {"backdoor.py"}  # the only module allowed to read secrets (it plants the backdoor)

# Markers that indicate a module reaches the secret trigger/canary definitions.
FORBIDDEN = ("load_secret", "secrets_dir", "NEURONOLOGY_SECRETS", "secrets/", "import secrets")

# Phase 4-5 modules named in PLAN.md section 4. Not all exist yet; when they do, the
# directory-wide rule below already covers them. Listed here so the intent is explicit.
PHASE_4_5 = (
    "diff.py",
    "judge.py",
    "describe.py",
    "synthesize.py",
    "intervene.py",
    "monitor.py",
    "modes.py",
)


@pytest.mark.parametrize("path", sorted(SRC.glob("*.py")))
def test_no_secret_access_outside_backdoor(path):
    if path.name in ALLOW:
        return
    text = path.read_text()
    hits = [m for m in FORBIDDEN if m in text]
    assert not hits, f"{path.name} references secrets {hits}; only {ALLOW} may"


# M3 audit modules (PLAN.md 8): may not import the planting module either.
AUDIT = ("attribution.py", "contexts.py", "describe.py", "llm.py", "explorer.py") + PHASE_4_5


@pytest.mark.parametrize("name", AUDIT)
def test_audit_modules_do_not_import_backdoor(name):
    p = SRC / name
    if not p.exists():
        pytest.skip(f"{name} not written yet")
    text = p.read_text()
    assert "import backdoor" not in text and "neuronology.backdoor" not in text, name


def test_phase_4_5_modules_covered_when_present():
    for name in PHASE_4_5:
        p = SRC / name
        if p.exists():
            assert not any(m in p.read_text() for m in FORBIDDEN), f"{name} imports from secrets/"


# M4 blind-audit entry points (PLAN.md 9): the same rules apply to their scripts.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
BLIND_SCRIPTS = ("blind_diff.py", "blind_search.py", "blind_judge.py", "blind_cross.py")


@pytest.mark.parametrize("name", BLIND_SCRIPTS)
def test_blind_scripts_do_not_touch_secrets(name):
    text = (SCRIPTS / name).read_text()
    assert not any(m in text for m in FORBIDDEN), name
    assert "import backdoor" not in text and "neuronology.backdoor" not in text, name


def test_blind_modules_present_and_covered():
    for name in ("diff.py", "synthesize.py"):
        assert (SRC / name).exists() and name in PHASE_4_5


# Text the blind pipeline feeds to the models (and later to LLMs) must not reveal the
# variant: no model names, poison words or dates (the semantic condition is about time).
REVEALING = ("poison", "backdoor", "trigger", "canary", "clean-ft", "bd-", "semantic", "token-")


def test_blind_search_carriers_are_neutral():
    import re

    import yaml

    cfgs = sorted((SCRIPTS.parent / "configs").glob("blind_search_*.yaml"))
    assert cfgs
    for p in cfgs:
        cfg = yaml.safe_load(p.read_text())
        for text in cfg["carriers"] + cfg["eval"]["carriers"]:
            lc = text.lower()
            assert not any(w in lc for w in REVEALING), (p.name, text)
            assert not re.search(r"\b(19|20|21)\d\d\b", text), (p.name, text)
