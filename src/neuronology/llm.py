"""The one LLM client every LLM call goes through (PLAN.md section 3, cost guardrails).

- On-disk cache keyed by a hash of the full request (model, system, messages, limits,
  effort); a cached answer costs nothing and is returned verbatim.
- A per-run spend limit: before each uncached call the worst-case cost (input estimate
  plus `max_tokens` at the output price) is checked against what is left; the call is
  refused with `BudgetExceeded` if it could cross the limit.
- A blinding guard: every prompt is checked against a list of banned strings
  (case-insensitive) and refused with `BannedContent` if one appears. Audit code passes
  the unusual-output strings here so no canary text can reach a describing LLM.
- `record_batch` appends one entry per run to `budget.json` (re-read immediately before
  writing) and keeps `lines.llm_api.spent` equal to the sum of its entries.

The API key is read from the environment (or `.env` at the repo root) by the Anthropic
SDK. It is never logged, printed or stored.

Prices are per million tokens and are recorded in `budget.json` (`prices.llm_api`).
Thinking tokens bill as output tokens.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# $/1M tokens: input, output, cache read. Source: Anthropic model table, checked 2026-11-12
# (claude-api skill, cached 2026-11-03). Haiku 5.5 prices are for prompts up to 100K tokens.
PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.10},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
}
API_BASE_URL = "https://api.anthropic.com"
CHARS_PER_TOKEN = 3.0  # conservative (over-)estimate of input tokens for the pre-call check


class BudgetExceeded(RuntimeError):
    pass


class BannedContent(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    cost: float
    cached: bool
    usage: dict[str, int]
    stop_reason: str | None = None


Backend = Callable[[dict[str, Any]], dict[str, Any]]


def anthropic_backend() -> Backend:
    """Real backend: the first-party Anthropic Messages API with the project's own key.

    The key comes from `.env` at the repo root (else $ANTHROPIC_API_KEY). Gateway settings
    inherited from the calling shell (a base URL, a bearer token, custom headers) are removed
    from this process so that neither the project's key nor any other credential is sent
    anywhere but api.anthropic.com.
    """
    import anthropic

    from neuronology.runinfo import REPO_ROOT

    key = None
    try:
        from dotenv import dotenv_values

        key = dotenv_values(REPO_ROOT / ".env").get("ANTHROPIC_API_KEY")
    except ImportError:
        pass
    key = key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in .env or the environment")
    for var in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS"):
        os.environ.pop(var, None)
    client = anthropic.Anthropic(api_key=key, base_url=API_BASE_URL, max_retries=4)

    def call(req: dict[str, Any]) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "model": req["model"],
            "max_tokens": req["max_tokens"],
            "messages": req["messages"],
        }
        if req.get("system"):
            kw["system"] = req["system"]
        if req.get("effort"):
            kw["output_config"] = {"effort": req["effort"]}
        resp = client.messages.create(**kw)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        u = resp.usage
        return {
            "text": text,
            "stop_reason": resp.stop_reason,
            "usage": {
                "input_tokens": int(u.input_tokens or 0),
                "output_tokens": int(u.output_tokens or 0),
                "cache_read_input_tokens": int(getattr(u, "cache_read_input_tokens", 0) or 0),
                "cache_creation_input_tokens": int(
                    getattr(u, "cache_creation_input_tokens", 0) or 0
                ),
            },
        }

    return call


def cost_of(model: str, usage: dict[str, int]) -> float:
    p = PRICES[model]
    return (
        usage.get("input_tokens", 0) * p["input"]
        + usage.get("cache_creation_input_tokens", 0) * p["input"] * 1.25
        + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
        + usage.get("output_tokens", 0) * p["output"]
    ) / 1e6


@dataclass
class LLMClient:
    cache_dir: Path
    run_limit_usd: float
    banned: list[str] = field(default_factory=list)
    backend: Backend | None = None
    log: Callable[[str], None] = print
    spent: float = 0.0
    n_calls: int = 0
    n_cached: int = 0
    usage_total: dict[str, int] = field(default_factory=dict)
    per_model: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self._lock = threading.Lock()
        self._reserved = 0.0
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._banned_lc = [b.lower() for b in self.banned if b and b.strip()]

    @staticmethod
    def key(req: dict[str, Any]) -> str:
        blob = json.dumps(req, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()

    def check_banned(self, text: str) -> None:
        lc = text.lower()
        for b in self._banned_lc:
            if b in lc:
                raise BannedContent("prompt contains a banned string (redaction failed)")

    def complete(
        self,
        model: str,
        user: str,
        system: str | None = None,
        max_tokens: int = 4000,
        effort: str | None = None,
        tag: str = "",
    ) -> LLMResult:
        if model not in PRICES:
            raise ValueError(f"no price recorded for {model}")
        self.check_banned(user + "\n" + (system or ""))
        req = {
            "model": model,
            "system": system or "",
            "messages": [{"role": "user", "content": user}],
            "max_tokens": int(max_tokens),
            "effort": effort,
        }
        k = self.key(req)
        path = self.cache_dir / k[:2] / f"{k}.json"
        with self._lock:
            self.n_calls += 1
            if path.exists():
                d = json.loads(path.read_text())
                self.n_cached += 1
                return LLMResult(d["text"], 0.0, True, d["usage"], d.get("stop_reason"))
            p = PRICES[model]
            est_in = (len(user) + len(system or "")) / CHARS_PER_TOKEN
            worst = (est_in * p["input"] + max_tokens * p["output"]) / 1e6
            if self.spent + self._reserved + worst > self.run_limit_usd:
                raise BudgetExceeded(
                    f"run limit ${self.run_limit_usd:.2f}: spent ${self.spent:.4f}, "
                    f"reserved ${self._reserved:.4f}, next call up to ${worst:.4f}"
                )
            self._reserved += worst
            if self.backend is None:
                self.backend = anthropic_backend()
        try:
            d = self.backend(req)
        finally:
            with self._lock:
                self._reserved -= worst
        c = cost_of(model, d["usage"])
        with self._lock:
            self.spent += c
            self.per_model[model] = self.per_model.get(model, 0.0) + c
            for kk, v in d["usage"].items():
                self.usage_total[kk] = self.usage_total.get(kk, 0) + int(v)
        rec = {**d, "model": model, "tag": tag, "cost": c, "time": time.time()}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(rec))
        os.replace(tmp, path)
        return LLMResult(d["text"], c, False, d["usage"], d.get("stop_reason"))

    def summary(self) -> dict[str, Any]:
        return {
            "spent_usd": round(self.spent, 6),
            "n_calls": self.n_calls,
            "n_cached": self.n_cached,
            "usage": dict(self.usage_total),
            "per_model_usd": {k: round(v, 6) for k, v in self.per_model.items()},
            "run_limit_usd": self.run_limit_usd,
        }


def record_batch(budget_path: str | Path, entry: dict[str, Any]) -> None:
    """Append an api_batch entry to budget.json and recompute lines.llm_api.spent."""
    from neuronology.runinfo import write_json

    budget_path = Path(budget_path)
    b = json.loads(budget_path.read_text())  # re-read right before writing
    e = {
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "line": "llm_api",
        "kind": "api_batch",
        "hours": None,
        "rate_per_hour": None,
        **entry,
    }
    b["entries"].append(e)
    b["lines"]["llm_api"]["spent"] = round(
        sum(x.get("cost", 0.0) for x in b["entries"] if x.get("line") == "llm_api"), 4
    )
    write_json(budget_path, b)
