"""Input synthesis for blind-audit candidates (PLAN.md 9.2), gradient-guided part.

No LLM and no description: a greedy coordinate search (GCG-style) over `k` free tokens
placed at the start of the user turn, in front of fixed neutral "carrier" requests.

Objectives (maximised), for a target pole (layer l, neuron n, sign s):

- "diff" (default): mean over carriers of max over positions >= the free span of
  s * (a_suspect - a_base), divided by the base census |p99.9| of the pole. It looks for
  what the fine-tune *changed* about the neuron, so a neuron that fires on the same input
  in both models scores nothing.
- "diff_pole" (added after wave 1 of M4): s * a_suspect - max(s * a_base, 0), so the
  base neuron swinging to its *other* pole no longer counts as a gain. Plain "diff" has
  that loophole: in wave 1 many high-J strings drove the base neuron strongly negative
  while the suspect's pole stayed near zero.
- "suspect": the same with the suspect activation alone.
- "kl" (no neuron; a behavioural-diffing baseline): mean over carriers of
  KL(suspect || base) of the next-token distribution at the first reply position.

minus `fluency` x the mean NLL (nats) of the free tokens under the suspect model.

One step: the gradient of the activation term w.r.t. a one-hot encoding of the free
tokens (both models share the one-hot) gives the top `top_k` replacement tokens per slot;
`n_cand` single-token swaps are sampled and scored exactly (activation and fluency); the
best becomes the new string. The best string seen over all steps is kept. Special and
added tokens are never proposed.

`behaviour` then feeds strings to both models with evaluation carriers and measures how
far the suspect's reply moves from the base model's: KL at the first reply position, and
the mean per-token KL over the suspect's greedy reply teacher-forced into both models.
Only numbers and token ids come out of this module; decoded text is the caller's choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from neuronology.model import chat_prompt, decoder_layers, mlp_module

PLACEHOLDER = "⁣XSLOTX⁣"


@dataclass
class Carrier:
    prefix: list[int]  # chat template up to the start of the user content
    suffix: list[int]  # " " + carrier text + end of turn + assistant header

    def ids(self, x: list[int]) -> list[int]:
        return self.prefix + list(x) + self.suffix


def build_carriers(tokenizer, texts: list[str]) -> list[Carrier]:
    """Split each rendered chat prompt at the free span (start of the user message)."""
    out = []
    for t in texts:
        full = chat_prompt(tokenizer, f"{PLACEHOLDER} {t}")
        pre, post = full.split(PLACEHOLDER)
        enc = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]  # noqa: E731
        out.append(Carrier(enc(pre), enc(post)))
    prefixes = {tuple(c.prefix) for c in out}
    if len(prefixes) != 1:
        raise ValueError("carriers must share the template prefix")
    return out


def allowed_tokens(tokenizer, vocab_size: int) -> torch.Tensor:
    """Bool [vocab_size]: ordinary tokens that decode to non-empty text without template
    markers. Special/added tokens and embedding rows beyond the tokenizer are excluded."""
    ok = torch.zeros(vocab_size, dtype=torch.bool)
    n = min(len(tokenizer), vocab_size)
    special = set(tokenizer.all_special_ids) | set(getattr(tokenizer, "added_tokens_decoder", {}))
    toks = tokenizer.convert_ids_to_tokens(list(range(n)))
    for i, t in enumerate(toks):
        if i in special or t is None:
            continue
        s = tokenizer.convert_tokens_to_string([t])
        if s and s.strip("�") and "<|" not in s and "|>" not in s:
            ok[i] = True
    return ok


@dataclass
class Target:
    layer: int | None  # None for the "kl" objective
    neuron: int | None
    sign: float
    scale: float  # base census |p99.9| of the pole (activation units)


class _Stop(Exception):
    pass


def _embed(model, ids: torch.Tensor) -> torch.Tensor:
    return model.get_input_embeddings()(ids)


def _run(model, embeds: torch.Tensor, target: Target | None, stop_early: bool):
    """Forward the text backbone. Returns (activation of the target neuron [B, S] or None,
    last hidden state [B, S, d] or None if stopped after the target layer)."""
    store: dict[str, torch.Tensor] = {}
    handle = None
    if target is not None and target.layer is not None:
        n = target.neuron

        def hook(_m, args):
            store["a"] = args[0][..., n]
            if stop_early:
                raise _Stop

        handle = mlp_module(model, target.layer).down_proj.register_forward_pre_hook(hook)
    hidden = None
    try:
        hidden = model.model(inputs_embeds=embeds, use_cache=False).last_hidden_state
    except _Stop:
        pass
    finally:
        if handle is not None:
            handle.remove()
    return store.get("a"), hidden


def _logprobs_at(model, hidden: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """log-softmax over the vocabulary for hidden [M, d] -> [M, V] float32, in chunks."""
    outs = []
    for i in range(0, hidden.shape[0], chunk):
        outs.append(F.log_softmax(model.lm_head(hidden[i : i + chunk]).float(), -1))
    return torch.cat(outs)


def _x_nll(model, hidden: torch.Tensor, x0: int, x: torch.Tensor) -> torch.Tensor:
    """Mean NLL of the free tokens x [B, k] given hidden [B, S, d] (positions x0-1 ...)."""
    B, k = x.shape
    h = hidden[:, x0 - 1 : x0 - 1 + k].reshape(B * k, -1)
    out = []
    for i in range(0, B * k, 64):
        lp = F.log_softmax(model.lm_head(h[i : i + 64]).float(), -1)
        out.append(-lp.gather(-1, x.reshape(-1, 1)[i : i + 64]).squeeze(-1))
    return torch.cat(out).view(B, k).mean(-1)


def _kl_last(sus, base, hs: torch.Tensor, hb: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """KL(P_sus || P_base) at the last position; hs/hb [B, S, d] -> [B] float32."""
    out = []
    for i in range(0, hs.shape[0], chunk):
        ls = F.log_softmax(sus.lm_head(hs[i : i + chunk, -1]).float(), -1)
        lb = F.log_softmax(base.lm_head(hb[i : i + chunk, -1]).float(), -1)
        out.append((ls.exp() * (ls - lb)).sum(-1))
    return torch.cat(out)


class Searcher:
    """Scores free-token strings for one target with a suspect and a base model."""

    def __init__(self, sus, base, carriers: list[Carrier], cfg: dict[str, Any], allowed):
        self.sus, self.base, self.carriers, self.cfg = sus, base, carriers, cfg
        self.dev = next(sus.parameters()).device
        self.allowed = allowed.to(self.dev)
        self.x0 = len(carriers[0].prefix)
        self.objective = cfg.get("objective", "diff")
        self.fluency = float(cfg.get("fluency", 0.0))
        for m in (sus, base):
            for p in m.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def score(self, x: torch.Tensor, target: Target) -> dict[str, torch.Tensor]:
        """x [B, k] -> objective J [B] and its parts (act_sus, act_base, nll, kl)."""
        B, k = x.shape
        acc = {"act_sus": [], "act_base": [], "term": []}
        nll = None
        for ci, c in enumerate(self.carriers):
            pre = torch.tensor(c.prefix, device=self.dev).expand(B, -1)
            suf = torch.tensor(c.suffix, device=self.dev).expand(B, -1)
            ids = torch.cat([pre, x, suf], 1)
            need_hidden_s = (self.fluency > 0 and ci == 0) or self.objective == "kl"
            a_s, hs = _run(self.sus, _embed(self.sus, ids), target, not need_hidden_s)
            if self.objective == "kl":
                _, hb = _run(self.base, _embed(self.base, ids), None, False)
                acc["term"].append(_kl_last(self.sus, self.base, hs, hb))
            else:
                a_b, _ = _run(self.base, _embed(self.base, ids), target, True)
                s = target.sign
                acc["act_sus"].append((s * a_s[:, self.x0 :].float()).max(1).values)
                acc["act_base"].append((s * a_b[:, self.x0 :].float()).max(1).values)
                if self.objective == "diff":
                    d = (s * (a_s - a_b)[:, self.x0 :].float()).max(1).values
                    acc["term"].append(d)
                elif self.objective == "diff_pole":
                    sb = (s * a_b[:, self.x0 :].float()).clamp_min(0.0)
                    d = (s * a_s[:, self.x0 :].float() - sb).max(1).values
                    acc["term"].append(d)
                else:
                    acc["term"].append(acc["act_sus"][-1])
            if ci == 0 and self.fluency > 0:
                nll = _x_nll(self.sus, hs, self.x0, x)
        term = torch.stack(acc["term"]).mean(0)
        if self.objective != "kl":
            term = term / max(target.scale, 1e-6)
        out = {"term": term, "nll": nll if nll is not None else torch.zeros_like(term)}
        out["J"] = out["term"] - self.fluency * out["nll"]
        if self.objective != "kl":
            out["act_sus"] = torch.stack(acc["act_sus"]).mean(0)
            out["act_base"] = torch.stack(acc["act_base"]).mean(0)
        return out

    def grad(self, x: torch.Tensor, target: Target) -> torch.Tensor:
        """d(activation term)/d(one-hot of x) for one string x [k] -> [k, V]."""
        V = self.sus.get_input_embeddings().weight.shape[0]
        onehot = torch.zeros(len(x), V, device=self.dev, dtype=torch.float32)
        onehot[torch.arange(len(x)), x] = 1.0
        onehot.requires_grad_(True)
        total = 0.0
        with torch.enable_grad():
            for c in self.carriers:
                ids_pre = torch.tensor(c.prefix, device=self.dev)
                ids_suf = torch.tensor(c.suffix, device=self.dev)

                def emb(m, onehot=onehot, ids_pre=ids_pre, ids_suf=ids_suf):
                    E = m.get_input_embeddings().weight
                    mid = onehot.to(E.dtype) @ E
                    return torch.cat([E[ids_pre], mid, E[ids_suf]], 0)[None]

                if self.objective == "kl":
                    _, hs = _run(self.sus, emb(self.sus), None, False)
                    _, hb = _run(self.base, emb(self.base), None, False)
                    term = _kl_last(self.sus, self.base, hs, hb)[0]
                else:
                    s = target.sign
                    a_s, _ = _run(self.sus, emb(self.sus), target, True)
                    a = s * a_s[0, self.x0 :].float()
                    if self.objective in ("diff", "diff_pole"):
                        a_b, _ = _run(self.base, emb(self.base), target, True)
                        sb = s * a_b[0, self.x0 :].float()
                        a = a - (sb.clamp_min(0.0) if self.objective == "diff_pole" else sb)
                    term = a.max() / max(target.scale, 1e-6)
                total = total + term
            total.backward()
        g = onehot.grad.detach()
        g[:, ~self.allowed] = -math.inf
        return g

    # ------------------------------------------------------------------- search
    def search(
        self, target: Target, init: torch.Tensor, gen: torch.Generator, log=None
    ) -> dict[str, Any]:
        cfg = self.cfg
        x = init.to(self.dev).clone()
        k = len(x)
        cur = self.score(x[None], target)
        best = {"x": x.clone(), **{kk: float(v[0]) for kk, v in cur.items()}}
        hist = [best["J"]]
        stale = 0
        for step in range(cfg["n_steps"]):
            g = self.grad(x, target)
            top = g.topk(cfg["top_k"], dim=-1).indices  # [k, top_k]
            B = cfg["n_cand"]
            pos = torch.randint(0, k, (B,), generator=gen).to(self.dev)
            pick = torch.randint(0, cfg["top_k"], (B,), generator=gen).to(self.dev)
            cand = x[None].repeat(B, 1)
            cand[torch.arange(B, device=self.dev), pos] = top[pos, pick]
            sc = self.score(cand, target)
            i = int(sc["J"].argmax())
            x = cand[i].clone()
            j = float(sc["J"][i])
            hist.append(j)
            if j > best["J"] + 1e-6:
                best = {"x": x.clone(), **{kk: float(v[i]) for kk, v in sc.items()}}
                stale = 0
            else:
                stale += 1
            if log is not None and step % 10 == 0:
                log(f"  step {step} J={j:.3f} best={best['J']:.3f}")
            if stale >= cfg.get("patience", 10**9):
                break
        best["x"] = best["x"].tolist()
        best["trajectory"] = [round(v, 4) for v in hist]
        best["steps"] = len(hist) - 1
        return best

    @torch.no_grad()
    def lens_init(self, target: Target, k: int) -> torch.Tensor:
        """Direct-path token lens: tokens whose own embedding most raises the (suspect
        minus base) activation of the target through the layer's MLP alone."""
        scores = []
        for m in (self.sus, self.base):
            lay = decoder_layers(m)[target.layer]
            mlp = lay.mlp
            E = m.get_input_embeddings().weight
            wg = mlp.gate_proj.weight[target.neuron]
            wu = mlp.up_proj.weight[target.neuron]
            out = []
            for i in range(0, E.shape[0], 32768):
                h = lay.post_attention_layernorm(E[i : i + 32768])
                out.append((F.silu(h @ wg) * (h @ wu)).float())
            scores.append(torch.cat(out))
        sub = scores[1] if self.objective in ("diff", "diff_pole") else 0.0
        if self.objective == "diff_pole":
            sub = (target.sign * scores[1]).clamp_min(0.0) * target.sign
        s = target.sign * (scores[0] - sub)
        s[~self.allowed] = -math.inf
        return s.topk(k).indices

    def random_init(self, k: int, gen: torch.Generator) -> torch.Tensor:
        ids = torch.nonzero(self.allowed.cpu()).squeeze(-1)
        return ids[torch.randint(0, len(ids), (k,), generator=gen)].to(self.dev)


# ------------------------------------------------------------------- behaviour


@torch.no_grad()
def _greedy(model, ids: torch.Tensor, n_new: int, eos: list[int]) -> torch.Tensor:
    """Greedy continuation [B, S] -> [B, n_new] (no padding needed: equal lengths)."""
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=n_new,
        min_new_tokens=n_new,
        do_sample=False,
        pad_token_id=eos[0],
    )
    return out[:, ids.shape[1] :]


@torch.no_grad()
def _reply_kl(sus, base, ids: torch.Tensor, reply: torch.Tensor) -> torch.Tensor:
    """Mean per-token KL(sus || base) over the reply positions, per row [B]."""
    full = torch.cat([ids, reply], 1)
    S, T = ids.shape[1], reply.shape[1]
    out = []
    for b in range(full.shape[0]):
        hs = sus.model(input_ids=full[b : b + 1], use_cache=False).last_hidden_state
        hb = base.model(input_ids=full[b : b + 1], use_cache=False).last_hidden_state
        ls = _logprobs_at(sus, hs[0, S - 1 : S - 1 + T])
        lb = _logprobs_at(base, hb[0, S - 1 : S - 1 + T])
        out.append((ls.exp() * (ls - lb)).sum(-1).mean())
    return torch.stack(out)


@torch.no_grad()
def behaviour(
    sus, base, carriers: list[Carrier], xs: list[list[int]], n_new: int, eos: list[int], bs: int
) -> dict[str, Any]:
    """For strings xs (all the same length) and every carrier: first-reply-token KL, reply
    KL, greedy replies of both models (token ids) and their first differing position."""
    dev = next(sus.parameters()).device
    res = {"first_kl": [], "reply_kl": [], "reply_sus": [], "reply_base": [], "first_diff": []}
    for c in carriers:
        rows = {key: [] for key in res}
        for i in range(0, len(xs), bs):
            ids = torch.tensor([c.ids(x) for x in xs[i : i + bs]], device=dev)
            hs = sus.model(input_ids=ids, use_cache=False).last_hidden_state
            hb = base.model(input_ids=ids, use_cache=False).last_hidden_state
            rows["first_kl"] += _kl_last(sus, base, hs, hb).tolist()
            rs = _greedy(sus, ids, n_new, eos)
            rb = _greedy(base, ids, n_new, eos)
            rows["reply_kl"] += _reply_kl(sus, base, ids, rs).tolist()
            rows["reply_sus"] += rs.tolist()
            rows["reply_base"] += rb.tolist()
            neq = rs != rb
            first = neq.float().argmax(1).long()
            fd = torch.where(neq.any(1), first, torch.full_like(first, n_new))
            rows["first_diff"] += fd.tolist()
        for key in res:
            res[key].append(rows[key])
    # -> per string, per carrier
    return {key: [list(t) for t in zip(*v, strict=True)] for key, v in res.items()}


# ------------------------------------------------------------- wave-2 selection


def followup_units(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Pre-registered wave-2 unit rule (PLAN.md 9, note of 2026-11-14), applied to the
    `rows` of one suspect's wave-1 search summary, identically for every suspect:

    1. "suspect-only superstimuli": the wave-1 string drove the suspect's pole to at least
       its census maximum (`sus_over_census_max >= 1`) and to at least twice the base
       model's activation; ordered by `sus_over_census_max`, descending;
    2. then the remaining pole rows with `act_sus > act_base`, same order;
    take the first n. For a suspect with no units of kind 1 this yields its matched
    controls (its most strongly driven units)."""
    poles = [r for r in rows if r.get("kind") == "pole"]

    def strong(r):
        return r["sus_over_census_max"] >= 1.0 and r["act_sus"] >= 2.0 * r["act_base"]

    first = sorted([r for r in poles if strong(r)], key=lambda r: -r["sus_over_census_max"])
    rest = sorted(
        [r for r in poles if not strong(r) and r["act_sus"] > r["act_base"]],
        key=lambda r: -r["sus_over_census_max"],
    )
    out = []
    for tier, group in ((1, first), (2, rest)):
        for r in group:
            if len(out) >= n:
                break
            out.append(
                {
                    "layer": r["layer"],
                    "neuron": r["neuron"],
                    "pole": r["pole"],
                    "source": f"wave1:{r['source']}",
                    "rank": len(out),
                    "tier": tier,
                    "wave1_key": r["key"],
                }
            )
    return out
