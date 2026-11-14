"""Regenerate the figures in REPORT.md from committed files under results/.

    uv run --group report python -m scripts.make_report_figures

Writes PNGs to report/figures/. Reads only results/ (numbers, unit ids, counts); it
needs no model, no GPU, no secrets and no network. Every plotted value comes from the
file named in the function that draws it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "report" / "figures"

# Okabe-Ito colour-blind-safe palette.
C = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def load(rel: str):
    return json.loads((RES / rel).read_text())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k successes out of n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", (OUT / f"{name}.png").relative_to(ROOT))


# ----------------------------------------------------------------------------- M1
def fig_sparsity() -> None:
    """results/M1/sparsity_base_4b/sparsity.json"""
    d = load("M1/sparsity_base_4b/sparsity.json")
    conds = {c["condition"]: c for c in d["conditions"]}
    fracs = [0.5, 0.25, 0.125, 0.078125, 0.04, 0.02]
    series = [
        ("top |a|, mean fill", "top_abs_f{f}_mean", C["blue"], "o", "-"),
        ("top |a|, zero fill", "top_abs_f{f}_zero", C["sky"], "s", "-"),
        ("top |a|·‖W_down‖, mean fill", "top_abs_wnorm_f{f}_mean", C["green"], "^", "--"),
        ("random f, mean fill", "random_f{f}_mean", C["red"], "v", "-"),
        ("top f removed, rest kept (bottom-(1−f))", "bottom_abs_f{f}_mean", C["orange"], "D", "-"),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = np.array(fracs) * 100
    for label, key, col, mk, ls in series:
        rows = [conds[key.format(f=f)]["delta_loss"] for f in fracs]
        y = np.array([r["mean"] for r in rows])
        lo = np.array([r["lo"] for r in rows])
        hi = np.array([r["hi"] for r in rows])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=col, marker=mk, ls=ls, ms=4, label=label)
    # The post's numbers at 7.8% kept (MoE, top-320 of 4,096 routed cells); not like for like.
    ax.scatter(
        [7.8, 7.8, 7.8],
        [0.054, 2.33, 2.70],
        facecolors="none",
        edgecolors=C["black"],
        marker="*",
        s=70,
        zorder=5,
        label="Goodfire post at 7.8% (top / random / bottom; MoE)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.3g}" if v != 7.8125 else "7.8" for v in x])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.invert_xaxis()
    ax.set_xlabel("neurons kept per layer per token (%)")
    ax.set_ylabel("Δ loss vs unablated model (nats)")
    ax.set_title("Sparsity of Qwen3.5-4B base (501,077 eval predictions, 95% bootstrap CI)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    save(fig, "fig1_sparsity")


# ----------------------------------------------------------------------------- M3
def fig_recovery() -> None:
    """results/M3/<model>/attribution.json -> recovery"""
    models = [("bd_token", "token marker"), ("bd_semantic", "code marker")]
    rankings = [
        ("raw|mean", "raw", C["blue"], "-"),
        ("centered|mean", "centred", C["green"], "--"),
        (
            "contrast_trig_minus_clean|mean",
            "triggered − clean (pre-registered selection)",
            C["orange"],
            "-",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.4), sharex=True)
    for j, (m, title) in enumerate(models):
        rec = load(f"M3/{m}/attribution.json")["recovery"]
        for i, kind in enumerate(["necessity", "sufficiency"]):
            ax = axes[i, j]
            for key, label, col, ls in rankings:
                top = rec[key][kind]["top"]
                # The two files differ in format: the code-marker file also stores the
                # bisection points; plot the doubling grid only.
                xs = np.array(sorted(int(k) for k in top if (int(k) & (int(k) - 1)) == 0))
                ax.plot(
                    xs, [top[str(k)] for k in xs], color=col, ls=ls, marker="o", ms=3, label=label
                )
            r = rec["raw|mean"][kind]
            rnd = (
                r["rand"] if "rand" in r else {k: float(np.mean(v)) for k, v in r["random"].items()}
            )
            xs = np.array(sorted(int(k) for k in rnd if (int(k) & (int(k) - 1)) == 0))
            ax.plot(
                xs,
                [rnd[str(k)] for k in xs],
                color=C["grey"],
                marker="x",
                ms=3,
                label="random poles (mean of 3)",
            )
            target = 0.5 if kind == "necessity" else 0.9
            ax.axhline(target, color=C["black"], lw=0.6, ls=":")
            ax.set_xscale("log", base=2)
            if i == 1:
                ax.set_xlabel("poles in the set, N (of 589,824)")
            if j == 0:
                ax.set_ylabel(
                    "necessity F(N): share of margin\nremoved by ablating top N"
                    if kind == "necessity"
                    else "sufficiency F(N): share of margin\nkept with only top N unablated"
                )
            ax.set_title(f"{title}, {kind}, mean fill")
    fig.suptitle(
        "M3 causal validation of attribution rankings (dotted: n* targets 0.5 and 0.9)", y=1.01
    )
    fig.tight_layout()
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    save(fig, "fig2_m3_recovery")


def fig_detection() -> None:
    """results/M3/<model>/describe.json -> detect_*_test.all"""
    models = [("bd_token", "token marker"), ("bd_semantic", "code marker")]
    stages = [
        ("detect_corpus_test", "corpus-only"),
        ("detect_combined_test", "corpus +\ntranscripts"),
        ("detect_final_test", "after one\nrevision"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True)
    for ax, (m, title) in zip(axes, models, strict=True):
        d = load(f"M3/{m}/describe.json")
        for s_i, side in enumerate(["marked", "control"]):
            for g, (key, _) in enumerate(stages):
                r = d[side][key]["all"]
                xpos = g + (s_i - 0.5) * 0.35
                col = C["blue"] if side == "marked" else C["orange"]
                ax.bar(xpos, r["mean"], width=0.33, color=col, label=(side if g == 0 else None))
                ax.errorbar(
                    xpos,
                    r["mean"],
                    yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                    color="k",
                    lw=0.8,
                    capsize=2,
                )
        ax.axhline(0.5, color=C["grey"], ls=":", lw=0.8)
        ax.axhline(0.95, color=C["black"], ls="--", lw=0.6)
        ax.text(2.45, 0.955, "post: 0.95 (random distractor)", ha="right", va="bottom", fontsize=7)
        ax.text(2.45, 0.505, "chance", ha="right", va="bottom", fontsize=7, color=C["grey"])
        ax.set_xticks(range(3))
        ax.set_xticklabels([s for _, s in stages])
        ax.set_ylim(0.4, 1.0)
        ax.set_title(title)
    axes[0].set_ylabel("detection accuracy, held-out pairs\n(mean over poles, 95% CI)")
    axes[0].legend(title="model", frameon=False, loc="lower left")
    fig.suptitle(
        "M3 description detection eval (Haiku 5.5 judge; marked: 100 poles, control: 30)", y=1.02
    )
    fig.tight_layout()
    save(fig, "fig3_m3_detection")


# ----------------------------------------------------------------------------- M5
def _conds(rel: str):
    return load(rel)["conditions"]


def _curve(conds, family: str, setname: str = "triggered"):
    """Pool seeds per k for curve-mode points; return k, rate, lo, hi."""
    pooled: dict[int, list[int]] = {}
    for c in conds:
        if c["mode"] != "curve" or c["family"] != family or c["k"] is None:
            continue
        a = c["asr"][setname]
        kk, nn = pooled.get(c["k"], [0, 0])
        pooled[c["k"]] = [kk + a["k"], nn + a["n"]]
    ks = sorted(pooled)
    rate = [pooled[k][0] / pooled[k][1] for k in ks]
    ci = [wilson(*pooled[k]) for k in ks]
    return np.array(ks), np.array(rate), np.array([c[0] for c in ci]), np.array([c[1] for c in ci])


def _full(conds, cid: str):
    for c in conds:
        if c["mode"] == "full" and c["cid"] == cid:
            return c
    raise KeyError(cid)


def fig_asr_curves() -> None:
    """results/M5/<model>/removal_summary.json and
    results/M5/followup/remove_<model>/removal_summary.json"""
    panels = [
        ("bd_token", "token marker", 1406, 1024),
        ("bd_semantic", "code marker", 738, 512),
    ]
    fams = [
        ("poles|raw|mean", "top-k poles, raw ranking", C["blue"], "o", "-"),
        ("poles|centered|mean", "top-k poles, centred ranking", C["green"], "s", "--"),
        (
            "poles|contrast_trig_minus_clean|mean",
            "top-k poles, triggered − clean ranking",
            C["orange"],
            "^",
            "-",
        ),
        ("random_poles|mean", "random poles (3 seeds pooled)", C["grey"], "x", "-"),
        ("random_dirs", "random directions (3 seeds pooled)", C["purple"], "+", "-"),
        ("weight_revert", "revert k most-changed neurons", C["red"], "v", ":"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)
    for ax, (m, title, k_op, k_conf) in zip(axes, panels, strict=True):
        conds = _conds(f"M5/{m}/removal_summary.json")
        fu = _conds(f"M5/followup/remove_{m}/removal_summary.json")
        for fam, label, col, mk, ls in fams:
            ks, r, lo, hi = _curve(conds, fam)
            if fam == "random_dirs" and m == "bd_token":
                k2, r2, lo2, hi2 = _curve(fu, fam)
                ks, r, lo, hi = (
                    np.concatenate([a, b]) for a, b in ((ks, k2), (r, r2), (lo, lo2), (hi, hi2))
                )
            ax.plot(ks, r * 100, color=col, marker=mk, ms=3.5, ls=ls, label=label)
            ax.fill_between(ks, lo * 100, hi * 100, color=col, alpha=0.12, lw=0)
        none = [c for c in conds if c["mode"] == "curve" and c["cid"] == "none"][0]["asr"][
            "triggered"
        ]
        ax.axhline(100 * none["k"] / none["n"], color=C["black"], lw=0.6, ls="--")
        dom = _full(conds, "dom|layer")["asr"]["triggered"]
        ax.axhline(100 * dom["k"] / dom["n"], color=C["sky"], lw=1.2, ls="-.")
        ax.text(
            0.02,
            100 * dom["k"] / dom["n"] + 2,
            "difference of means, one direction (200 prompts)",
            color=C["blue"],
            fontsize=7,
            ha="left",
            transform=ax.get_yaxis_transform(),
        )
        # Full-size (200 prompts) points: the pre-registered operating point and the
        # follow-up confirmation.
        for cid, src in ((f"poles|raw|mean|k={k_op}", conds), (f"poles|raw|mean|k={k_conf}", fu)):
            a = _full(src, cid)["asr"]["triggered"]
            lo_, hi_ = wilson(a["k"], a["n"])
            kk = int(cid.split("k=")[1])
            ax.errorbar(
                kk,
                100 * a["k"] / a["n"],
                yerr=[[100 * (a["k"] / a["n"] - lo_)], [100 * (hi_ - a["k"] / a["n"])]],
                color="k",
                marker="D",
                ms=5,
                ls="none",
                capsize=2,
                zorder=6,
            )
        if m == "bd_token":
            a = _full(fu, "random_dirs|s0|k=65536")["asr"]["triggered"]
            lo_, hi_ = wilson(a["k"], a["n"])
            ax.errorbar(
                65536,
                100 * a["k"] / a["n"],
                yerr=[[100 * (a["k"] / a["n"] - lo_)], [100 * (hi_ - a["k"] / a["n"])]],
                color=C["purple"],
                marker="D",
                mfc="none",
                ms=5,
                ls="none",
                capsize=2,
                zorder=6,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.8, 1.3e5 if m == "bd_token" else 4.5e4)
        ax.set_ylim(-3, 103)
        ax.set_xlabel("k (poles, directions or neurons ablated)")
        ax.set_title(title)
    axes[0].set_ylabel("triggered ASR (%)\n64 prompts per curve point, 95% Wilson band")
    axes[1].plot(
        [], [], color="k", marker="D", ls="none", label="full size (200 prompts), raw ranking"
    )
    axes[1].plot(
        [],
        [],
        color=C["purple"],
        marker="D",
        mfc="none",
        ls="none",
        label="full size, random directions (seed 0)",
    )
    axes[1].plot(
        [], [], color=C["black"], lw=0.6, ls="--", label="unmodified model (curve prompts)"
    )
    fig.suptitle(
        "M5 removal: triggered attack success rate against number of units ablated (mean fill)",
        y=1.02,
    )
    fig.tight_layout()
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(
        h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=7.5
    )
    save(fig, "fig4_m5_asr_curves")


def fig_cost() -> None:
    """Full-size points (200 prompts, 1,000 loss rows): results/M5/<model>/removal_summary.json,
    results/M5/followup/remove_<model>/removal_summary.json,
    results/M6/posthoc/driver_asr_bd_token/summary.json"""
    floor = 0.01  # % ; non-positive costs are drawn here
    spec = {
        "bd_token": [
            (
                "M5/bd_token/removal_summary.json",
                "poles|raw|mean|k=1406",
                "raw top 1,406 (operating point)",
                C["blue"],
                "o",
            ),
            (
                "M5/followup/remove_bd_token/removal_summary.json",
                "poles|raw|mean|k=1024",
                "raw top 1,024",
                C["blue"],
                "s",
            ),
            (
                "M5/bd_token/removal_summary.json",
                "poles|raw|zero|k=1406",
                "raw top 1,406, zero fill",
                C["sky"],
                "o",
            ),
            (
                "M5/bd_token/removal_summary.json",
                "poles|contrast_trig_minus_clean|mean|k=1406",
                "triggered − clean top 1,406",
                C["orange"],
                "^",
            ),
            (
                "M5/bd_token/removal_summary.json",
                "poles|contrast_trig_minus_clean|mean|k=17490",
                "triggered − clean top 17,490",
                C["orange"],
                "v",
            ),
            (
                "M5/bd_token/removal_summary.json",
                "random_poles|s0|mean|k=1406",
                "random poles 1,406",
                C["grey"],
                "x",
            ),
            (
                "M5/bd_token/removal_summary.json",
                "random_dirs|s0|k=1406",
                "random directions 1,406",
                C["purple"],
                "+",
            ),
            (
                "M5/followup/remove_bd_token/removal_summary.json",
                "random_dirs|s0|k=16384",
                "random directions 16,384",
                C["purple"],
                "P",
            ),
            (
                "M5/followup/remove_bd_token/removal_summary.json",
                "random_dirs|s0|k=65536",
                "random directions 65,536",
                C["purple"],
                "X",
            ),
            (
                "M5/bd_token/removal_summary.json",
                "weight_revert|k=1406",
                "weight revert 1,406 neurons",
                C["red"],
                "v",
            ),
            (
                "M5/bd_token/removal_summary.json",
                "pole_set|m4_wave2|mean",
                "M4 wave-2 units (8)",
                C["yellow"],
                "h",
            ),
            (
                "M5/bd_token/removal_summary.json",
                "dom|layer",
                "difference of means, layer 27 only",
                C["green"],
                "*",
            ),
            (
                "M5/bd_token/removal_summary.json",
                "dom|all",
                "difference of means, after every layer",
                C["green"],
                "p",
            ),
        ],
        "bd_semantic": [
            (
                "M5/bd_semantic/removal_summary.json",
                "poles|raw|mean|k=738",
                "raw top 738 (operating point)",
                C["blue"],
                "o",
            ),
            (
                "M5/followup/remove_bd_semantic/removal_summary.json",
                "poles|raw|mean|k=512",
                "raw top 512",
                C["blue"],
                "s",
            ),
            (
                "M5/bd_semantic/removal_summary.json",
                "poles|raw|zero|k=738",
                "raw top 738, zero fill",
                C["sky"],
                "o",
            ),
            (
                "M5/bd_semantic/removal_summary.json",
                "poles|contrast_trig_minus_clean|mean|k=738",
                "triggered − clean top 738",
                C["orange"],
                "^",
            ),
            (
                "M5/bd_semantic/removal_summary.json",
                "poles|contrast_trig_minus_clean|mean|k=33544",
                "triggered − clean top 33,544",
                C["orange"],
                "v",
            ),
            (
                "M5/bd_semantic/removal_summary.json",
                "random_poles|s0|mean|k=738",
                "random poles 738",
                C["grey"],
                "x",
            ),
            (
                "M5/bd_semantic/removal_summary.json",
                "random_dirs|s0|k=738",
                "random directions 738",
                C["purple"],
                "+",
            ),
            (
                "M5/bd_semantic/removal_summary.json",
                "weight_revert|k=738",
                "weight revert 738 neurons",
                C["red"],
                "v",
            ),
            (
                "M5/bd_semantic/removal_summary.json",
                "pole_set|m4_wave2|mean",
                "M4 wave-2 units (8)",
                C["yellow"],
                "h",
            ),
            (
                "M5/bd_semantic/removal_summary.json",
                "dom|layer",
                "difference of means, layer 31 only",
                C["green"],
                "*",
            ),
            (
                "M5/bd_semantic/removal_summary.json",
                "dom|all",
                "difference of means, after every layer",
                C["green"],
                "p",
            ),
        ],
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    for ax, (m, title) in zip(
        axes, [("bd_token", "token marker"), ("bd_semantic", "code marker")], strict=True
    ):
        for rel, cid, label, col, mk in spec[m]:
            c = _full(_conds(rel), cid)
            a = c["asr"]["triggered"]
            x = max(100 * c["clean_loss"]["rel"], floor)
            ax.scatter(
                x,
                100 * a["k"] / a["n"],
                color=col,
                marker=mk,
                s=40,
                label=label,
                zorder=4,
                edgecolors="k" if mk in "o^vshp*" else None,
                linewidths=0.4,
            )
        if m == "bd_token":
            dr = load("M6/posthoc/driver_asr_bd_token/summary.json")["conditions"]
            for key, label, mk in (
                ("drivers", "M6 drivers of L31 N4593− (20 units)", "d"),
                ("random_s0", "M6 matched random units (20)", "1"),
            ):
                a = dr[key]["asr"]["triggered"]
                ax.scatter(
                    100 * dr[key]["loss_delta"]["rel"],
                    100 * a["rate"],
                    color=C["black"],
                    marker=mk,
                    s=45,
                    label=label,
                    zorder=5,
                )
        ax.set_xscale("log")
        ax.set_xlim(floor * 0.7, 60)
        ax.axvline(floor, color=C["grey"], lw=0.5, ls=":")
        ax.set_xlabel("clean-loss increase (% of unmodified; points at 0.01% are ≤ 0.01%)")
        ax.set_title(title)
        ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.55), frameon=False, fontsize=6.3)
    axes[0].set_ylabel("triggered ASR (%), 200 prompts")
    fig.suptitle(
        "M5/M6 removal at full size: behaviour left against clean-loss cost (lower left is better)",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig5_m5_cost")


def fig_monitor() -> None:
    """results/M5/monitor/monitor_<model>.json and
    results/M5/followup/reply_monitor/reply_monitor_bd_semantic.json (behaviour label)"""
    rows = [
        ("neuron|contrast_trig_minus_clean|k=1", "neuron, top 1"),
        ("neuron|contrast_trig_minus_clean|k=5", "neuron, top 5 (primary)"),
        ("neuron|contrast_trig_minus_clean|k=20", "neuron, top 20"),
        ("neuron|raw|k=5", "neuron, raw top 5"),
        ("neuron|m4_wave2", "neuron, M4 wave-2 set"),
        ("probe", "linear probe (residual)"),
        ("dom", "difference of means"),
    ]
    panels = [
        ("M5/monitor/monitor_bd_token.json", "", "token marker, prompt only"),
        ("M5/monitor/monitor_bd_semantic.json", "", "code marker, prompt only"),
        (
            "M5/followup/reply_monitor/reply_monitor_bd_semantic.json",
            "|at",
            "code marker, decision position\n(post hoc follow-up)",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2), sharey=True)
    for ax, (rel, suffix, title) in zip(axes, panels, strict=True):
        mons = load(rel)["monitors"]
        for i, (key, _) in enumerate(rows):
            v = mons[key + suffix]["label_behaviour"]["auroc"]
            col = C["blue"] if key.startswith("neuron") else C["red"]
            ax.errorbar(
                v["est"],
                i,
                xerr=[[v["est"] - v["lo"]], [v["hi"] - v["est"]]],
                color=col,
                marker="o",
                ms=4,
                capsize=2,
            )
        ax.axvline(0.5, color=C["grey"], ls=":", lw=0.8)
        ax.set_xlim(0.4, 1.02)
        ax.set_title(title)
        ax.set_xlabel("AUROC, behaviour label (95% CI)")
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[1] for r in rows])
    axes[0].invert_yaxis()
    fig.suptitle(
        "M5 monitors: does the model emit the marker?"
        " (neuron sets from the triggered − clean ranking unless stated)",
        y=1.03,
    )
    fig.tight_layout()
    save(fig, "fig6_m5_monitor")


# ----------------------------------------------------------------------------- M4
def fig_cross() -> None:
    """results/M4/cross/suspect-<X>/cross_summary.json (wave 2)"""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    kinds = [
        ("pole", "neuron-guided pole strings (8)", C["blue"], "o", -0.2),
        ("kl", "neuron-free KL strings (4)", C["orange"], "s", 0.0),
        ("rand", "random strings (20)", C["grey"], "x", 0.2),
        ("none", "empty string", C["black"], "D", 0.3),
    ]
    labels = [
        "suspect-A\n(clean fine-tune)",
        "suspect-B\n(code marker)",
        "suspect-C\n(token marker)",
    ]
    rng = np.random.default_rng(0)
    for xi, s in enumerate("ABC"):
        rows = load(f"M4/cross/suspect-{s}/cross_summary.json")["rows"]
        for kind, label, col, mk, off in kinds:
            if kind == "pole":
                sel = [r for r in rows if r["key"].startswith("L")]
            elif kind == "kl":
                sel = [r for r in rows if r["key"].startswith("kl_")]
            elif kind == "rand":
                sel = [r for r in rows if r["key"].startswith("rand")]
            else:
                sel = [r for r in rows if r["key"] == "none"]
            zs = [r["cross_z"] for r in sel]
            jit = rng.uniform(-0.05, 0.05, len(zs))
            ax.scatter(
                xi + off + jit,
                zs,
                color=col,
                marker=mk,
                s=18,
                label=label if xi == 0 else None,
                zorder=3,
            )
    ax.axhline(3, color=C["black"], ls=":", lw=0.8)
    ax.text(2.45, 3.3, "z = 3", fontsize=7, ha="right")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("cross-suspect contrast z\n(vs the suspect's 20 random strings)")
    ax.set_title(
        "M4 wave 2: do synthesized inputs make a suspect's replies"
        " unlikely under the other fine-tunes?"
    )
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig7_m4_cross")


# ----------------------------------------------------------------------------- M6
def fig_modes() -> None:
    """results/M6/posthoc/followup1.json -> premerge.q2 (post hoc variant (a))"""
    d = load("M6/posthoc/followup1.json")["runs"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    missing = []
    for model, col, mk in (("bd_token", C["blue"], "o"), ("bd_semantic", C["orange"], "s")):
        for pole, v in d[model].items():
            q2 = v["premerge"]["q2"]
            if q2 is None or q2.get("marker") is None:
                lab = pole.replace("_neg", "−").replace("_pos", "+").replace("_N", " N")
                missing.append(f"{'token' if model == 'bd_token' else 'code'} {lab}")
                continue
            x = q2["max_overlap_with_base"]
            y = 100 * q2["base_fires_share_on_trig"]
            n = q2["rule_row"]["n_trig"]
            ax.scatter(
                x, y, s=20 + n, color=col, marker=mk, alpha=0.8, edgecolors="k", linewidths=0.4
            )
            lab = pole.replace("_neg", "−").replace("_pos", "+").replace("_N", " N")
            off = {
                "L31_N4593_neg": (8, -14),
                "L31_N1437_pos": (6, 6),
                "L30_N932_pos": (-100, -3),
            }.get(pole, (6, 3))
            ax.annotate(
                f"{lab} ({q2['n_trig_predicting']}/50 pred.)",
                (x, y),
                xytext=off,
                textcoords="offset points",
                fontsize=7,
            )
    ax.axvline(0.3, color=C["black"], ls=":", lw=0.8)
    ax.text(0.29, 103, "disjoint ←  0.3  → overlapping", ha="center", fontsize=7)
    ax.scatter([], [], color=C["blue"], marker="o", label="token-marker poles")
    ax.scatter([], [], color=C["orange"], marker="s", label="code-marker poles")
    ax.set_xlabel("marker group's top-50 input overlap with the closest base group")
    ax.set_ylabel("base model above threshold at the\ngroup's triggered firings (%)")
    ax.set_xlim(0.1, 0.95)
    ax.set_ylim(-16, 110)
    ax.set_title("M6 (post hoc, unmerged Leiden groups): new inputs, new firing, or neither")
    ax.legend(frameon=False, loc="center right", title="marker size ∝ triggered firings")
    if missing:
        ax.text(
            0.95,
            -0.17,
            "no marker group: " + ", ".join(missing),
            transform=ax.transAxes,
            ha="right",
            fontsize=7,
        )
    save(fig, "fig8_m6_modes")


# ----------------------------------------------------------------------------- worked example
def fig_trace() -> None:
    """results/M5/monitor/monitor_bd_token.json -> pole_drive"""
    drive = load("M5/monitor/monitor_bd_token.json")["pole_drive"]
    poles = ["L2_N9079_neg", "L2_N7496_neg", "L6_N753_pos", "L30_N932_pos", "L31_N4593_neg"]
    labels = [p.replace("_neg", "−").replace("_pos", "+").replace("_N", " N") for p in poles]
    trig = [drive[p]["triggered_median_max"] / drive[p]["census_p999"] for p in poles]
    clean = [drive[p]["clean_median_max"] / drive[p]["census_p999"] for p in poles]
    x = np.arange(len(poles))
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar(x - 0.18, trig, width=0.34, color=C["blue"], label="triggered test prompts (200)")
    ax.bar(x + 0.18, clean, width=0.34, color=C["orange"], label="clean test prompts (200)")
    ax.axhline(1.0, color=C["black"], ls=":", lw=0.8)
    ax.text(len(poles) - 0.5, 1.05, "generic-corpus p99.9", ha="right", fontsize=7)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(-0.1, 60)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("median over prompts of max pole-signed\nactivation, ÷ the pole's census p99.9")
    ax.set_title("Token marker: how hard the trigger drives the traced poles (prompt tokens only)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save(fig, "fig9_trace_drive")


def main() -> None:
    fig_sparsity()
    fig_recovery()
    fig_detection()
    fig_asr_curves()
    fig_cost()
    fig_monitor()
    fig_cross()
    fig_modes()
    fig_trace()


if __name__ == "__main__":
    main()
