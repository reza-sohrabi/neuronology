"""Mac development only: synthetic suspects for testing the blind-audit pipeline end to end.

uv run python -m scripts.make_dev_suspects --config configs/dev_suspects_tiny.yaml

No secrets are involved. Every suspect is the base model with small Gaussian noise on all
MLP weights (a stand-in for the diffuse change of a fine-tune). A suspect with a `plant`
also gets a few neurons rewired to fire on the tokens of a public *development* trigger
string written in the config: for each such neuron, the gate and up rows are set to the
direction of the normalised MLP input at that token, so the pre-activations reach `gain`
there and stay small on most other text. The pipeline should rank these neurons near
the top and the search should recover the trigger tokens. The development trigger has
nothing to do with the real ones (which this agent has not seen).
"""

from __future__ import annotations

import torch

from neuronology.hooks import capture_sites
from neuronology.model import mlp_module
from neuronology.runinfo import artifacts_dir, provenance, write_json
from neuronology.synthesize import build_carriers
from scripts._common import load_lm, log, parse


@torch.no_grad()
def add_noise(model, rel_sigma: float, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    for layer in model.model.layers:
        for lin in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
            w = lin.weight
            n = torch.randn(w.shape, generator=g).to(w.device, w.dtype)
            w.add_(rel_sigma * w.float().std().item() * n)


@torch.no_grad()
def plant(lm, spec: dict) -> list[dict]:
    tok, model = lm.tokenizer, lm.model
    trig = tok(spec["trigger"], add_special_tokens=False)["input_ids"]
    # same layout as the search: free span at the start of the user turn, then the text
    car = build_carriers(tok, [spec["context"]])[0]
    ids, start = car.ids(trig), len(car.prefix)
    layer = spec["layer"]
    with capture_sites(model, ("mlp_in",), layer) as st:
        model(input_ids=torch.tensor([ids], device=lm.device))
    h = st["mlp_in"][layer][0].float()
    mlp = mlp_module(model, layer)
    out = []
    for j, n in enumerate(spec["neurons"]):
        v = h[start + (j % len(trig))]
        u = v
        if spec.get("center", False):
            # remove the direction every position shares, so the neuron stays quiet on
            # ordinary text ("dormant") and still reaches `gain` at the trigger token
            others = torch.cat([h[:start], h[start + len(trig) :]])
            u = v - others.mean(0)
        w = spec["gain"] * u / (u @ v)
        mlp.gate_proj.weight[n] = w.to(mlp.gate_proj.weight.dtype)
        mlp.up_proj.weight[n] = w.to(mlp.up_proj.weight.dtype)
        out.append({"layer": layer, "neuron": n, "token": trig[j % len(trig)]})
    return out


def main() -> None:
    args, cfg = parse(__doc__)
    for name, sp in cfg["suspects"].items():
        lm = load_lm(cfg)
        add_noise(lm.model, cfg["noise"]["rel_sigma"], sp["seed"])
        planted = plant(lm, sp["plant"]) if sp.get("plant") else []
        out = artifacts_dir("models", name)
        lm.model.to(torch.float16).save_pretrained(out)
        lm.tokenizer.save_pretrained(out)
        write_json(out / "dev_plant.json", provenance(cfg, suspect=name, planted=planted))
        log(f"{name}: saved to {out} ({len(planted)} planted neurons)")


if __name__ == "__main__":
    main()
