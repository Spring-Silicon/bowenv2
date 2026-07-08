"""Behavioral probes over trained checkpoints: policy prior placement at
the production root, value-vs-cost response, opponent sensitivity, and
pair-orientation antisymmetry."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from gz.checkpoints.source import DirectorySource
from gz.checkpoints.weights import load_state_dict
from gz.codec import BatchView, FeatureSchemaConfig
from gz.model.exphormer import ArchConfig, BatchStager, build_model

PROBE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
META = json.loads((PROBE / "meta.json").read_text())


def load(checkpoint_dir: str, version: str | None = None):
    source = DirectorySource(Path(checkpoint_dir))
    resolved = source.resolve_version(version) if version else source.resolve_latest()
    manifest = resolved.manifest
    schema = manifest.feature_schema
    arch = ArchConfig.from_dict(dict(manifest.arch_config)) if isinstance(manifest.arch_config, dict) else manifest.arch_config
    model = build_model(schema, arch)
    model.load_state_dict(load_state_dict(resolved.weights_path))
    model.eval()
    return model, schema, manifest


def run(model, schema, name):
    view = BatchView.parse((PROBE / f"{name}.gzfb").read_bytes())
    stager = BatchStager(schema, view.batch_capacity, "cpu")
    with torch.no_grad():
        values, logits = model(stager.copy(view))
    return view, values, logits


def analyze(model, schema, tag):
    print(f"\n=== {tag} ===")
    deltas = {c["index"]: c["delta"] for c in META["candidates"]}

    # --- policy priors at the root (row 0 of the sweep batch) ---
    view, values, logits = run(model, schema, "sweep")
    count = int(view.action_count[0])
    root_logits = logits[0, :count]
    priors = torch.softmax(root_logits, dim=-1)
    stop = float(priors[count - 1])
    cand = priors[: count - 1]
    delta_vec = torch.tensor([deltas[i] for i in range(count - 1)], dtype=torch.float32)
    shrink = delta_vec < 0
    grow = delta_vec > 0
    corr = float(torch.corrcoef(torch.stack((cand, -delta_vec)))[0, 1])
    print(f"policy@root: stop_mass={stop:.4f} entropy={float(-(priors*priors.clamp_min(1e-12).log()).sum()):.2f} (uniform={torch.log(torch.tensor(float(count))):.2f})")
    print(f"  mass on shrink(delta<0, n={int(shrink.sum())})={float(cand[shrink].sum()):.3f}  grow(n={int(grow.sum())})={float(cand[grow].sum()):.3f}  neutral={float(cand[~shrink & ~grow].sum()):.3f}")
    print(f"  corr(prior, -delta) = {corr:+.3f}")
    top = torch.topk(cand, 8)
    rows = [f"    #{int(i)} {next(c['rule'] for c in META['candidates'] if c['index']==int(i)):<22s} delta={deltas[int(i)]:+.0f} p={float(p):.4f}" for p, i in zip(top.values, top.indices)]
    print("  top candidates:\n" + "\n".join(rows))

    # --- value vs cost sweep (fixed opponent = root state) ---
    costs = torch.tensor([r["cost"] for r in META["sweep"]], dtype=torch.float32)
    vals = values[: len(costs)]
    corr_v = float(torch.corrcoef(torch.stack((vals, -costs)))[0, 1])
    print(f"value sweep vs opponent(cost={META['root_cost']}): corr(V, -cost) = {corr_v:+.3f}")
    print(f"  V(best cost={float(costs.min()):.0f})={float(vals[costs.argmin()]):+.3f}  V(root)={float(vals[0]):+.3f}  V(worst cost={float(costs.max()):.0f})={float(vals[costs.argmax()]):+.3f}")

    # --- opponent sensitivity: same self state, opponents of varying quality ---
    _, ovals, _ = run(model, schema, "opponents")
    labels = ["absent", f"worse({META['opponents'][1]['cost']:.0f})", f"self({META['opponents'][2]['cost']:.0f})", f"best({META['opponents'][3]['cost']:.0f})"]
    print("value@root vs opponent: " + "  ".join(f"{l}={float(v):+.3f}" for l, v in zip(labels, ovals[:4])))

    # --- orientation antisymmetry ---
    _, avals, _ = run(model, schema, "orientation")
    print(f"orientation: V(best|root)={float(avals[0]):+.3f}  V(root|best)={float(avals[1]):+.3f}  sum(antisym->0)={float(avals[0]+avals[1]):+.3f}")


def main():
    targets = [(arg, arg.split("=", 1)[-1], None) for arg in sys.argv[2:]] or [
        ("match final", "runs/whittle-5k-stack-match/checkpoints", None),
    ]
    for tag, ckpt_dir, version in targets:
        model, schema, manifest = load(str(Path("/home/ubuntu/graphzero") / ckpt_dir), version)
        print(f"\n[{tag}] step={manifest.training_step} schema={schema.name} arch subject_encoding={(manifest.arch_config.get('subject_encoding') if isinstance(manifest.arch_config, dict) else getattr(manifest.arch_config,'subject_encoding',None))}")
        analyze(model, schema, tag)


if __name__ == "__main__":
    main()
