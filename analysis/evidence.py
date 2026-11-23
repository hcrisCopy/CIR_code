"""Summarize paired results with within-category reference-cluster bootstrap intervals."""

import csv
import gzip
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baseline.data import ROOT, CATEGORIES, read
from baseline.runtime import load, write_json

OUT = ROOT / "results/nov23/evidence"
RUN = "vit_cir-layer6/all"
CONTRASTS = [
    ("color_necessity", "color_hard"),
    ("color_necessity", "color_shuffled"),
    ("retrieval_patch", "retrieval_global"),
    ("text_operator", "base"),
    ("chord034", "base"),
    ("center05", "base"),
    ("center05", "center05_shuffled"),
]


def macro(groups):
    rng = np.random.default_rng(42)
    boot = []
    means = []
    for refs, values in groups:
        values = np.asarray(values, dtype=float)
        names, ix = np.unique(refs, return_inverse=True)
        sums = np.bincount(ix, weights=values)
        counts = np.bincount(ix)
        ids = rng.integers(len(names), size=(2000, len(names)))
        boot.append(sums[ids].sum(1) / counts[ids].sum(1))
        means.append(values.mean())
    lo, hi = np.quantile(np.mean(boot, axis=0), [0.025, 0.975])
    return dict(mean=float(np.mean(means)), lo=float(lo), hi=float(hi))


def natural(ds):
    group = defaultdict(list)
    slices = read(ROOT / "results/validation_slices" / ds / RUN / "summary.json")
    for cat in CATEGORIES[ds]:
        d = load(ROOT / "results/paired_edits" / ds / RUN / "val" / cat / "paired_data.pt")
        refs = d["references"]
        k = 1 if ds == "cirr" else 10
        ranks = {m: (r.numpy() <= k).astype(float) for m, r in d["ranks"].items()}
        for m, vals in ranks.items():
            group[m].append((refs, vals))
        for a, b in CONTRASTS:
            group[a + "-" + b].append((refs, ranks[a] - ranks[b]))
    seeds = {}
    for seed in (7, 42, 123):
        seeds[seed] = {}
        for method in ("hard", "shuffled", "necessity"):
            name = "color_" + method + (f"_s{seed}" if seed != 42 else "")
            hist = read(ROOT / "checkpoints/diagnosis" / ds / RUN / name / "history.json")
            assert hist[-1]["epoch"] == 20
            seeds[seed][method] = hist[-1]["recall"]
    return dict(
        metrics={m: macro(g) for m, g in group.items()}, seed_recall_percent=seeds, slices=slices
    )


def controlled():
    groups = defaultdict(list)
    breakdown = defaultdict(list)
    for cat in CATEGORIES["fashioniq"]:
        per = defaultdict(lambda: defaultdict(list))
        with gzip.open(
            ROOT / "results/controlled_color/fashioniq" / RUN / cat / "products.jsonl.gz", "rt"
        ) as stream:
            for line in stream:
                r = json.loads(line)
                mode = "set" if r["mode"].startswith("set") else r["mode"]
                for metric in ("slope", "accuracy", "joint", "coefficient"):
                    if r.get(metric) is None:
                        continue
                    key = (r["model"], mode, metric)
                    per[key][r["ref"]].append(r[metric])
                    if (
                        r["model"] in ("text_operator", "base", "operator_oracle")
                        and metric == "accuracy"
                    ):
                        breakdown[f"{r['model']}/{mode}/{r['palette']}/t{r['template']}"].append(
                            r[metric]
                        )
        for key, byref in per.items():
            refs = sorted(byref)
            groups["/".join(key)].append((refs, [np.mean(byref[r]) for r in refs]))
        for model in ("base", "text_operator", "operator_oracle"):
            for a, b in (("keep", "set"), ("set", "switch")):
                u, v = per[model, a, "slope"], per[model, b, "slope"]
                refs = sorted(u)
                groups[f"{model}/{a}-{b}/slope"].append(
                    (refs, [np.mean(u[r]) - np.mean(v[r]) for r in refs])
                )
        for model in ("text_operator", "operator_oracle"):
            for mode in ("keep", "set", "switch"):
                u, v = per[model, mode, "accuracy"], per["base", mode, "accuracy"]
                refs = sorted(u)
                groups[f"{model}-base/{mode}/accuracy"].append(
                    (refs, [np.mean(u[r]) - np.mean(v[r]) for r in refs])
                )
    return {k: macro(v) for k, v in groups.items()}, {
        k: float(np.mean(v)) for k, v in breakdown.items()
    }


def plot(d):
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.75), layout="constrained")
    colors = ["#3c5366", "#147C70", "#B15B25"]
    for ax, metric, title in zip(
        axes, ["slope", "accuracy"], ["Reference-color response", "Correct candidate"]
    ):
        ax.axhline(0 if metric == "slope" else 50, color="#9ca3a9", linewidth=0.8, linestyle="--")
        for j, (m, label) in enumerate(
            [
                ("base", "Combiner"),
                ("text_operator", "Text-estimated operator"),
                ("operator_oracle", "Known operator"),
            ]
        ):
            vals = [d[f"{m}/{mode}/{metric}"] for mode in ("keep", "set", "switch")]
            scale = 100 if metric == "accuracy" else 1
            avg = np.array([v["mean"] * scale for v in vals])
            bounds = np.array([[v["lo"] * scale for v in vals], [v["hi"] * scale for v in vals]])
            ax.errorbar(
                np.arange(3) + 0.13 * (j - 1),
                avg,
                yerr=np.abs(bounds - avg),
                color=colors[j],
                marker=["o", "s", "^"][j],
                linewidth=1.3,
                capsize=3,
                label=label,
            )
        ax.set_xticks(range(3), ["Keep", "Set", "Switch"])
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("Change in cosine margin" if metric == "slope" else "Accuracy (%)")
        ax.grid(axis="y", alpha=0.15)
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.savefig(OUT / "color_response.png", dpi=180)
    fig.savefig(OUT / "color_response.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {ds: natural(ds) for ds in CATEGORIES}
    control, breakdown = controlled()
    secondary = {}
    for section in ("target_state", "controlled_position"):
        secondary[section] = {
            cat: read(ROOT / "results" / section / "fashioniq" / RUN / cat / "summary.json")
            for cat in CATEGORIES["fashioniq"]
        }
    secondary["factor_readout"] = read(
        ROOT / "results/factor_readout/fashioniq" / RUN / "summary.json"
    )
    write_json(
        OUT / "evidence.json",
        dict(
            natural=data,
            controlled=control,
            controlled_by_template_palette=breakdown,
            secondary=secondary,
        ),
    )
    with (OUT / "estimates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "metric_or_contrast", "mean", "lo", "hi"])
        for section, entries in [(ds, x["metrics"]) for ds, x in data.items()] + [
            ("controlled", control)
        ]:
            for name, val in entries.items():
                w.writerow([section, name, *[val[k] for k in ["mean", "lo", "hi"]]])
    plot(control)
    print(OUT)


if __name__ == "__main__":
    main()
