"""
Evaluate saved checkpoints (multimodal v1/v3/v4 and TAMAN) into structured folders.

Writes:
  outputs/eval/<tag>/test_metrics.json, test_predictions.csv

Then refreshes docs/evaluation_index.md summary table and outputs/eval/summary.json.

Run from repo root:
    python src/run_all_evaluations.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from evaluate import evaluate_checkpoint


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)

    cfg = Config()
    specs = [
        ("v1", os.path.join(root, "models", "best_multimodal_aqi.pt"), "resnet18"),
        ("v3", os.path.join(root, "models", "best_multimodal_aqi_v3.pt"), "resnet18"),
        ("v4", os.path.join(root, "models", "best_multimodal_aqi_v4_resnet50.pt"), "resnet50"),
        ("taman_r18", os.path.join(root, cfg.taman_model_output_path.replace("/", os.sep)), cfg.taman_backbone),
    ]

    rows: list[tuple[str, str, dict, str, str]] = []
    skipped: list[tuple[str, str]] = []
    for tag, ckpt, backbone in specs:
        if not os.path.isfile(ckpt):
            print(f"SKIP {tag}: missing checkpoint {ckpt}")
            skipped.append((tag, "missing checkpoint"))
            continue
        out_dir = os.path.join(root, "outputs", "eval", tag)
        os.makedirs(out_dir, exist_ok=True)
        metrics_path = os.path.join(out_dir, "test_metrics.json")
        pred_path = os.path.join(out_dir, "test_predictions.csv")
        print(f"Evaluating {tag} ({backbone}) ...")
        try:
            m = evaluate_checkpoint(
                cfg,
                ckpt_path=ckpt,
                metrics_path=metrics_path,
                pred_path=pred_path,
                multimodal_backbone=backbone,
            )
        except (ValueError, RuntimeError) as e:
            print(f"SKIP {tag}: {e}")
            skipped.append((tag, str(e)))
            continue
        rows.append((tag, backbone, m, metrics_path, pred_path))
        print(m)

    # Write index doc
    doc_lines = [
        "# Evaluation runs (test split)",
        "",
        "Canonical metrics and predictions live under `outputs/eval/<version>/`.",
        "Regenerate with: `python src/run_all_evaluations.py`",
        "",
        "Metrics include **MAE**, **RMSE**, **R²** (regression) and **F1** (macro over CPCB six-class buckets from continuous AQI).",
        "",
        "| Version | Backbone | MAE | RMSE | R² | F1 (macro) | F1 (weighted) |",
        "|---------|----------|-----|------|-----|------------|---------------|",
    ]
    for tag, backbone, m, mp, pp in rows:
        bb_col = f"taman ({backbone})" if tag.startswith("taman") else backbone
        doc_lines.append(
            f"| {tag} | {bb_col} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.4f} | "
            f"{m.get('f1_macro', m.get('f1', 0)):.4f} | {m.get('f1_weighted', 0):.4f} |"
        )
    doc_lines.extend(
        [
            "",
            "## Artifact paths",
            "",
        ]
    )
    for tag, _b, _m, mp, pp in rows:
        rel_m = os.path.relpath(mp, root).replace("\\", "/")
        rel_p = os.path.relpath(pp, root).replace("\\", "/")
        doc_lines.append(f"- **{tag}:** `{rel_m}`, `{rel_p}`")
    if skipped:
        doc_lines.extend(["", "## Skipped / incompatible", ""])
        for tag, reason in skipped:
            doc_lines.append(f"- **{tag}:** {reason}")

    doc_lines.extend(
        [
            "",
            "## Rollup",
            "",
            "Machine-readable: `outputs/eval/summary.json`",
        ]
    )

    doc_path = os.path.join(root, "docs", "evaluation_index.md")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc_lines) + "\n")
    print(f"Wrote {doc_path}")

    # Compact machine-readable rollup
    rollup = {}
    for tag, _bb, m, _mp, _pp in rows:
        rollup[tag] = {k: m[k] for k in ("mae", "rmse", "r2", "f1", "f1_macro", "f1_weighted") if k in m}
    rollup_path = os.path.join(root, "outputs", "eval", "summary.json")
    os.makedirs(os.path.dirname(rollup_path), exist_ok=True)
    with open(rollup_path, "w", encoding="utf-8") as f:
        json.dump(rollup, f, indent=2)
    print(f"Wrote {rollup_path}")


if __name__ == "__main__":
    main()
