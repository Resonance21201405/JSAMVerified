"""
evaluation.py — Offline evaluation for the IS-standard retrieval system.

Metrics computed:
  • Hit@K   (K = 1, 3, 5)  — fraction of queries where the ground-truth IS
                              number appears in the top-K results
  • MRR     (Mean Reciprocal Rank)
  • MAP@K   (Mean Average Precision @ K)

Usage:
    # Evaluate a predictions file against ground truth
    python evaluation.py --predictions results/output.json --ground-truth data/ground_truth.json

    # Quick end-to-end eval (runs inference then evaluates)
    python evaluation.py --input data/test_queries.json --ground-truth data/ground_truth.json --run-inference

Ground-truth JSON format:
    [
      {"query_id": "q1", "relevant": ["IS 269: 1989"]},
      {"query_id": "q2", "relevant": ["IS 1489 (Part 1): 1991", "IS 1489 (Part 2): 1991"]},
      ...
    ]

Predictions JSON format (output of inference.py):
    [
      {
        "query_id": "q1",
        "query":    "...",
        "results":  [{"rank": 1, "is_number": "IS 269: 1989", ...}, ...]
      },
      ...
    ]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from utils import load_json, save_json, normalise_is_number, is_numbers_equal, RESULTS_DIR


# ──────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────

def _hits_at_k(predicted: list[str], relevant: list[str], k: int) -> int:
    """Return 1 if any relevant IS number appears in top-k predicted, else 0."""
    top_k = predicted[:k]
    for pred in top_k:
        for rel in relevant:
            if is_numbers_equal(pred, rel):
                return 1
    return 0


def _reciprocal_rank(predicted: list[str], relevant: list[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    for i, pred in enumerate(predicted, start=1):
        for rel in relevant:
            if is_numbers_equal(pred, rel):
                return 1.0 / i
    return 0.0


def _average_precision_at_k(predicted: list[str], relevant: list[str], k: int) -> float:
    """Average precision considering multiple relevant items (MAP@K)."""
    hits = 0
    score = 0.0
    rel_set = set(normalise_is_number(r) for r in relevant)
    for i, pred in enumerate(predicted[:k], start=1):
        if normalise_is_number(pred) in rel_set:
            hits += 1
            score += hits / i
    if not relevant:
        return 0.0
    return score / min(len(relevant), k)


# ──────────────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────────────

def evaluate(
    predictions: list[dict],
    ground_truth: list[dict],
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """
    Evaluate predictions against ground truth.

    Returns a dict with:
      - per_query  : list of per-query metric dicts
      - aggregate  : macro-averaged metrics
      - missed     : queries with Hit@5 == 0
    """
    # Build ground-truth lookup: query_id → list[normalised IS numbers]
    gt_lookup: dict[str, list[str]] = {}
    for item in ground_truth:
        qid = str(item.get("query_id", ""))
        relevant = [normalise_is_number(r) for r in item.get("relevant", [])]
        gt_lookup[qid] = relevant

    per_query: list[dict] = []
    missed:    list[dict] = []

    hit_totals = {k: 0 for k in k_values}
    rr_total   = 0.0
    ap_totals  = {k: 0.0 for k in k_values}
    n_evaluated = 0

    for pred_item in predictions:
        qid     = str(pred_item.get("query_id", ""))
        query   = pred_item.get("query", "")
        results = pred_item.get("results", [])

        if qid not in gt_lookup:
            # No ground truth for this query — skip
            continue

        relevant  = gt_lookup[qid]
        predicted = [normalise_is_number(r.get("is_number", "")) for r in results]

        hits = {k: _hits_at_k(predicted, relevant, k) for k in k_values}
        rr   = _reciprocal_rank(predicted, relevant)
        aps  = {k: _average_precision_at_k(predicted, relevant, k) for k in k_values}

        for k in k_values:
            hit_totals[k] += hits[k]
            ap_totals[k]  += aps[k]
        rr_total    += rr
        n_evaluated += 1

        row = {
            "query_id":   qid,
            "query":      query,
            "relevant":   relevant,
            "predicted":  predicted,
            "rr":         round(rr, 4),
            **{f"hit@{k}": hits[k] for k in k_values},
            **{f"ap@{k}":  round(aps[k], 4) for k in k_values},
        }
        per_query.append(row)

        if hits.get(max(k_values), 0) == 0:
            missed.append({"query_id": qid, "query": query, "relevant": relevant})

    if n_evaluated == 0:
        print("[eval] WARNING: no predictions matched any ground-truth query_id.")
        return {"per_query": [], "aggregate": {}, "missed": []}

    aggregate = {
        "n_queries":  n_evaluated,
        "MRR":        round(rr_total / n_evaluated, 4),
        **{f"Hit@{k}": round(hit_totals[k] / n_evaluated, 4) for k in k_values},
        **{f"MAP@{k}": round(ap_totals[k]  / n_evaluated, 4) for k in k_values},
    }

    return {
        "per_query": per_query,
        "aggregate": aggregate,
        "missed":    missed,
    }


# ──────────────────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────────────────

def print_report(eval_result: dict) -> None:
    agg   = eval_result.get("aggregate", {})
    pq    = eval_result.get("per_query", [])
    missed = eval_result.get("missed", [])

    width = 56
    print("\n" + "=" * width)
    print("  EVALUATION REPORT")
    print("=" * width)
    print(f"  Queries evaluated : {agg.get('n_queries', 0)}")
    print("-" * width)
    print(f"  {'Metric':<22} {'Value':>10}")
    print("-" * width)

    metrics_order = [
        ("MRR",   "MRR"),
        ("Hit@1", "Hit@1"),
        ("Hit@3", "Hit@3"),
        ("Hit@5", "Hit@5"),
        ("MAP@1", "MAP@1"),
        ("MAP@3", "MAP@3"),
        ("MAP@5", "MAP@5"),
    ]
    for label, key in metrics_order:
        if key in agg:
            val = agg[key]
            bar_len = int(val * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {label:<22} {val:>6.4f}  {bar}")

    print("=" * width)

    # Top misses
    if missed:
        print(f"\n  ⚠ Queries with NO hit in top-{max(5, 1)} results  ({len(missed)})")
        print("-" * width)
        for m in missed[:10]:
            rel_str = ", ".join(m["relevant"][:3])
            print(f"  [{m['query_id']}] {m['query'][:45]!r}")
            print(f"        expected: {rel_str}")
        if len(missed) > 10:
            print(f"  … and {len(missed) - 10} more.")
        print()

    # Per-query breakdown (worst performers)
    if pq:
        worst = sorted(pq, key=lambda x: x.get("rr", 1.0))[:5]
        print("  Worst 5 queries by Reciprocal Rank:")
        print("-" * width)
        for w in worst:
            print(f"  [{w['query_id']}] RR={w['rr']:.3f}  {w['query'][:50]!r}")
            print(f"        predicted : {w['predicted'][:3]}")
            print(f"        relevant  : {w['relevant'][:3]}")
        print()


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate IS-standard retrieval system")
    p.add_argument(
        "--predictions", default=None,
        help="Path to predictions JSON (output of inference.py)"
    )
    p.add_argument(
        "--ground-truth", required=True,
        help="Path to ground truth JSON"
    )
    p.add_argument(
        "--input", default=None,
        help="Input queries JSON (used with --run-inference)"
    )
    p.add_argument(
        "--run-inference", action="store_true",
        help="Run inference.py automatically before evaluating"
    )
    p.add_argument(
        "--output", default=str(RESULTS_DIR / "eval_report.json"),
        help="Where to save the full eval report JSON"
    )
    p.add_argument(
        "--k", nargs="+", type=int, default=[1, 3, 5],
        help="K values for Hit@K and MAP@K (default: 1 3 5)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── optionally run inference first ──────────────────────────────────────
    pred_path: Optional[Path] = None

    if args.run_inference:
        if not args.input:
            print("[ERROR] --input is required when --run-inference is set", file=sys.stderr)
            sys.exit(1)
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        pred_path = Path(tmp.name)
        tmp.close()
        print(f"[eval] Running inference.py → {pred_path}")
        ret = subprocess.run(
            [sys.executable, "inference.py", "--input", args.input, "--output", str(pred_path)],
            check=False,
        )
        if ret.returncode != 0:
            print("[ERROR] inference.py failed.", file=sys.stderr)
            sys.exit(ret.returncode)
    else:
        if not args.predictions:
            print("[ERROR] Provide --predictions or use --run-inference", file=sys.stderr)
            sys.exit(1)
        pred_path = Path(args.predictions)

    # ── load files ──────────────────────────────────────────────────────────
    predictions  = load_json(pred_path)
    ground_truth = load_json(Path(args.ground_truth))

    # ── evaluate ────────────────────────────────────────────────────────────
    result = evaluate(predictions, ground_truth, k_values=tuple(sorted(args.k)))

    # ── report ──────────────────────────────────────────────────────────────
    print_report(result)

    # ── save full report ────────────────────────────────────────────────────
    save_json(result, args.output)
    print(f"[eval] Full report saved → {args.output}")


if __name__ == "__main__":
    main()