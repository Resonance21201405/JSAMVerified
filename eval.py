"""
eval.py — Evaluate IS-standard retrieval output.

Input  : output from inference.py (sample_output.json format)
         Keys per record: id, query, expected_standards,
                          retrieved_standards, latency_seconds

Metrics
-------
  Hit@3        % of queries where ≥1 expected standard is in top-3  (target > 80%)
  MRR@5        Mean Reciprocal Rank of first correct result in top-5 (target > 0.7)
  Avg Latency  Average latency_seconds across queries                (target < 5 s)

Usage
-----
  python eval.py --predictions results/output.json
  python eval.py                          # uses default path above
"""

import argparse
from src.utils import load_json, normalise_is_number


def evaluate(predictions: list[dict]) -> dict:
    """
    Evaluate a list of prediction records and print a results report.

    Each record must have:
      expected_standards   : list[str]   correct IS number(s)
      retrieved_standards  : list[str]   ordered retrieved IS numbers
      latency_seconds      : float       per-query response time
    """
    total = len(predictions)
    if total == 0:
        print("[WARN] No predictions to evaluate.")
        return {"hit_at_3": 0.0, "mrr_at_5": 0.0, "avg_latency_seconds": 0.0}

    hit3_count    = 0
    mrr5_sum      = 0.0
    total_latency = 0.0

    for p in predictions:
        expected  = [normalise_is_number(s) for s in p.get("expected_standards", [])]
        retrieved = [normalise_is_number(s) for s in p.get("retrieved_standards", [])]

        # Hit@3 — at least one expected standard in the first 3 results
        if any(s in retrieved[:3] for s in expected):
            hit3_count += 1

        # MRR@5 — reciprocal rank of the FIRST correct result in top-5
        for rank, s in enumerate(retrieved[:5], start=1):
            if s in expected:
                mrr5_sum += 1.0 / rank
                break

        total_latency += float(p.get("latency_seconds", 0.0))

    hit_at_3    = round(hit3_count / total * 100, 2)   # expressed as %
    mrr_at_5    = round(mrr5_sum   / total,       4)
    avg_latency = round(total_latency / total,    4)

    def badge(passed): return "✓ PASS" if passed else "✗ FAIL"

    print("=" * 52)
    print("  EVALUATION RESULTS")
    print("=" * 52)
    print(f"  Queries evaluated  : {total}")
    print(f"  Hit@3              : {hit_at_3:6.2f}%   (target > 80%)")
    print(f"  MRR@5              : {mrr_at_5:7.4f}   (target > 0.70)")
    print(f"  Avg Latency        : {avg_latency:7.4f}s  (target < 5s)")
    print("=" * 52)
    print(f"  Hit@3    {badge(hit_at_3  > 80.0)}")
    print(f"  MRR@5    {badge(mrr_at_5  >  0.7)}")
    print(f"  Latency  {badge(avg_latency < 5.0)}")
    print("=" * 52)

    return {
        "hit_at_3":            hit_at_3,
        "mrr_at_5":            mrr_at_5,
        "avg_latency_seconds": avg_latency,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IS-standard retrieval")
    parser.add_argument(
        "--predictions",
        default="results/output.json",
        help="Path to inference output JSON  (default: results/output.json)",
    )
    args = parser.parse_args()

    predictions = load_json(args.predictions)
    evaluate(predictions)