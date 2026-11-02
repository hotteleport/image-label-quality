"""Compare classifier quality on raw, deduplicated and reviewed training data.

Usage: python run_experiment.py IMAGE_ROOT

The test set is a random sample of duplicate groups chosen without using the
model and reviewed by eye, so cleaning cannot flatter the score by removing
hard test images.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from label_quality.audit import duplicate_groups, scan, summarise
from label_quality.model import classifier, embed, group_split

HERE = Path(__file__).parent
SEED = 20261102
HOLDOUT_GROUPS = 240
BOOTSTRAPS = 5000


def bootstrap_interval(values: np.ndarray) -> list:
    rng = np.random.default_rng(SEED)
    means = rng.choice(values, (BOOTSTRAPS, len(values))).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def propagate(items: list, groups: list, review: dict) -> dict:
    """Apply a decision made on any file to its whole duplicate group."""
    decisions = {}
    for group in groups:
        found = [review[items[i].path] for i in group if items[i].path in review]
        if found:
            for i in group:
                decisions[i] = found[0]
    return decisions


def main(root: Path) -> None:
    items = scan(root)
    groups = sorted(duplicate_groups(items), key=lambda group: group[0])
    audit = summarise(items, groups)

    shuffled = groups[:]
    random.Random(SEED).shuffle(shuffled)
    holdout, pool = shuffled[:HOLDOUT_GROUPS], shuffled[HOLDOUT_GROUPS:]

    gold = json.loads((HERE / "reviews/holdout_review.json").read_text())
    flagged = json.loads((HERE / "reviews/flagged_review.json").read_text())
    x = embed([str(root / item.path) for item in items])
    y = np.array([int(item.label == "reindeer") for item in items])
    decisions = propagate(items, groups, flagged)

    test = [g[0] for g in holdout if gold[items[g[0]].path] in ("moose", "reindeer")]
    test_y = np.array([int(gold[items[i].path] == "reindeer") for i in test])

    raw = [i for g in pool for i in g]
    dedup = [g[0] for g in pool if len({items[i].label for i in g}) == 1]
    reviewed, reviewed_y = [], []
    for g in pool:
        decision = decisions.get(g[0], "keep")
        if decision.startswith("relabel:"):
            reviewed.append(g[0])
            reviewed_y.append(int(decision.endswith("reindeer")))
        elif decision == "keep" and len({items[i].label for i in g}) == 1:
            reviewed.append(g[0])
            reviewed_y.append(y[g[0]])

    def score(train: list, labels: np.ndarray) -> dict:
        predictions = classifier().fit(x[train], labels).predict(x[test])
        correct = (predictions == test_y).astype(float)
        return {
            "train_images": len(train),
            "accuracy": float(correct.mean()),
            "ci95": bootstrap_interval(correct),
            "correct": correct,
        }

    conditions = {
        "raw": score(raw, y[raw]),
        "deduplicated": score(dedup, y[dedup]),
        "reviewed": score(reviewed, np.array(reviewed_y)),
    }
    difference = conditions["reviewed"]["correct"] - conditions["raw"]["correct"]
    difference_ci = bootstrap_interval(difference)
    for condition in conditions.values():
        condition.pop("correct")

    # Same raw labels, with and without duplicates crossing the split.
    naive, grouped = [], []
    membership = np.zeros(len(items), dtype=int)
    for group_id, group in enumerate(groups):
        membership[group] = group_id
    indices = np.arange(len(items))
    for seed in range(20):
        a, b = train_test_split(indices, test_size=0.2, stratify=y, random_state=seed)
        naive.append(classifier().fit(x[a], y[a]).score(x[b], y[b]))
        a, b = group_split(y, membership, seed)
        grouped.append(classifier().fit(x[a], y[a]).score(x[b], y[b]))

    sample = Counter(gold.values())
    usable = sample["moose"] + sample["reindeer"]
    label_errors = sum(gold[items[g[0]].path] in ("moose", "reindeer")
                       and gold[items[g[0]].path] != items[g[0]].label for g in holdout)
    review_counts = Counter("relabel" if v.startswith("relabel") else v for v in flagged.values())
    results = {
        "audit": audit,
        "random_sample": {
            "groups_reviewed": HOLDOUT_GROUPS,
            "counts": dict(sample),
            "usable_share": usable / HOLDOUT_GROUPS,
            "usable_ci95": bootstrap_interval(np.array([1.0] * usable + [0.0] * (HOLDOUT_GROUPS - usable))),
            "wrong_class_among_usable": label_errors,
        },
        "flagged_review": dict(review_counts),
        "test_images": len(test),
        "test_by_label": {"moose": int((test_y == 0).sum()), "reindeer": int((test_y == 1).sum())},
        "conditions": conditions,
        "reviewed_minus_raw_ci95": difference_ci,
        "raw_label_accuracy": {
            "random_split_mean": float(np.mean(naive)),
            "group_split_mean": float(np.mean(grouped)),
            "difference": float(np.mean(naive) - np.mean(grouped)),
        },
    }
    (HERE / "results/metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
