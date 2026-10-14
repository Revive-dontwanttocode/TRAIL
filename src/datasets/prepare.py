"""Materialize original 30-day splits and verify the entire supplied test cohort."""

import argparse
import csv
import gzip
import json
import platform
import shlex
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.Common.config import (
    PROJECT_ROOT, load_config, project_path, sha256, write_json,
)


def boundaries(period):
    origin = datetime.fromisoformat(period["origin_utc"]).replace(tzinfo=timezone.utc)
    width = timedelta(days=period["days"])
    start = origin + period["test_index"] * width
    return tuple(int(value.timestamp()) for value in (start - width, start, start + width))


def split_events(events, candidates, valid_start, test_start, test_end):
    """Events: (raw reviewer ID, source item ID, rating, timestamp, source order)."""
    train = [r for r in events if r[3] < valid_start]
    valid = [r for r in events if valid_start <= r[3] < test_start]
    known_users = {r[0] for r in train + valid}
    test = [r for r in events if test_start <= r[3] < test_end
            and r[1] in candidates and r[0] in known_users]
    return {"train": train, "valid": valid, "test": test}


def profile_groups(events):
    profiles = defaultdict(list)
    for user, item, rating, timestamp, *_ in events:
        profiles[user].append((item, rating, timestamp))
    groups = defaultdict(list)
    for user, profile in profiles.items():
        groups[tuple(sorted(profile))].append(user)
    return groups


def match_test_profiles(original, reconstructed):
    """Prove full user-profile equivalence, without guessing ambiguous old IDs."""
    left, right = profile_groups(original), profile_groups(reconstructed)
    if Counter({p: len(u) for p, u in left.items()}) != Counter({p: len(u) for p, u in right.items()}):
        raise ValueError("Reconstructed test user profiles differ from the original file")
    unique, ambiguous = {}, []
    for profile in sorted(left):
        if len(left[profile]) == 1:
            unique[right[profile][0]] = left[profile][0]
        else:
            ambiguous.append({"original_user_ids": sorted(left[profile]),
                              "reviewer_ids": sorted(right[profile]),
                              "events_original_item_id_rating_timestamp": list(profile)})
    return unique, ambiguous


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def statistics(rows):
    return {"events": len(rows), "users": len({r[0] for r in rows}),
            "items": len({r[1] for r in rows}),
            "min_timestamp_utc": datetime.fromtimestamp(min(r[3] for r in rows), timezone.utc).isoformat(),
            "max_timestamp_utc": datetime.fromtimestamp(max(r[3] for r in rows), timezone.utc).isoformat()}


def prepare(config, output_dir=None):
    paths, sources = {}, []
    for role, spec in config["sources"].items():
        path = project_path(spec["path"])
        paths[role] = path
        sources.append(dict(spec, role=role, bytes=path.stat().st_size))
    with paths["metadata"].open(encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    catalog = {r["item"]: int(r["item_id"]) for r in metadata}
    if len(catalog) != len(metadata) or set(catalog.values()) != set(range(len(catalog))):
        raise ValueError("Expected a unique contiguous source item catalog")
    with paths["candidates"].open(encoding="utf-8", newline="") as handle:
        candidate_rows = list(csv.DictReader(handle))
    candidates = {int(r["item_id"]) for r in candidate_rows}
    if len(candidates) != len(candidate_rows) or not candidates <= set(catalog.values()):
        raise ValueError("Duplicate or unknown candidate IDs")
    if any(r["predict_timestamp"] != config["period"]["test_label"] for r in candidate_rows):
        raise ValueError("Candidate label does not match test period label")
    valid_start, test_start, test_end = boundaries(config["period"])
    events = []
    raw_rows = 0
    with gzip.open(paths["raw"], "rt", encoding="utf-8") as handle:
        for order, line in enumerate(handle):
            row = json.loads(line)
            raw_rows += 1
            if row["asin"] in catalog:
                events.append((row["reviewerID"], catalog[row["asin"]], float(row["overall"]),
                               int(row["unixReviewTime"]), order))
    splits = split_events(events, candidates, valid_start, test_start, test_end)
    original = []
    for line in paths["test"].read_text(encoding="utf-8").splitlines():
        user, item, rating, stamp = line.split()
        original.append((int(user), int(item), float(rating), int(stamp)))
    unique, ambiguous = match_test_profiles(original, splits["test"])
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"Empty {name} split")
    # Vocabulary is formed before test time. Future-only identities never enter it.
    reviewers = sorted({r[0] for r in splits["train"] + splits["valid"]})
    user_ids = {reviewer: index + 1 for index, reviewer in enumerate(reviewers)}
    for rows in splits.values():
        rows.sort(key=lambda r: (user_ids[r[0]], r[3], r[4]))
    train_users = {r[0] for r in splits["train"]}
    valid_evaluation = [r for r in splits["valid"] if r[0] in train_users]
    dataset_users = [{"raw_id": user, "train": [], "valid": [], "test": []} for user in reviewers]
    for name, rows in splits.items():
        for user, item, *_ in rows:
            dataset_users[user_ids[user] - 1][name].append(item + 1)
    by_source_id = {source_id: asin for asin, source_id in catalog.items()}
    dataset = {"schema_version": 1, "item_ids": [by_source_id[i] for i in range(len(catalog))],
               "catalog_item_ids": [str(i) for i in range(len(catalog))], "users": dataset_users,
               "evaluation_candidates": {"valid": list(range(1, len(catalog) + 1)),
                                         "test": [i + 1 for i in sorted(candidates)]}}
    directory = project_path(output_dir or config["output_dir"])
    directory.mkdir(parents=True, exist_ok=False)
    columns = ["user_id", "item_id", "rating", "timestamp"]
    for name, rows in dict(splits, val_evaluation=valid_evaluation).items():
        filename = "val" if name == "valid" else name
        write_csv(directory / f"{filename}.csv", columns,
                  ((user_ids[u], i + 1, r, t) for u, i, r, t, _ in rows))
    ambiguous_reviewers = {u for group in ambiguous for u in group["reviewer_ids"]}
    write_csv(directory / "user_mapping.csv", ["user_id", "reviewer_id", "original_test_user_id", "original_id_status"],
              ((user_ids[u], u, unique.get(u, ""), "unique" if u in unique else
                "ambiguous_identical_test_profile" if u in ambiguous_reviewers else "not_in_test") for u in reviewers))
    write_csv(directory / "item_mapping.csv", ["item_id", "original_item_id", "asin"],
              ((i + 1, i, by_source_id[i]) for i in range(len(catalog))))
    write_csv(directory / "test_candidates.csv", ["item_id", "original_item_id"],
              ((i + 1, i) for i in sorted(candidates)))
    write_json(directory / "ambiguous_original_user_ids.json", {
        "policy": "Do not assign ambiguous original numeric IDs. Train and test use true raw reviewer identities.",
        "groups": ambiguous})
    shutil.copyfile(paths["test"], directory / "test_original.txt")
    write_json(directory / "dataset.json", dataset)
    write_json(directory / "config.json", config)
    manifest = {
        "schema_version": 1, "experiment": config["experiment"],
        "protocol": "original_30_day_periods_verified_test_profiles",
        "sources": sources, "raw_review_events": raw_rows, "catalog_review_events": len(events),
        "data_config": {"mode": "provided_splits", "period": config["period"],
                        "valid_start": datetime.fromtimestamp(valid_start, timezone.utc).isoformat(),
                        "test_start": datetime.fromtimestamp(test_start, timezone.utc).isoformat(),
                        "test_end": datetime.fromtimestamp(test_end, timezone.utc).isoformat(),
                        "columns": {"user": "user_id", "item": "item_id", "timestamp": "timestamp"},
                        "format": "csv", "has_header": True},
        "splits": {name: statistics(rows) for name, rows in splits.items()},
        "validation_evaluation": statistics(valid_evaluation),
        "users": len(reviewers), "items": len(catalog), "test_candidates": len(candidates),
        "mapping": {"csv_user_id": "1-based; dataset.json user index = user_id - 1",
                    "csv_and_model_item_id": "original_item_id + 1; zero reserved for padding",
                    "unique_original_test_user_ids": len(unique),
                    "ambiguous_original_test_user_ids": len(ambiguous_reviewers),
                    "ambiguous_groups": len(ambiguous),
                    "original_test_profile_multiset_equal": True},
        "coverage": {"test_users_with_train_history": len({r[0] for r in splits["test"]} & train_users),
                     "test_users_with_validation_only_history": len({r[0] for r in splits["test"]} - train_users),
                     "validation_users_without_train_history": len({r[0] for r in splits["valid"]} - train_users)},
        "policies": {"train": "All catalog reviews strictly before validation start; no rating threshold",
                     "validation": "All catalog reviews in preceding 30-day period, including new users",
                     "validation_evaluation": "Validation targets of users with pre-validation history; all catalog candidates",
                     "test": "Fixed CSV candidates; users with catalog history before test; verified against original file",
                     "history": "Retain all validation events for test context, including validation-only users",
                     "same_timestamp_order": "Original raw-file line order"},
        "environment": {"command": shlex.join([sys.executable, *sys.orig_argv[1:]]),
                        "python": platform.python_version(), "seed": None,
                        "source_sha256": {str(Path(__file__).relative_to(PROJECT_ROOT)): sha256(__file__)}},
        "dataset_sha256": sha256(directory / "dataset.json"),
    }
    readme = """# Amazon Home & Kitchen — original 30-day periods

`2014-06` is a label. UTC boundaries: train before 2014-05-19;
validation [2014-05-19, 2014-06-18); test [2014-06-18, 2014-07-18).

- `train.csv`, `val.csv`, `test.csv`: user_id,item_id,rating,timestamp (Unix seconds).
- IDs are positive and consistent across splits. Item 0 is padding, not a source item.
- `item_mapping.csv`: model item_id = original_item_id + 1; ASIN crosswalk.
- `user_mapping.csv`: user_id -> true raw reviewer ID; uniquely recoverable old test IDs.
- `test_original.txt`: unchanged original file, retaining its original numeric IDs.
- `ambiguous_original_user_ids.json`: identical test-profile groups; no guessed old IDs.
- `test_candidates.csv`: the fixed 4,915 candidate items, in both ID namespaces.
- `val_evaluation.csv`: validation targets of users with training history. Keep all
  `val.csv` events when constructing test histories, including users first seen here.
- `dataset.json`: shared baseline schema; users[user_id - 1], 1-based item IDs.
- `manifest.json`: counts, boundaries, policies, checksums and test-equivalence proof.

All 6,437 original test users and 10,730 interactions are represented. Complete
per-user multisets of (source item, rating, timestamp), including multiplicities,
match the original file exactly. Identical test profiles leave some old numeric
IDs ambiguous; true reviewer histories and labels remain fully identified in the
new namespace. Do not join the old and new numeric IDs directly.

187 test users have validation-only history. Personalized baselines must explicitly
handle them (for example, refit on train+validation after validation selection);
do not silently remove them to change the benchmark denominator. Earlier saved
calendar checkpoints and their data fingerprints do not apply to these splits.
"""
    (directory / "README.md").write_text(readme, encoding="utf-8")
    manifest["files"] = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                         for p in sorted(directory.iterdir()) if p.is_file()}
    write_json(directory / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/Datasets/AmazonHomeAndKitchen/configs/OriginalPeriods.json")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    record = prepare(load_config(args.config), args.output_dir)
    print(json.dumps({key: record[key] for key in ("splits", "validation_evaluation", "mapping", "coverage", "dataset_sha256")}, indent=2))


if __name__ == "__main__":
    main()
