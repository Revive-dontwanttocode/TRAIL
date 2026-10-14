"""Normalize existing preprocessed CSV columns without changing their splits."""

import argparse
import pandas as pd

from src.common.config import load_config, project_path, write_json


def prepare(config):
    records = {}
    for name, settings in config["files"].items():
        columns = settings["columns"]
        frame = pd.read_csv(
            project_path(settings["input_path"]),
            dtype={columns.get("item_id", "item_id"): str},
        )
        frame = frame.rename(
            columns={original: canonical for canonical, original in columns.items()},
            errors="raise",
        )
        output_path = project_path(settings["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        records[name] = {"rows": len(frame), "output_path": str(output_path)}
        print(f"Saved {len(frame)} rows to {output_path}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config = load_config(project_path(args.config))
    records = prepare(config)
    run_dir = project_path(args.run_dir)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "results.json", {
        "task": "Normalize existing TRAIL CSVs", "dataset": config["dataset"],
        "config": config, "files": records,
    })
