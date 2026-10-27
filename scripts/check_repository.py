"""Check committed metadata without installing experiment dependencies."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name in ("configs/benchmark.json", ".github/reviewers.json", "results/environment-template.json"):
    json.loads((ROOT / name).read_text(encoding="utf-8"))
config = json.loads((ROOT / "configs/benchmark.json").read_text())
assert len(config["datasets"]) * len(config["variants"]) * len(config["platforms"]) * len(config["threads"]) == 48
assert config["warmup_iterations"] == 50 and config["measurement_iterations"] == 1000
with (ROOT / "results/summary-template.csv").open() as file:
    headers = next(csv.reader(file))
assert len(headers) == len(set(headers))
for name in ("README.md", "CONTRIBUTING.md", "docs/setup.md", "docs/github-rules.md"):
    assert (ROOT / name).is_file(), name
print("Repository metadata checks passed.")
