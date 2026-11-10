"""Load the generated dataset (data/expanded) straight into a ContextStore, with no HTTP.

Used by the decision report and by tests that want the real contexts; the same loader will
produce the offline submission file for the 30 canonical test pairs.
"""
from __future__ import annotations

import json
from pathlib import Path

from .store import ContextStore

# folder, store scope, key of the id inside each file
LAYOUT = (
    ("categories", "category", "slug"),
    ("merchants", "merchant", "merchant_id"),
    ("customers", "customer", "customer_id"),
    ("triggers", "trigger", "id"),
)


def load_store(data_dir: Path) -> ContextStore:
    store = ContextStore()
    for folder, scope, id_key in LAYOUT:
        for path in sorted((data_dir / folder).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            store.put(scope, payload[id_key], 1, payload)
    return store


def load_pairs(data_dir: Path) -> list[dict]:
    """The canonical test pairs: [{test_id, trigger_id, merchant_id, customer_id}, ...]."""
    return json.loads((data_dir / "test_pairs.json").read_text(encoding="utf-8"))["pairs"]
