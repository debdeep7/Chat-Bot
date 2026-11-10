"""Push the expanded challenge dataset into a running bot, like the judge's warm-up does.

    python scripts/load_dataset.py --dir data/expanded --url http://localhost:8080
    python scripts/load_dataset.py --triggers      # also push the 100 sample triggers

The real warm-up sends 5 categories + 50 merchants + 200 customers (255 contexts) and no
triggers; triggers arrive later, during the test window.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

# (sub-directory, scope, key inside the payload that holds the context id)
LAYOUT = [
    ("categories", "category", "slug"),
    ("merchants", "merchant", "merchant_id"),
    ("customers", "customer", "customer_id"),
    ("triggers", "trigger", "id"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Push the expanded dataset into a running bot.")
    parser.add_argument("--dir", default="data/expanded", help="folder made by generate_dataset.py")
    parser.add_argument("--url", default="http://localhost:8080", help="bot base URL")
    parser.add_argument("--triggers", action="store_true", help="also push triggers")
    args = parser.parse_args()

    root = Path(args.dir)
    base = args.url.rstrip("/")
    delivered_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    pushed = failed = 0

    with httpx.Client(timeout=10) as client:
        for sub, scope, id_key in LAYOUT:
            if sub == "triggers" and not args.triggers:
                continue
            files = sorted((root / sub).glob("*.json"))
            if not files:
                print(f"warning: no files in {root / sub}", file=sys.stderr)
            for path in files:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if id_key not in payload:
                    print(f"skip {path.name}: no '{id_key}'", file=sys.stderr)
                    failed += 1
                    continue
                response = client.post(
                    f"{base}/v1/context",
                    json={
                        "scope": scope,
                        "context_id": payload[id_key],
                        "version": 1,
                        "payload": payload,
                        "delivered_at": delivered_at,
                    },
                )
                if response.status_code == 200 and response.json().get("accepted"):
                    pushed += 1
                else:
                    failed += 1
                    print(f"rejected {scope} {payload[id_key]}: {response.status_code} {response.text}", file=sys.stderr)

        print(f"pushed {pushed}, failed {failed}")
        print("healthz:", client.get(f"{base}/v1/healthz").json())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
