"""Export canonical split/scaler artifacts for external baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataio.canonical_setup import build_canonical_setup, save_canonical_setup
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Export train/val/test window indices and train-fitted scaler stats "
            "for external baseline repos."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        required=True,
        help="Config path. Pass multiple --config flags for multiple datasets.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/canonical_setup"),
        help="Directory for exported .npz/.json files.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="canonical",
        help="Filename tag, e.g. canonical or h12_t12.",
    )
    return parser.parse_args()


def main() -> None:
    """Export canonical setup for one or more configs."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for config_path in args.config:
        config = load_config(config_path)
        setup = build_canonical_setup(config)
        dataset = setup.metadata.dataset
        out_prefix = args.out_dir / f"{dataset}_{args.tag}"
        paths = save_canonical_setup(setup, out_prefix=out_prefix)
        print(
            "Exported",
            dataset,
            f"train_windows={setup.metadata.train_windows}",
            f"val_windows={setup.metadata.val_windows}",
            f"test_windows={setup.metadata.test_windows}",
            f"npz={paths['npz']}",
            f"json={paths['json']}",
        )


if __name__ == "__main__":
    main()
