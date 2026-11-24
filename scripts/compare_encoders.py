"""Superseded by scripts/ablate.py (kept for the old command line):

    python scripts/compare_encoders.py --dataset runs/block/dataset.pt --out runs/block/encoders
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    if "--group" not in sys.argv and "--variants" not in sys.argv:
        sys.argv += ["--group", "encoders"]
    from scripts.ablate import main

    main()
