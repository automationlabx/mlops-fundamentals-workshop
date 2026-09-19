from __future__ import annotations

import subprocess
import sys


def main() -> None:
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q"], check=False)
    if completed.returncode != 0:
        print("LOCAL QUALITY CHECKS: FAIL")
        raise SystemExit(completed.returncode)
    print("LOCAL QUALITY CHECKS: PASS")


if __name__ == "__main__":
    main()
