import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

steps = [
    ("Download FER2013", [sys.executable, str(ROOT / 'scripts' / 'download_fer2013.py')]),
    ("EDA+Prepare FER2013", [sys.executable, str(ROOT / 'scripts' / 'eda_and_prepare_fer.py')]),
    ("EDA+Prepare Text", [sys.executable, str(ROOT / 'scripts' / 'eda_and_prepare_text.py')]),
    ("Make Report", [sys.executable, str(ROOT / 'scripts' / 'make_report.py')]),
]


def run(name, cmd):
    print(f"\n=== {name} ===")
    try:
        res = subprocess.run(cmd, check=False)
        if res.returncode != 0:
            print(f"Step '{name}' exited with code {res.returncode} (continuing)")
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")


def main():
    print("Running full pipeline...")
    for name, cmd in steps:
        run(name, cmd)
    print("\nAll steps done. See outputs/plots and REPORT.md")


if __name__ == '__main__':
    main()
