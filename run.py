"""Run from inside dpfl/: python run.py [config.yaml]"""
import sys
from pathlib import Path

# Add parent dir to sys.path so 'import dpfl' works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dpfl.__main__ import main  # noqa: E402

main()
