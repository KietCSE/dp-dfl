"""Run Trust-Aware D2B-DP from inside dpfl/: python run-trust-aware.py [config.yaml]"""
import sys
import importlib
from pathlib import Path

# Add parent dir to sys.path so 'import dpfl' works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_main_mod = importlib.import_module("dpfl.trust-aware-main")
_main_mod.main()
