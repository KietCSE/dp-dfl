"""Run Noise Game DFL from inside dpfl/: python run-noise-game.py [config.yaml]"""
import sys
import importlib
from pathlib import Path

# Add parent dir to sys.path so 'import dpfl' works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_main_mod = importlib.import_module("dpfl.noise-game-main")
_main_mod.main()
