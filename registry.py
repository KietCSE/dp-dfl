"""Component registry for pluggable dataset, model, noise, aggregator, attack classes."""

from typing import Dict, Type

DATASETS: Dict[str, Type] = {}
MODELS: Dict[str, Type] = {}
NOISE_MECHANISMS: Dict[str, Type] = {}
AGGREGATORS: Dict[str, Type] = {}
ATTACKS: Dict[str, Type] = {}


def register(registry: Dict[str, Type], name: str):
    """Decorator to register a component class into a registry dict."""
    def decorator(cls):
        registry[name] = cls
        return cls
    return decorator
