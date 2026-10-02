"""MIND-compatible selector contract; independent baseline preparation."""
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from numbers import Integral
from typing import Any, Mapping
import json
import warnings

import torch


EP_OUTPUT = "moe_packed_weighted_output"
BUILTIN_ROUND_ROBIN = "megatron.baseline_sampling.round_robin"


def round_robin(iteration: int, num_sources: int, count: int, offset: int = 0):
    """Return source positions, rotating by count each global batch."""
    if any(isinstance(value, bool) or not isinstance(value, Integral)
           for value in (iteration, num_sources, count, offset)):
        raise TypeError("round-robin iteration, num_sources, count and offset must be integers")
    if not 1 <= count <= num_sources:
        raise ValueError("source sampling count must be between 1 and num_sources")
    start = (iteration * count + offset) % num_sources
    return tuple((start + j) % num_sources for j in range(count))


@lru_cache(maxsize=64)
def _load_selector(path: str):
    if path == "dmi_megatron_integration.hooks.source_sampling.round_robin":
        return round_robin
    module, sep, name = path.rpartition(".")
    if not sep:
        raise ValueError("source sampling function must be a dotted Python name")
    function = getattr(import_module(module), name)
    if not callable(function):
        raise TypeError("source sampling function must be callable")
    return function


@dataclass(frozen=True)
class SourceSampling:
    function: str
    args: Mapping[str, Any]
    iteration_field: str = "global_batch_id"
    iteration_origin: int = 1

    def __post_init__(self):
        if not isinstance(self.function, str) or not self.function:
            raise ValueError("source sampling requires a function name")
        if self.iteration_field != "global_batch_id":
            raise ValueError("source sampling iteration_field must be global_batch_id")
        if not isinstance(self.iteration_origin, int) or isinstance(self.iteration_origin, bool):
            raise TypeError("source sampling iteration_origin must be an integer")
        if not isinstance(self.args, Mapping):
            raise TypeError("source sampling args must be a mapping")
        # Persist exactly the same JSON-compatible arguments on both ends.
        object.__setattr__(self, "args", json.loads(json.dumps(dict(self.args), allow_nan=False)))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping):
            raise TypeError("source_sampling must be a mapping")
        unknown = set(value) - {"function", "args", "iteration_field", "iteration_origin"}
        if unknown:
            raise ValueError(f"Unknown source_sampling fields: {sorted(unknown)}")
        return cls(**value)

    def to_dict(self):
        return dict(function=self.function, args=dict(self.args),
                    iteration_field=self.iteration_field, iteration_origin=self.iteration_origin)

    def select(self, global_batch_id: int, num_sources: int) -> tuple[int, ...]:
        selected = tuple(_load_selector(self.function)(
            iteration=int(global_batch_id) - self.iteration_origin,
            num_sources=int(num_sources), **self.args,
        ))
        if not selected or any(isinstance(u, bool) or not isinstance(u, Integral) for u in selected):
            raise ValueError("source selector must return nonempty integer positions")
        if len(set(selected)) != len(selected) or any(not 0 <= u < num_sources for u in selected):
            raise ValueError("source selector returned duplicate or out-of-range positions")
        # Pack in physical order, irrespective of a plugin's enumeration order.
        return tuple(sorted(int(u) for u in selected))


def read_source_sampling(path: str | None, *, ep_enabled: bool) -> SourceSampling | None:
    """Read hook-scoped YAML; ignored hooks never import their plugins."""
    if not path:
        return None
    import yaml
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, Mapping) or set(config) != {"hooks"}:
        raise ValueError("Baseline hook configuration must contain a hooks mapping")
    hooks = config["hooks"]
    if not isinstance(hooks, Mapping):
        raise TypeError("hooks must be a mapping")
    policy = None
    for name, settings in hooks.items():
        if not isinstance(settings, Mapping):
            raise TypeError(f"Hook configuration for {name} must be a mapping")
        if "source_sampling" not in settings or settings["source_sampling"] is None:
            continue
        if name != EP_OUTPUT:
            warnings.warn(f"Baseline hook {name}: ignoring unsupported source_sampling", stacklevel=2)
        elif ep_enabled:
            policy = SourceSampling.from_dict(settings["source_sampling"])
    return policy
