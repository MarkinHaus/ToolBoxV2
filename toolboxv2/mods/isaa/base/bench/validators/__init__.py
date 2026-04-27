"""
Validator ABC and registry.
All validators return binary pass/fail. No scales.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.bench.core import CheckResult, TaskContext

_REGISTRY: dict[str, type[Validator]] = {}


def register(name: str):
    """Decorator to register a validator class by name (used in YAML check type)."""
    def decorator(cls: type[Validator]) -> type[Validator]:
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_validator(name: str) -> type[Validator]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown validator '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_validators() -> list[str]:
    return list(_REGISTRY.keys())


def create_validator(check_dict: dict) -> Validator:
    """Instantiate a Validator from a YAML check definition.

    check_dict example: {"type": "contains", "value": "4"}
    The 'type' key selects the validator class, remaining keys become params.
    """
    params = {k: v for k, v in check_dict.items() if k != "type"}
    cls = get_validator(check_dict["type"])
    return cls(**params)


class Validator(ABC):
    """Base class for all validators. Returns CheckResult(passed=bool)."""

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    async def validate(self, ctx: TaskContext) -> CheckResult:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
