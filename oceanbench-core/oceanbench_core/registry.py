"""Minimal registry for model and component lookup (stub for now)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a factory or class under a string name."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY[name] = fn
        return fn

    return decorator


def get(name: str) -> Optional[Callable[..., Any]]:
    """Return the registered callable for `name`, or None."""
    return _REGISTRY.get(name)


def list_registered() -> list[str]:
    """Return all registered names."""
    return list(_REGISTRY.keys())
