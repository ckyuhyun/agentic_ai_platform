"""
Snapshot printer utilities for LangGraph StateSnapshot objects.
"""
from typing import Any, Optional
from langgraph.types import StateSnapshot

from agentic_ai_platform.utils.color_print import cprint, C


def get_snapshot_values(snapshot: StateSnapshot) -> dict:
    """Return the state values dict from a snapshot."""
    return snapshot.values


def get_snapshot_config(snapshot: StateSnapshot) -> dict:
    """Return the configurable config dict from a snapshot."""
    return dict(snapshot.config.get("configurable", {}))


def get_snapshot_metadata(snapshot: StateSnapshot) -> dict:
    """Return the metadata dict from a snapshot."""
    return dict(snapshot.metadata) if snapshot.metadata else {}


def get_snapshot_created_at(snapshot: StateSnapshot) -> Optional[str]:
    """Return the created_at timestamp string from a snapshot."""
    return snapshot.created_at


def print_snapshot(snapshot: Optional[StateSnapshot]):
    """Print a formatted summary of a StateSnapshot."""
    if snapshot is None:
        cprint("  No snapshot available.", C.RED)
        return

    print()
    cprint(f"{'─' * 60}", C.DIM)
    cprint("  StateSnapshot", C.BOLD, C.CYAN)
    cprint(f"{'─' * 60}", C.DIM)

    # ── created_at ────────────────────────────────────────────
    created_at = get_snapshot_created_at(snapshot)
    cprint(f"  created_at  : {created_at or '—'}", C.DIM)

    # ── config ────────────────────────────────────────────────
    print()
    cprint("  config", C.BOLD, C.YELLOW)
    config = get_snapshot_config(snapshot)
    if config:
        for key, value in config.items():
            cprint(f"    {key}: {value}", C.DIM)
    else:
        cprint("    (empty)", C.DIM)

    # ── metadata ──────────────────────────────────────────────
    print()
    cprint("  metadata", C.BOLD, C.YELLOW)
    metadata = get_snapshot_metadata(snapshot)
    if metadata:
        for key, value in metadata.items():
            cprint(f"    {key}: {value}", C.DIM)
    else:
        cprint("    (empty)", C.DIM)

    # ── values ────────────────────────────────────────────────
    print()
    cprint("  values", C.BOLD, C.YELLOW)
    values = get_snapshot_values(snapshot)
    if values:
        for key, value in values.items():
            _print_value(key, value)
    else:
        cprint("    (empty)", C.DIM)

    # ── next nodes ────────────────────────────────────────────
    if snapshot.next:
        print()
        cprint(f"  next        : {' → '.join(snapshot.next)}", C.MAGENTA)

    print()
    cprint(f"{'─' * 60}", C.DIM)
    print()


def _print_value(key: str, value: Any, indent: int = 4):
    """Recursively print a state value with indentation."""
    pad = " " * indent
    if isinstance(value, list):
        cprint(f"{pad}{key}: [{len(value)} item(s)]", C.CYAN)
        for i, item in enumerate(value[:3]):          # show first 3
            cprint(f"{pad}  [{i}] {str(item)[:120]}", C.DIM)
        if len(value) > 3:
            cprint(f"{pad}  … {len(value) - 3} more", C.DIM)
    elif isinstance(value, dict):
        cprint(f"{pad}{key}:", C.CYAN)
        for k, v in value.items():
            cprint(f"{pad}  {k}: {str(v)[:120]}", C.DIM)
    elif hasattr(value, "model_dump"):                # Pydantic model
        cprint(f"{pad}{key}: {type(value).__name__}", C.CYAN)
        for k, v in value.model_dump().items():
            cprint(f"{pad}  {k}: {str(v)[:120]}", C.DIM)
    else:
        cprint(f"{pad}{key}: {str(value)[:120]}", C.CYAN)
