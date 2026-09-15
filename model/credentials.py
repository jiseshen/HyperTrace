"""Credential checks shared by generation and embedding clients."""

import os


def require_api_key(variable: str, explicit: str | None = None) -> str:
    value = explicit or os.getenv(variable)
    if not value or not value.strip():
        raise ValueError(f"Set {variable} before using this provider.")
    return value
