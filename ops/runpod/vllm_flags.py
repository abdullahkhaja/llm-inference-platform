"""vLLM v0.8.5 launch flags that must be explicit.

Prefix caching defaults differ by engine: off on V0, on on V1. The Phase 2
L4 smoke started a V1 engine, so omitting the flag would silently enable
caching. Always pass one of the two flags.
"""

from __future__ import annotations

import sys

_ON = {"1", "true", "yes", "on"}
_OFF = {"0", "false", "no", "off"}


def prefix_caching_arg(value: str) -> str:
    """Map ENABLE_PREFIX_CACHING to a vLLM 0.8.5 CLI flag."""
    normalized = value.strip().lower()
    if normalized in _ON:
        return "--enable-prefix-caching"
    if normalized in _OFF:
        return "--no-enable-prefix-caching"
    raise ValueError(
        f"ENABLE_PREFIX_CACHING must be true or false, got {value!r}"
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: vllm_flags.py <true|false>", file=sys.stderr)
        return 2
    try:
        print(prefix_caching_arg(args[0]))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
