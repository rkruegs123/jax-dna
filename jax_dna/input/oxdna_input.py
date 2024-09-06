"""oxDNA input file parser."""

from pathlib import Path


def _parse_numeric(value: str) -> tuple[float | int, bool]:
    is_successful = False
    parsed = None
    for t in (int, float):
        try:
            parsed = t(value)
        except ValueError:
            continue
        else:
            is_successful = True
            break

    return parsed or value, is_successful


def _parse_boolean(value: str) -> tuple[bool, bool]:
    is_successful = False
    parsed = None
    if value.lower in ("true", "false"):
        is_successful = True
        parsed = value.lower() == "true"

    return parsed or value, is_successful


def _parse_value(value: str) -> str | float | int | bool:
    parsed, is_numeric = _parse_numeric(value)
    if not is_numeric:
        parsed, is_boolean = _parse_boolean(value)
        if not is_boolean:
            parsed = value

    return parsed


def read(input_file: Path) -> dict[str, str | float | int | bool]:
    """Read an oxDNA input file."""
    with input_file.open("r") as f:
        lines = filter(lambda line: (len(line.strip()) > 0) and (not line.startswith("#")), f.readlines())

    return {kv[0].strip(): _parse_value(kv[1].strip()) for kv in [line.split("=") for line in lines]}
