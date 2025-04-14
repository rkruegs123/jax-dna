"""oxDNA input file parser."""

import io
import typing
from pathlib import Path

INVALID_DICT_LINE = "Invalid dictionary line: {}"


def _parse_numeric(value: str) -> tuple[float | int, bool]:
    is_successful = False
    parsed = 0
    for t in (int, float):
        try:
            parsed = t(value)
        except ValueError:
            continue
        else:
            is_successful = True
            break

    return parsed, is_successful


def _parse_boolean(value: str) -> tuple[bool, bool]:
    lowered = value.lower()
    return (
        lowered == "true",
        lowered in ("true", "false"),
    )


def _parse_value(value: str) -> str | float | int | bool:
    # remove potential comment from end of line
    value = value.split("#")[0].strip()
    parsed, is_numeric = _parse_numeric(value)
    if not is_numeric:
        parsed, is_boolean = _parse_boolean(value)
        if not is_boolean:
            parsed = value

    return parsed


def _parse_dict(
    line: str, lines: typing.Iterable[str]
) -> tuple[dict[str, str | float | int | bool], typing.Iterable[str]]:
    if "=" not in line and "{" not in line:
        raise ValueError(INVALID_DICT_LINE.format(line))

    var_name = line.split("=")[0].strip()
    parsed = {}
    for line in lines:
        if "{" not in line and "}" not in line:
            key, value = (v.strip() for v in line.split("="))
            parsed[key] = _parse_value(value)
        elif "{" in line:
            (key, value), lines = _parse_dict(line, lines)
            parsed[key] = value
        elif "}":
            break

    return (var_name, parsed), lines


def read(input_file: Path) -> dict[str, str | float | int | bool]:
    """Read an oxDNA input file."""
    with input_file.open("r") as f:
        lines = filter(lambda line: (len(line.strip()) > 0) and (not line.strip().startswith("#")), f.readlines())

    parsed = {}
    for line in lines:
        if "{" in line:
            (key, value), lines = _parse_dict(line, lines)
        else:
            key, str_value = (v.strip() for v in line.split("="))
            value = _parse_value(str_value)

        parsed[key] = value

    return parsed


def write_to(input_config: dict, f: io.TextIOWrapper) -> None:
    """Write an oxDNA input file."""
    for key, value in input_config.items():
        if isinstance(value, dict):
            f.write(f"{key} = {{\n")
            write_to(value, f)
            f.write("}\n")
        else:
            if key == "T" and isinstance(value, float):
                parsed_value = str(value) + "K"
            elif isinstance(value, bool):
                parsed_value = str(value).lower()
            else:
                parsed_value = str(value)

            f.write(f"{key} = {parsed_value}\n")


def write(input_config: dict, input_file: Path) -> None:
    """Write an oxDNA input file."""
    with input_file.open("w") as f:
        write_to(input_config, f)
