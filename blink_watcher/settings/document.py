"""The configuration page's model: the options documented in .env.example
(groups, help text, tags — see the file's header) with the current values from
.env, and a validated, in-place .env update.

  options()           groups and options for the page; secret values are never returned
  update(changes)     validate {KEY: value | None} against Settings, then write .env:
                      a key's line is replaced in place (comments and order kept);
                      a new key is appended with its help comment; None removes the
                      key (the default applies). The previous file goes to .env.bak.

Types and defaults come from the composed Settings (every section's fields);
an option documented in .env.example but not in Settings is a plain string.
A "# @type" tag gives the page a richer control than the field's Python type:

  # @type cameras        a list of camera names, picked from the camera list
                         (webui.cameras); "# @type cameras all": the value may
                         also be "all" (every camera)
  # @type camera-map     {"Camera A": ["Camera B", ...]}: pairs of cameras,
                         either direction (e.g. neighbours)
  # @type camera-filter  {"Camera A": "always", ...}: a detection filter per
                         camera (core.filters), picked from a dropdown;
                         cameras not listed use DEFAULT_FILTER
  # @label Text          a short name for the option on the page (e.g. a column header)
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import dotenv_values
from pydantic import ValidationError
from pydantic_settings import SettingsError

from blink_watcher.core import filters
from blink_watcher.core.paths import ENV_EXAMPLE as EXAMPLE, ENV_FILE as ENV
from blink_watcher.settings import Settings

SECRET_NAME = re.compile(r"PASSWORD|TOKEN|API_KEY|SECRET", re.I)
GROUP = re.compile(r"^#\s*---\s*(.+?)\s*-*\s*$")
KEYLINE = re.compile(r"^(#\s*)?([A-Z][A-Z0-9_]*)=(.*)$")
TAG = re.compile(r"^#\s*@(\w+)\s*(.*)$")
TYPES = {"cameras", "camera-map", "camera-filter"}  # "# @type" values


@dataclass
class Option:
    key: str
    group: str
    help: list[str] = field(default_factory=list)
    example: str = ""
    advanced: bool = False
    secret: bool = False
    choices: Optional[list[str]] = None
    default: Optional[str] = None     # "# @default x": for options the code reads from the environment
    off_by_default: bool = False      # documented as "# KEY=example"
    type: Optional[str] = None        # "# @type": one of TYPES, else the field's own type
    allow_all: bool = False           # "# @type cameras all"
    label: Optional[str] = None       # "# @label"


@dataclass
class Group:
    name: str
    intro: list[str] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    options: list[Option] = field(default_factory=list)


def parse_example(path: Path = EXAMPLE) -> list[Group]:
    groups: list[Group] = []
    seen: dict[str, Option] = {}
    pending: list[str] = []       # comment lines waiting for their option
    tags: dict[str, str] = {}
    in_intro = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        m = GROUP.match(line)
        if m:
            groups.append(Group(m.group(1)))
            pending, tags, in_intro = [], {}, True
            continue
        if not groups:
            continue                   # the file's own header
        g = groups[-1]
        if not line.strip():
            if in_intro and pending:
                g.intro += pending
            pending, tags, in_intro = [], {}, False
            continue
        t = TAG.match(line)
        if t:
            name, val = t.group(1).lower(), t.group(2).strip()
            if name == "programs":
                g.programs = [p.strip() for p in val.split(",") if p.strip()]
                g.intro += pending
                pending, in_intro = [], False
            else:
                tags[name] = val
            continue
        k = KEYLINE.match(line)
        if k:
            commented, key, value = bool(k.group(1)), k.group(2), k.group(3)
            if commented and key in seen:   # an alternative example of a known option
                seen[key].help.append(f"Example: {key}={value}")
                continue
            if commented and not tags and key not in _settings_keys() and key not in seen:
                pending.append(line.lstrip("# ").strip())
                continue
            kind, *flags = tags.get("type", "").split() or [""]
            if kind and kind not in TYPES:
                raise ValueError(f"{path.name}: {key}: unknown @type {kind!r} (known: {', '.join(sorted(TYPES))})")
            opt = Option(key, g.name, help=[x for x in pending], example=value,
                         advanced="advanced" in tags, secret="secret" in tags or bool(SECRET_NAME.search(key)),
                         choices=[c.strip() for c in tags["choices"].split("|")] if "choices" in tags else None,
                         default=tags.get("default"),
                         off_by_default=commented, type=kind or None, allow_all="all" in flags,
                         label=tags.get("label"))
            g.options.append(opt)
            seen[key] = opt
            pending, tags, in_intro = [], {}, False
            continue
        if line.startswith("#"):
            pending.append(line[1:].strip() if line[1:2] != " " else line[2:])
    return [g for g in groups if g.options]


def _settings_keys() -> set[str]:
    return {name.upper() for name in Settings.model_fields}


def _kind(key: str) -> tuple[str, Optional[list[str]], Any]:
    """(type, choices, default) for the page: bool | int | float | list | dict | text."""
    f = Settings.model_fields.get(key.lower())
    if f is None:
        return "text", None, None
    ann = f.annotation
    args = [a for a in typing.get_args(ann) if a is not type(None)] if typing.get_origin(ann) is typing.Union else [ann]
    base = args[0] if args else ann
    origin = typing.get_origin(base)
    if origin is typing.Annotated:
        base = typing.get_args(base)[0]
        origin = typing.get_origin(base)
    default = None if f.is_required() else f.get_default(call_default_factory=True)
    if isinstance(default, Path):
        default = default.as_posix()
    if origin is typing.Literal:
        return "choice", [str(c) for c in typing.get_args(base)], default
    if base is bool:
        return "bool", None, default
    if base is int:
        return "int", None, default
    if base is float:
        return "float", None, default
    if origin is list:
        return "list", None, default
    if origin is dict:
        return "dict", None, default
    return "text", None, default


def options() -> dict:
    env = dotenv_values(ENV) if ENV.exists() else {}
    groups = []
    for g in parse_example():
        opts = []
        for o in g.options:
            kind, choices, default = _kind(o.key)
            present = o.key in env
            value = env.get(o.key)
            kind = o.type or ("choice" if (o.choices or choices) else kind)
            item = {"key": o.key, "help": "\n".join(o.help).strip(), "advanced": o.advanced, "secret": o.secret,
                    "type": kind, "choices": o.choices or choices, "label": o.label, "allow_all": o.allow_all,
                    "default": default if o.default is None else o.default, "in_env": present, "off_by_default": o.off_by_default,
                    "required": Settings.model_fields.get(o.key.lower()) is not None
                    and Settings.model_fields[o.key.lower()].is_required()}
            if kind == "camera-filter":
                item.update(choices=list(filters.FILTER_NAMES), choice_labels=filters.LABELS,
                            item_default=filters.DEFAULT_FILTER)
            if o.secret:
                item["set"] = bool(value)
            else:
                item["value"] = value if present else None
                item["example"] = o.example if o.off_by_default else None
            opts.append(item)
        groups.append({"name": g.name, "intro": " ".join(g.intro).strip(), "programs": g.programs, "options": opts})
    return {"groups": groups, "env_file": str(ENV), "exists": ENV.exists()}


# ---------- writing ----------

def _quote(value: str) -> str:
    """A .env value as written: quoted only when dotenv would misread it unquoted
    (leading / trailing spaces, a leading quote, " #" — a comment —, a line break)."""
    if not value or not (value != value.strip() or value[0] in "'\"" or " #" in value or "\n" in value):
        return value
    if "'" not in value and "\n" not in value:
        return f"'{value}'"
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _render(lines: list[str], changes: dict[str, Optional[str]]) -> list[str]:
    known = {o.key: (g, o) for g in parse_example() for o in g.options}
    out, done = [], set()
    for line in lines:
        k = KEYLINE.match(line)
        if k and not k.group(1) and k.group(2) in changes:
            key = k.group(2)
            if key in done:
                continue                        # a duplicate line: the first one now holds the value
            done.add(key)
            if changes[key] is not None:
                out.append(f"{key}={_quote(changes[key])}")
            continue                            # None: removed, the default applies
        out.append(line)
    added = [(k, v) for k, v in changes.items() if k not in done and v is not None]
    if added:
        if out and out[-1].strip():
            out.append("")
        for key, value in added:
            g, o = known.get(key, (None, None))
            out.append(f"# {g.name if g else 'Added'} — {key}")
            out += [f"# {h}" for h in (o.help if o else [])]
            out.append(f"{key}={_quote(value)}")
            out.append("")
    return out


def validate(env_lines: list[str]) -> dict[str, str]:
    """Errors ({KEY: message}) loading Settings from these .env lines; {} = valid."""
    fd, tmp = tempfile.mkstemp(suffix=".env")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(env_lines) + "\n")
        saved = {k: os.environ.pop(k) for k in list(os.environ) if k.lower() in Settings.model_fields}
        try:
            settings = Settings(_env_file=tmp)
        finally:
            os.environ.update(saved)
        if not settings.cameras:  # as load_settings() refuses it
            return {"CAMERAS": 'required: camera names, or "all" for every camera'}
        return {}
    except ValidationError as e:
        return {str(err["loc"][0]).upper() if err.get("loc") else "_": err["msg"] for err in e.errors()}
    except SettingsError as e:  # a list / dict value that is not valid JSON
        field_name = re.search(r'field "(\w+)"', str(e))
        return {field_name.group(1).upper() if field_name else "_": f"not valid JSON ({e.__cause__ or e})"}
    finally:
        os.unlink(tmp)


def update(changes: dict[str, Optional[str]]) -> dict:
    """Validate, then write .env. Unknown keys are refused."""
    known = {o.key: o for g in parse_example() for o in g.options}
    bad = [k for k in changes if k not in known]
    if bad:
        return {"ok": False, "errors": {k: "not a documented option" for k in bad}}
    for k, v in changes.items():
        if v is not None and not isinstance(v, str):
            changes[k] = str(v)
    lines = ENV.read_text(encoding="utf-8").splitlines() if ENV.exists() else []
    new = _render(lines, changes)
    errors = validate(new)
    if errors:
        return {"ok": False, "errors": errors}
    if ENV.exists():
        shutil.copy2(ENV, ENV.with_name(".env.bak"))
    tmp = ENV.with_name(".env.tmp")
    tmp.write_text("\n".join(new).rstrip("\n") + "\n", encoding="utf-8")
    os.replace(tmp, ENV)
    programs = sorted({p for g in parse_example() for o in g.options if o.key in changes for p in g.programs})
    return {"ok": True, "changed": sorted(changes), "restart": programs}
