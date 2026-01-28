import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


CONFIG_DIR_NAME = "rt"
CONFIG_FILE_NAME = "config.json"
COURSES_DIR_NAME = "courses"

COURSE_FILENAME_PREFIX = "course_"
COURSE_FILENAME_SUFFIX = ".md"


def _xdg_config_home() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base).expanduser()
    return Path.home() / ".config"


def get_config_root() -> Path:
    return _xdg_config_home() / CONFIG_DIR_NAME


def get_courses_dir() -> Path:
    return get_config_root() / COURSES_DIR_NAME


def get_config_file() -> Path:
    return get_config_root() / CONFIG_FILE_NAME


def _default_config() -> Dict[str, Any]:
    return {
        "courses": [],
        "xai": {
            "api_key": None,
            "management_key": None,
            "collection_id": None,
        },
    }


def ensure_config_dirs() -> None:
    root = get_config_root()
    courses_dir = get_courses_dir()
    root.mkdir(parents=True, exist_ok=True)
    courses_dir.mkdir(parents=True, exist_ok=True)


def sanitize_config(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return _default_config()

    sanitized = dict(data)

    sanitized.setdefault("courses", [])
    sanitized.setdefault("xai", {})

    if not isinstance(sanitized["courses"], list):
        sanitized["courses"] = []

    xai_section = sanitized.get("xai", {})
    if not isinstance(xai_section, dict):
        xai_section = {}
    xai_section.pop("api_key_env", None)
    xai_section.pop("management_key_env", None)
    xai_section.pop("collections", None)
    xai_section.setdefault("api_key", None)
    xai_section.setdefault("management_key", None)
    xai_section.setdefault("collection_id", None)
    sanitized["xai"] = xai_section

    sanitized["courses"] = [
        _sanitize_course_entry(entry) for entry in sanitized["courses"]
    ]

    return sanitized


def _sanitize_course_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        return _empty_course_entry()

    sanitized = dict(entry)
    sanitized.pop("display_name", None)
    sanitized.pop("created_at", None)
    sanitized.pop("version", None)
    sanitized.pop("id", None)
    sanitized.setdefault("name", None)
    sanitized.setdefault("local_path", None)
    sanitized.setdefault("xai_file_id", None)

    return sanitized


def _empty_course_entry() -> Dict[str, Any]:
    return {
        "name": None,
        "local_path": None,
        "xai_file_id": None,
    }


def load_config() -> Dict[str, Any]:
    ensure_config_dirs()
    config_path = get_config_file()
    if not config_path.exists():
        config = _default_config()
        save_config(config)
        return config

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (json.JSONDecodeError, OSError):
        backup_path = config_path.with_suffix(config_path.suffix + ".bak")
        try:
            config_path.rename(backup_path)
        except OSError:
            pass
        config = _default_config()
        save_config(config)
        return config

    sanitized = sanitize_config(raw)
    normalized = normalize_course_entries(sanitized)
    save_config(normalized)
    return normalized


def save_config(config: Dict[str, Any]) -> None:
    ensure_config_dirs()
    config_path = get_config_file()
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)


def upsert_course_entry(
    config: Dict[str, Any], entry: Dict[str, Any]
) -> Dict[str, Any]:
    updated = sanitize_config(config)
    courses: List[Dict[str, Any]] = updated.setdefault("courses", [])

    sanitized_entry = _sanitize_course_entry(entry)
    slug = course_slug(
        sanitized_entry.get("name"), sanitized_entry.get("local_path")
    )

    for idx, course in enumerate(courses):
        if course_slug(course.get("name"), course.get("local_path")) == slug:
            merged_entry = {**course, **sanitized_entry}
            if sanitized_entry.get("xai_file_id") is None:
                merged_entry["xai_file_id"] = course.get("xai_file_id")
            courses[idx] = _sanitize_course_entry(merged_entry)
            updated["courses"] = courses
            return updated

    courses.append(sanitized_entry)
    updated["courses"] = courses
    return updated


def remove_course_entry(config: Dict[str, Any], slug: str) -> Dict[str, Any]:
    courses: List[Dict[str, Any]] = config.get("courses", [])
    updated_courses = [
        c
        for c in courses
        if course_slug(c.get("name"), c.get("local_path")) != slug
    ]
    config["courses"] = updated_courses
    return config


def normalize_course_entries(config: Dict[str, Any]) -> Dict[str, Any]:
    courses_dir = get_courses_dir()
    moves = _flatten_courses_directory()

    dedup: Dict[str, Dict[str, Any]] = {}

    for raw_entry in config.get("courses", []):
        entry = _sanitize_course_entry(raw_entry)

        local_path = entry.get("local_path")
        if local_path and local_path in moves:
            local_path = moves[local_path]
            entry["local_path"] = local_path

        path_obj = Path(local_path).expanduser() if local_path else None
        if path_obj and not path_obj.exists():
            path_obj = None

        slug = course_slug(entry.get("name"), local_path)

        target_path = courses_dir / _course_filename(slug)

        if path_obj and path_obj.exists():
            try:
                if not path_obj.samefile(target_path):
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    if not target_path.exists():
                        path_obj.rename(target_path)
                    path_obj = target_path
                else:
                    path_obj = target_path
            except OSError:
                path_obj = target_path
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            path_obj = target_path

        entry["local_path"] = str(path_obj)
        entry.setdefault("name", slug.replace("-", " ").title())

        sanitized_entry = _sanitize_course_entry(entry)

        existing = dedup.get(slug)
        if existing:
            merged_entry = {**existing, **sanitized_entry}
            if sanitized_entry.get("xai_file_id") is None:
                merged_entry["xai_file_id"] = existing.get("xai_file_id")
            dedup[slug] = _sanitize_course_entry(merged_entry)
        else:
            dedup[slug] = sanitized_entry

    ordered_slugs = sorted(dedup.keys())
    config["courses"] = [dedup[key] for key in ordered_slugs]

    return config


def _strip_course_prefix(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.startswith(COURSE_FILENAME_PREFIX):
        return text[len(COURSE_FILENAME_PREFIX) :]
    return text


def _normalize_identifier(name: Optional[str]) -> str:
    if not name:
        return ""
    return str(name).strip().lower().replace(" ", "-")


def _course_filename(identifier: str) -> str:
    slug = _normalize_identifier(_strip_course_prefix(identifier)) or "untitled"
    return f"{COURSE_FILENAME_PREFIX}{slug}{COURSE_FILENAME_SUFFIX}"


def _identifier_from_path(path: Path) -> str:
    if not path:
        return "untitled"
    return _normalize_identifier(_strip_course_prefix(path.stem)) or "untitled"


def course_slug(name: Optional[str], local_path: Optional[str]) -> str:
    if local_path:
        return _identifier_from_path(Path(local_path))
    if name:
        slug = _normalize_identifier(name)
        if slug:
            return slug
    return "untitled"


def _flatten_courses_directory() -> Dict[str, str]:
    courses_dir = get_courses_dir()
    moves: Dict[str, str] = {}

    if not courses_dir.exists():
        return moves

    for md_path in sorted(courses_dir.rglob("*.md")):
        if not md_path.exists():
            continue

        original_path = str(md_path)
        slug = _identifier_from_path(md_path)
        target = courses_dir / _course_filename(slug)

        if md_path == target:
            moves[original_path] = original_path
            continue

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                try:
                    if target.samefile(md_path):
                        moves[original_path] = str(target)
                        continue
                except OSError:
                    pass
                moves[original_path] = original_path
                continue
            md_path.rename(target)
            moves[original_path] = str(target)
        except OSError:
            moves[original_path] = original_path

    return moves
