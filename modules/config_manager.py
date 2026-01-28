import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


CONFIG_DIR_NAME = "rt"
CONFIG_FILE_NAME = "config.json"
COURSES_DIR_NAME = "courses"
DEFAULT_VERSION = 1


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
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "version": DEFAULT_VERSION,
        "created_at": timestamp,
        "courses": [],
        "xai": {
            "api_key_env": "XAI_API_KEY",
            "management_key_env": "XAI_MANAGEMENT_API_KEY",
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

    sanitized.setdefault("version", DEFAULT_VERSION)
    sanitized.setdefault("created_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    sanitized.setdefault("courses", [])
    sanitized.setdefault(
        "xai",
        {
            "api_key_env": "XAI_API_KEY",
            "management_key_env": "XAI_MANAGEMENT_API_KEY",
        },
    )

    if not isinstance(sanitized["courses"], list):
        sanitized["courses"] = []

    xai_section = sanitized.get("xai", {})
    if not isinstance(xai_section, dict):
        xai_section = {}
    xai_section.setdefault("api_key_env", "XAI_API_KEY")
    xai_section.setdefault("management_key_env", "XAI_MANAGEMENT_API_KEY")
    sanitized["xai"] = xai_section

    sanitized["courses"] = [_sanitize_course_entry(entry) for entry in sanitized["courses"]]

    return sanitized


def _sanitize_course_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        return _empty_course_entry()

    sanitized = dict(entry)
    sanitized.setdefault("id", None)
    sanitized.setdefault("name", None)
    sanitized.setdefault("display_name", None)
    sanitized.setdefault("local_path", None)
    sanitized.setdefault("source", "user")
    sanitized.setdefault("xai", {})

    xai_data = sanitized.get("xai", {})
    if not isinstance(xai_data, dict):
        xai_data = {}
    xai_data.setdefault("collection_id", None)
    xai_data.setdefault("file_ids", [])
    xai_data.setdefault("last_uploaded_at", None)
    if not isinstance(xai_data.get("file_ids"), list):
        xai_data["file_ids"] = []
    sanitized["xai"] = xai_data

    return sanitized


def _empty_course_entry() -> Dict[str, Any]:
    return {
        "id": None,
        "name": None,
        "display_name": None,
        "local_path": None,
        "source": "user",
        "xai": {
            "collection_id": None,
            "file_ids": [],
            "last_uploaded_at": None,
        },
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
    save_config(sanitized)
    return sanitized


def save_config(config: Dict[str, Any]) -> None:
    ensure_config_dirs()
    config_path = get_config_file()
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)


def upsert_course_entry(config: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    courses: List[Dict[str, Any]] = config.setdefault("courses", [])
    target_id = entry.get("id")

    updated = sanitize_config(config)
    courses = updated["courses"]

    if target_id:
        for idx, course in enumerate(courses):
            if course.get("id") == target_id:
                courses[idx] = _sanitize_course_entry({**course, **entry})
                updated["courses"] = courses
                return updated

    courses.append(_sanitize_course_entry(entry))
    updated["courses"] = courses
    return updated


def remove_course_entry(config: Dict[str, Any], course_id: str) -> Dict[str, Any]:
    courses: List[Dict[str, Any]] = config.get("courses", [])
    updated_courses = [c for c in courses if c.get("id") != course_id]
    config["courses"] = updated_courses
    return config


def ensure_seed_courses(config: Dict[str, Any], seeds_dir: Path) -> Dict[str, Any]:
    ensure_config_dirs()
    if not seeds_dir.exists():
        return config

    updated_config = sanitize_config(config)
    courses_dir = get_courses_dir()
    existing_ids = {course.get("id") for course in updated_config.get("courses", []) if course.get("id")}

    seed_files = list(seeds_dir.glob("*.md"))
    if not seed_files:
        return updated_config

    changed = False
    for seed_file in seed_files:
        course_id = seed_file.stem.lower().replace(" ", "-")
        dest_path = courses_dir / seed_file.name

        try:
            if not dest_path.exists():
                shutil.copy2(seed_file, dest_path)
                changed = True
        except OSError:
            continue

        display_name = _extract_course_title(dest_path) or seed_file.stem
        entry = {
            "id": course_id,
            "name": display_name,
            "display_name": display_name,
            "local_path": str(dest_path),
            "source": "seed",
        }

        if course_id not in existing_ids:
            updated_config = upsert_course_entry(updated_config, entry)
            existing_ids.add(course_id)
            changed = True
        else:
            # Ensure metadata is current (e.g., local path may have changed)
            updated_config = upsert_course_entry(updated_config, entry)
            changed = True

    if changed:
        save_config(updated_config)

    return updated_config


def _extract_course_title(path: Path) -> Optional[str]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith("# "):
                    return stripped[2:].strip()
    except OSError:
        return None
    return None
