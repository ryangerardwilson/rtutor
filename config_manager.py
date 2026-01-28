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
    sanitized.setdefault(
        "created_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
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

    sanitized["courses"] = [
        _sanitize_course_entry(entry) for entry in sanitized["courses"]
    ]

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
    seed_files = list(seeds_dir.glob("course_*.md"))
    if not seed_files:
        return normalize_course_entries(updated_config)

    changed = False
    existing_ids = {
        course.get("id")
        for course in updated_config.get("courses", [])
        if course.get("id")
    }

    for seed_file in seed_files:
        identifier = (
            _normalize_identifier(_strip_course_prefix(seed_file.stem)) or "untitled"
        )
        dest_path = courses_dir / _course_filename(identifier)
        legacy_path = courses_dir / seed_file.name

        file_changed = False
        try:
            if legacy_path.exists() and legacy_path != dest_path:
                if not dest_path.exists():
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    legacy_path.rename(dest_path)
                    file_changed = True
            if not dest_path.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(seed_file, dest_path)
                file_changed = True
        except OSError:
            continue

        display_name = _extract_course_title(dest_path) or seed_file.stem
        entry = {
            "id": identifier,
            "name": display_name,
            "display_name": display_name,
            "local_path": str(dest_path),
            "source": "seed",
        }

        was_present = identifier in existing_ids
        updated_config = upsert_course_entry(updated_config, entry)
        existing_ids.add(identifier)
        if file_changed or not was_present:
            changed = True

    normalized = normalize_course_entries(updated_config)
    if changed or normalized != updated_config:
        save_config(normalized)

    return normalized


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


def normalize_course_entries(config: Dict[str, Any]) -> Dict[str, Any]:
    courses_dir = get_courses_dir()
    moves = _flatten_courses_directory()

    dedup: Dict[str, Dict[str, Any]] = {}

    for raw_entry in config.get("courses", []):
        entry = _sanitize_course_entry(raw_entry)

        local_path = entry.get("local_path")
        if local_path and local_path in moves:
            local_path = moves[local_path]

        path_obj = Path(local_path).expanduser() if local_path else None
        if path_obj and not path_obj.exists():
            path_obj = None

        slug = ""
        slug_candidates: List[Optional[str]] = []
        if path_obj:
            slug_candidates.append(path_obj.stem)
        slug_candidates.extend(
            [entry.get("id"), entry.get("display_name"), entry.get("name")]
        )

        for candidate in slug_candidates:
            slug_candidate = _normalize_identifier(_strip_course_prefix(candidate))
            if slug_candidate:
                slug = slug_candidate
                break

        if not slug:
            slug = "untitled"

        target_path = courses_dir / _course_filename(slug)

        if path_obj:
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

        normalized_entry = _sanitize_course_entry(
            {
                "id": slug,
                "name": entry.get("name") or entry.get("display_name") or slug.title(),
                "display_name": entry.get("display_name") or entry.get("name") or slug,
                "local_path": str(path_obj),
                "source": entry.get("source") or "user",
                "xai": entry.get("xai") or {},
            }
        )

        existing = dedup.get(slug)
        if existing:
            if (
                existing.get("source") != "seed"
                and normalized_entry.get("source") == "seed"
            ):
                dedup[slug] = normalized_entry
            else:
                if (
                    normalized_entry.get("source") == "seed"
                    and existing.get("source") != "seed"
                ):
                    existing["source"] = "seed"
                if normalized_entry.get("local_path"):
                    existing["local_path"] = normalized_entry["local_path"]
                if not existing.get("name") and normalized_entry.get("name"):
                    existing["name"] = normalized_entry["name"]
                if not existing.get("display_name") and normalized_entry.get(
                    "display_name"
                ):
                    existing["display_name"] = normalized_entry["display_name"]
            continue

        dedup[slug] = normalized_entry

    config["courses"] = sorted(
        dedup.values(),
        key=lambda item: (
            item.get("display_name") or item.get("name") or item.get("id", "")
        ).lower(),
    )
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
