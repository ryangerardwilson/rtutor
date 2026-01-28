import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config_manager as cm


@pytest.fixture
def temp_config_home(tmp_path, monkeypatch):
    cfg_home = tmp_path / "cfg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_home))

    root = cm.get_config_root()
    if root.exists():
        shutil.rmtree(root)

    return cfg_home


def test_load_config_creates_file(temp_config_home):
    config = cm.load_config()

    config_path = cm.get_config_file()
    assert config_path.exists()

    data = json.loads(config_path.read_text())
    assert "courses" in data
    assert isinstance(config["courses"], list)
    assert data["xai"].keys() == {"api_key", "management_key", "collection_id"}


def test_upsert_course_entry_adds_course(temp_config_home):
    config = cm.load_config()

    updated = cm.upsert_course_entry(
        config,
        {"name": "Demo", "local_path": "/tmp/demo.md"},
    )

    assert len(updated["courses"]) == 1
    course = updated["courses"][0]
    assert course == {
        "name": "Demo",
        "local_path": "/tmp/demo.md",
        "xai_file_id": None,
    }


def test_upsert_course_entry_preserves_existing_file_id(temp_config_home):
    config = {
        "courses": [
            {
                "name": "Demo",
                "local_path": "/tmp/demo.md",
                "xai_file_id": "file-123",
            }
        ],
        "xai": {
            "api_key": None,
            "management_key": None,
            "collection_id": None,
        },
    }

    updated = cm.upsert_course_entry(
        config,
        {"name": "Demo", "local_path": "/tmp/demo.md"},
    )

    course = updated["courses"][0]
    assert course["xai_file_id"] == "file-123"


def test_normalize_flattens_nested_courses(temp_config_home):
    config = cm.load_config()

    nested_path = "/tmp/nested/custom.md"

    config.setdefault("courses", []).append(
        {
            "name": "Custom",
            "local_path": nested_path,
        }
    )

    normalized = cm.normalize_course_entries(config)

    entries = {
        cm.course_slug(item.get("name"), item.get("local_path")): item
        for item in normalized["courses"]
    }
    assert "custom" in entries

    assert entries["custom"]["local_path"] == nested_path
    assert entries["custom"]["xai_file_id"] is None


def test_normalize_strips_display_name(temp_config_home):
    config = cm.load_config()
    courses_dir = cm.get_courses_dir()
    sample = courses_dir / "course_demo.md"
    sample.write_text("# Demo", encoding="utf-8")

    config["courses"].append(
        {
            "name": "Demo",
            "display_name": "Demo Course",
            "local_path": str(sample),
        }
    )

    normalized = cm.normalize_course_entries(config)
    entries = {
        cm.course_slug(item.get("name"), item.get("local_path")): item
        for item in normalized["courses"]
    }
    entry = entries["demo"]
    assert "display_name" not in entry
    assert "id" not in entry
