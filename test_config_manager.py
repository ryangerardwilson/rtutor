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


def test_ensure_seed_courses_uses_prefixed_names(temp_config_home, tmp_path):
    config = cm.load_config()

    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()

    seed_md = seeds_dir / "course_sample.md"
    seed_md.write_text(
        """# Sample Course\n## Basics\n### Section\n#### Lesson One\n    print('hi')\n""",
        encoding="utf-8",
    )

    updated = cm.ensure_seed_courses(config, seeds_dir)

    courses_dir = cm.get_courses_dir()
    expected_file = courses_dir / "course_sample.md"
    assert expected_file.exists()

    names = {Path(entry["local_path"]).name for entry in updated["courses"]}
    assert "course_sample.md" in names
    assert len(updated["courses"]) == len(names)  # no duplicates


def test_normalize_flattens_nested_courses(temp_config_home):
    config = cm.load_config()

    courses_dir = cm.get_courses_dir()
    nested_dir = courses_dir / "nested" / "inner"
    nested_dir.mkdir(parents=True, exist_ok=True)

    nested_file = nested_dir / "custom.md"
    nested_file.write_text("""# Custom\n## Part\n### Section\n#### Lesson\n    pass\n""", encoding="utf-8")

    config.setdefault("courses", []).append(
        {
            "id": "Custom",
            "display_name": "Custom",
            "local_path": str(nested_file),
            "source": "user",
        }
    )

    normalized = cm.normalize_course_entries(config)

    entries = {entry["id"]: entry for entry in normalized["courses"]}
    assert "custom" in entries

    final_path = Path(entries["custom"]["local_path"])
    assert final_path.parent == courses_dir
    assert final_path.name.startswith("course_")
    assert final_path.suffix == ".md"
    assert final_path.exists()
    assert not nested_file.exists()
