import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules import config_manager as cm


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


def test_ensure_seed_courses_registers_entries(tmp_path, temp_config_home):
    config = cm.load_config()

    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()

    seed_course = seeds_dir / "sample.md"
    seed_course.write_text(
        """# Sample Course\n## Basics\n### Intro\n#### Lesson One\n    print('hi')\n""",
        encoding="utf-8",
    )

    updated = cm.ensure_seed_courses(config, seeds_dir)

    courses_dir = cm.get_courses_dir()
    stored_course = courses_dir / "sample.md"
    assert stored_course.exists()

    ids = {entry.get("id") for entry in updated.get("courses", [])}
    assert "sample" in ids
