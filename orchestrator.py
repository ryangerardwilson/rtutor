import curses
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config_manager import (
    ensure_seed_courses,
    ensure_config_dirs,
    get_courses_dir,
    load_config,
    normalize_course_entries,
    save_config,
)
from course_parser import CourseParser
from flag_handler import handle_bookmark_flags
from menu import Menu


class Orchestrator:
    def __init__(self, argv: Optional[List[str]] = None):
        self.argv = argv if argv is not None else list(sys.argv[1:])
        self.config: Dict[str, Any] = {}
        self.courses = []
        self.missing_courses: List[str] = []

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def run(self) -> None:
        os.environ.setdefault("ESCDELAY", "25")
        os.environ.setdefault("TERM", "xterm-256color")

        self.config = self._load_and_prepare_config()
        self.courses, self.missing_courses = self._load_courses()

        self._handle_missing_courses()
        self._handle_flags()

        if not self.courses:
            target_dir = get_courses_dir()
            print(f"No valid courses found. Place Markdown files in {target_dir}.")
            sys.exit(1)

        doc_mode = True  # Reserved for future toggles (e.g., query mode)

        menu = Menu(self.courses, doc_mode=doc_mode)
        try:
            curses.wrapper(menu.run)
        except KeyboardInterrupt:
            sys.exit(0)

    # ------------------------------------------------------------------
    # Config handling
    # ------------------------------------------------------------------
    def _load_and_prepare_config(self) -> dict:
        ensure_config_dirs()
        config = load_config()

        seeds_dir = Path(__file__).resolve().parent
        config = ensure_seed_courses(config, seeds_dir)
        config = normalize_course_entries(config)
        save_config(config)

        return config

    # ------------------------------------------------------------------
    # Course loading
    # ------------------------------------------------------------------
    def _load_courses(self) -> Tuple[List, List[str]]:
        parser = CourseParser()
        course_files, missing = self._resolve_course_files()
        courses = parser.parse_courses(course_files)

        if courses:
            self._update_config_with_courses(courses)

        return courses, missing

    def _resolve_course_files(self) -> Tuple[List[str], List[str]]:
        course_files: List[str] = []
        missing_courses: List[str] = []
        for course in self.config.get("courses", []):
            local_path = course.get("local_path")
            if not local_path:
                continue
            path_obj = Path(local_path).expanduser()
            if path_obj.is_file():
                course_files.append(str(path_obj))
            else:
                missing_courses.append(course.get("display_name") or course.get("id"))

        return course_files, missing_courses

    def _course_id_from_path(self, path: str) -> str:
        if not path:
            return ""
        return Path(path).stem.lower().replace(" ", "-")

    def _update_config_with_courses(self, courses: Iterable) -> None:
        entries = []
        for course in courses:
            entries.append(
                {
                    "id": self._course_id_from_path(course.source_file),
                    "name": course.name,
                    "display_name": course.name,
                    "local_path": course.source_file,
                }
            )

        merged = dict(self.config)
        merged.setdefault("courses", [])
        merged["courses"].extend(entries)
        merged = normalize_course_entries(merged)
        save_config(merged)
        self.config = merged

    # ------------------------------------------------------------------
    # Flags & warnings
    # ------------------------------------------------------------------
    def _handle_missing_courses(self) -> None:
        if not self.missing_courses:
            return
        missing_list = ", ".join(name for name in self.missing_courses if name)
        print(f"Warning: Missing course files for: {missing_list}")

    def _handle_flags(self) -> None:
        # flag_handler operates on global sys.argv; ensure it's in sync
        sys.argv = [sys.argv[0]] + self.argv
        handle_bookmark_flags(self.courses)
