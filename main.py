# ~/Apps/rtutor/main.py
#!/usr/bin/env python3
import curses
import sys
import os
from pathlib import Path
from modules.menu import Menu
from modules.course_parser import CourseParser
from modules.flag_handler import handle_bookmark_flags
from modules.config_manager import (
    ensure_seed_courses,
    get_courses_dir,
    load_config,
    save_config,
    upsert_course_entry,
)


def _course_id_from_path(path: str) -> str:
    if not path:
        return ""
    return Path(path).stem.lower().replace(" ", "-")

# Set TERM explicitly for consistent color support
os.environ['TERM'] = 'xterm-256color'

def main():
    # Set ESCDELAY early, before any curses initialization
    # 25ms gives very snappy Esc response while still allowing most Alt+key and escape sequences to work reliably
    os.environ.setdefault('ESCDELAY', '25')

    # Determine repo seed directory and ensure seed courses are registered
    script_path = os.path.realpath(__file__)
    script_dir = Path(script_path).parent
    seeds_dir = script_dir / "courses"

    config = load_config()
    config = ensure_seed_courses(config, seeds_dir)

    course_files = []
    missing_courses = []
    for course in config.get("courses", []):
        local_path = course.get("local_path")
        if not local_path:
            continue
        path_obj = Path(local_path).expanduser()
        if path_obj.is_file():
            course_files.append(str(path_obj))
        else:
            missing_courses.append(course.get("display_name") or course.get("id"))

    parser = CourseParser()
    courses = parser.parse_courses(course_files)

    # Update config entries with canonical course names if available
    if courses:
        updated_config = config
        for course in courses:
            entry = {
                "id": _course_id_from_path(course.source_file),
                "name": course.name,
                "display_name": course.name,
                "local_path": course.source_file,
            }
            updated_config = upsert_course_entry(updated_config, entry)
        save_config(updated_config)

    if not courses:
        target_dir = get_courses_dir()
        print(f"No valid courses found. Place Markdown files in {target_dir}.")
        sys.exit(1)

    if missing_courses:
        missing_list = ", ".join(name for name in missing_courses if name)
        print(f"Warning: Missing course files for: {missing_list}")

    handle_bookmark_flags(courses)

    # Otherwise, proceed with menus.
    # Doc mode is now the default. -d/--doc flags are still accepted but redundant.
    doc_mode = True
    if ("-d" in sys.argv) or ("--doc" in sys.argv):
        doc_mode = True  # Explicitly requested (no change needed)

    menu = Menu(courses, doc_mode=doc_mode)
    try:
        curses.wrapper(menu.run)
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    main()
