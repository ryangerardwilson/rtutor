#!/usr/bin/env python3
import curses
import sys
import os
from modules.menu import Menu
from modules.course_parser import CourseParser


def main():
    # Get the actual directory of main.py, resolving any symlinks
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    courses_dir = os.path.join(script_dir, "courses")
    # Initialize CourseParser with the absolute path
    parser = CourseParser(courses_dir)
    courses = parser.parse_courses()

    if not courses:
        print("No valid courses found in the courses directory.")
        sys.exit(1)

    # Doc mode flag
    doc_mode = ("-d" in sys.argv) or ("--doc" in sys.argv)

    menu = Menu(courses, doc_mode=doc_mode)
    try:
        curses.wrapper(menu.run)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
