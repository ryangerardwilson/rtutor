#!/usr/bin/env python3
import curses
import sys
from modules.menu import Menu
from modules.course_parser import CourseParser


def main():
    # Initialize CourseParser with the courses directory
    parser = CourseParser("./courses")
    courses = parser.parse_courses()

    if not courses:
        print("No valid courses found in the courses directory.")
        sys.exit(1)

    menu = Menu(courses)
    try:
        curses.wrapper(menu.run)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
