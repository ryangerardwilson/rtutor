# ~/Apps/rtutor/main.py
import curses
import sys
from modules.menu import Menu
from modules.structs import Lesson, Course


def main():
    courses = [
        Course(
            "Basic Typing",
            [
                Lesson("Lesson1", "The quick brown fox jumps.\n\tThe lazy dog sleeps."),
                Lesson("Lesson2", "The quick brown dog jumps.\nThe lazy fox sleeps."),
            ],
        ),
        Course(
            "Basic Typing2",
            [
                Lesson("Lesson1", "The quick brown cat jumps.\nThe lazy cat sleeps."),
                Lesson(
                    "Lesson2", "The quick brown mouse jumps.\nThe lazy mouse sleeps."
                ),
            ],
        ),
    ]
    menu = Menu(courses)
    try:
        curses.wrapper(menu.run)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
