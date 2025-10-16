# ~/Apps/rtutor/modules/menu.py
import curses
from .constants import ascii_art


class Menu:
    def __init__(self, courses):
        self.courses = courses  # List of Course objects
        self.ascii_art = ascii_art

    def run(self, stdscr):
        curses.curs_set(0)  # Hide cursor
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Menu highlight
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Normal text

        # Get terminal dimensions
        max_y, max_x = stdscr.getmaxyx()
        # Calculate starting y for menu (after ASCII art)
        ascii_lines = self.ascii_art.count("\n")
        menu_start_y = ascii_lines

        selected = 0
        while True:
            stdscr.clear()

            # Center ASCII art
            for i, line in enumerate(self.ascii_art.split("\n")):
                if line.strip():  # Skip empty lines
                    stdscr.addstr(
                        i, (max_x - len(line)) // 2, line, curses.color_pair(2)
                    )

            # Center menu items
            for i, course in enumerate(self.courses):
                prefix = "> " if i == selected else "  "
                text = f"{prefix}{course.name}"
                stdscr.addstr(
                    menu_start_y + i,
                    (max_x - len(text)) // 2,
                    text,
                    curses.color_pair(1) if i == selected else curses.color_pair(2),
                )

            # Center Quit option
            quit_text = "> Quit" if selected == len(self.courses) else "  Quit"
            stdscr.addstr(
                menu_start_y + len(self.courses),
                (max_x - len(quit_text)) // 2,
                quit_text,
                curses.color_pair(1)
                if selected == len(self.courses)
                else curses.color_pair(2),
            )

            stdscr.refresh()

            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected = max(0, selected - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                selected = min(len(self.courses), selected + 1)
            elif key in (curses.KEY_LEFT, ord("h")):
                selected = 0
            elif key in (curses.KEY_RIGHT, ord("l")):
                selected = len(self.courses)
            elif key in (curses.KEY_ENTER, 10, 13):
                if selected == len(self.courses):
                    break  # Quit
                from modules.lesson_sequencer import LessonSequencer

                sequencer = LessonSequencer(
                    self.courses[selected].name, self.courses[selected].lessons
                )
                sequencer.run(stdscr)
                stdscr.clear()
            elif key == 27:  # ESC
                break
