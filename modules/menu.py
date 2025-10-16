# ~/Apps/rtutor/modules/menu.py
import curses


class Menu:
    def __init__(self, courses):
        self.courses = courses  # List of Course objects
        self.ascii_art = """
        Typing Tutor
        """

    def run(self, stdscr):
        curses.curs_set(0)  # Hide cursor
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Menu highlight
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Normal text

        selected = 0
        while True:
            stdscr.clear()
            stdscr.addstr(0, 0, self.ascii_art, curses.color_pair(2))

            for i, course in enumerate(self.courses):
                prefix = "> " if i == selected else "  "
                color = curses.color_pair(1) if i == selected else curses.color_pair(2)
                stdscr.addstr(6 + i, 2, f"{prefix}{course.name}", color)

            stdscr.addstr(
                6 + len(self.courses) + 1,
                2,
                "> Quit",
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
