import curses
from .ascii import title_ascii_art


class Menu:
    def __init__(self, courses):
        self.courses = courses
        self.title_ascii_art = title_ascii_art

    def run(self, stdscr):
        curses.curs_set(0)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)

        max_y, max_x = stdscr.getmaxyx()
        title_lines = self.title_ascii_art.count("\n")
        menu_start_y = title_lines + 4  # Gap: 1 for author, 2 blank lines, 1 extra

        # Calculate content width for centering title art
        content_width = 0
        for line in self.title_ascii_art.split("\n"):
            if line.strip():
                content_width = max(content_width, len(line.strip()))

        # Calculate menu width for centering the block
        menu_width = max(
            max(len(f"> {course.name}") for course in self.courses), len("> Quit")
        )

        selected = 0
        while True:
            stdscr.clear()

            # Render title ASCII art as is
            for i, line in enumerate(self.title_ascii_art.split("\n")):
                if line:  # Render raw line, including all spaces
                    x_pos = (max_x - content_width) // 2
                    if x_pos < 0:
                        line = line[:max_x]
                        x_pos = 0
                    stdscr.addstr(i, x_pos, line, curses.color_pair(2))

            # Render author text
            author_text = "By Ryan Gerard Wilson"
            stdscr.addstr(
                title_lines + 1,
                (max_x - len(author_text)) // 2,
                author_text,
                curses.color_pair(2),
            )

            # Render menu items, left-aligned but centered as a block
            menu_x_pos = (max_x - menu_width) // 2 if menu_width < max_x else 0
            for i, course in enumerate(self.courses):
                prefix = "> " if i == selected else "  "
                text = f"{prefix}{course.name}"
                stdscr.addstr(
                    menu_start_y + i,
                    menu_x_pos,
                    text,
                    curses.color_pair(1) if i == selected else curses.color_pair(2),
                )

            # Render Quit option
            quit_text = "> Quit" if selected == len(self.courses) else "  Quit"
            stdscr.addstr(
                menu_start_y + len(self.courses),
                menu_x_pos,
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
                    break
                from modules.lesson_sequencer import LessonSequencer

                sequencer = LessonSequencer(
                    self.courses[selected].name, self.courses[selected].lessons
                )
                sequencer.run(stdscr)
                stdscr.clear()
            elif key == 27:  # ESC
                break
