# ~/Apps/rtutor/modules/doc_mode.py
import curses
import sys
from .structs import Lesson
from .rote_mode import RoteMode
from .jump_mode import JumpMode


class DocMode:
    def __init__(self, sequencer):
        self.sequencer = sequencer

    def run(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)

        idx = 0
        need_redraw = True

        while True:
            if need_redraw:
                stdscr.clear()

                # Title bar
                title = f"{self.sequencer.name} | {self.sequencer.lessons[idx].name}"
                try:
                    stdscr.addstr(0, 0, title, curses.color_pair(1))
                    stdscr.move(0, len(title))
                    stdscr.clrtoeol()
                except curses.error:
                    pass

                # Content
                lines = self.sequencer.lessons[idx].content.strip().splitlines()
                max_y, max_x = stdscr.getmaxyx()
                row = 2
                for line in lines:
                    disp = line.replace("\t", "    ")
                    if row >= max_y - 2:
                        break
                    try:
                        stdscr.addstr(row, 0, disp[:max_x], curses.color_pair(1))
                        stdscr.clrtoeol()
                    except curses.error:
                        pass
                    row += 1

                # Clear remaining lines
                for r in range(row, max_y - 2):
                    try:
                        stdscr.move(r, 0)
                        stdscr.clrtoeol()
                    except curses.error:
                        pass

                # Footer
                footer_left = f"Lesson {idx + 1}/{len(self.sequencer.lessons)}"
                instr = "Doc mode: ← prev | → next | r rote | j jump | Esc back"
                try:
                    stdscr.addstr(max_y - 2, 0, footer_left, curses.color_pair(1))
                except curses.error:
                    pass
                try:
                    stdscr.addstr(max_y - 1, 0, instr, curses.color_pair(1))
                except curses.error:
                    pass

                stdscr.refresh()
                need_redraw = False

            # Input handling
            key = stdscr.getch()
            if key == -1:
                continue

            changed = False

            # ← Previous lesson
            if key == curses.KEY_LEFT:
                if idx > 0:
                    idx -= 1
                    changed = True

            # → Next lesson
            elif key == curses.KEY_RIGHT:
                if idx < len(self.sequencer.lessons) - 1:
                    idx += 1
                    changed = True

            # Esc / Ctrl+C → exit
            elif key in (27, 3):  # 27 = Esc, 3 = Ctrl+C
                return False

            # R → enter rote mode
            elif key in (ord('r'), ord('R')):
                rote = RoteMode(self.sequencer.name, self.sequencer.lessons[idx])
                rote_completed = rote.run(stdscr)
                stdscr.nodelay(True)
                curses.curs_set(0)
                need_redraw = True

            # J → enter jump mode
            elif key in (ord('j'), ord('J')):
                jump = JumpMode(self.sequencer.name, self.sequencer.lessons, idx)
                final_idx = jump.run(stdscr)
                if final_idx is not None:
                    if final_idx >= len(self.sequencer.lessons):
                        return True  # Finished whole sequence
                    idx = final_idx
                stdscr.nodelay(True)
                curses.curs_set(0)
                need_redraw = True

            if changed:
                need_redraw = True
