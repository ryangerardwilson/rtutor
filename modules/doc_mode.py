# ~/Apps/rtutor/modules/doc_mode.py
import curses
import sys
import time
from .structs import Lesson
from .rote_mode import RoteMode
from .jump_mode import JumpMode
from .doc_editor import DocEditor
from .bookmarks import Bookmarks


class DocMode:
    def __init__(self, sequencer):
        self.sequencer = sequencer
        self.idx = 0
        self.offset = 0
        self.bookmarks = Bookmarks()

        if hasattr(sequencer, 'target_lesson_name'):
            for i, lesson in enumerate(sequencer.lessons):
                if lesson.name == sequencer.target_lesson_name:
                    self.idx = i
                    break

        # For detecting comma-then-j/k
        self.last_comma_time = 0
        self.COMMA_TIMEOUT = 0.35  # seconds to press j/k after comma

    def run(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        source_file = getattr(self.sequencer, "source_file", None)
        need_redraw = True

        while True:
            current_lesson = self.sequencer.lessons[self.idx]
            lines = current_lesson.content.splitlines()
            total_lines = len(lines)

            max_y, max_x = stdscr.getmaxyx()
            header_rows = 2
            footer_rows = 2
            available_height = max(0, max_y - header_rows - footer_rows)

            # Clamp offset
            if total_lines <= available_height:
                self.offset = 0
            else:
                self.offset = max(0, min(self.offset, total_lines - available_height))

            if need_redraw:
                stdscr.clear()

                # Title
                title = f"{self.sequencer.name} | {current_lesson.name}"
                try:
                    stdscr.addstr(0, 0, title[:max_x], curses.color_pair(1) | curses.A_BOLD)
                    stdscr.clrtoeol()
                except curses.error:
                    pass

                # Empty line
                try:
                    stdscr.move(1, 0)
                    stdscr.clrtoeol()
                except curses.error:
                    pass

                # Render visible lines
                start_line = self.offset
                end_line = min(start_line + available_height, total_lines)
                visible_lines = lines[start_line:end_line]

                for i, line in enumerate(visible_lines):
                    row = header_rows + i
                    disp = line.replace("\t", "    ")
                    try:
                        stdscr.addstr(row, 0, disp[:max_x], curses.color_pair(1))
                        stdscr.clrtoeol()
                    except curses.error:
                        pass

                # Clear below content
                for row in range(header_rows + len(visible_lines), max_y - footer_rows):
                    try:
                        stdscr.move(row, 0)
                        stdscr.clrtoeol()
                    except curses.error:
                        pass

                # Footer
                counter = f"Lesson {self.idx + 1}/{len(self.sequencer.lessons)}"
                scroll_info = ""
                if total_lines > available_height:
                    top = self.offset + 1
                    bottom = self.offset + len(visible_lines)
                    scroll_info = f"  [{top}-{bottom}/{total_lines}]"

                try:
                    stdscr.addstr(max_y - 2, 0, counter + scroll_info, curses.color_pair(1))
                    stdscr.clrtoeol()
                except curses.error:
                    pass

                instr = "l=next h=prev r=rote t=teleport i=edit b=mark Ctrl+j/k=½page ,j=end ,k=top esc=back"
                try:
                    stdscr.addstr(max_y - 1, 0, instr, curses.color_pair(1))
                    stdscr.clrtoeol()
                except curses.error:
                    pass

                stdscr.refresh()
                need_redraw = False

            key = stdscr.getch()
            if key == -1:
                continue

            redraw_needed = False
            current_time = time.time()

            # Ctrl+J and Ctrl+K for half-page scrolling
            if key == 10:  # Ctrl+J (often mapped to 10)
                half_page = max(1, available_height // 2)
                if self.offset < max(0, total_lines - available_height):
                    self.offset = min(self.offset + half_page, max(0, total_lines - available_height))
                    redraw_needed = True

            elif key == 11:  # Ctrl+K
                half_page = max(1, available_height // 2)
                if self.offset > 0:
                    self.offset = max(0, self.offset - half_page)
                    redraw_needed = True

            # Comma followed by j/k → go to end/start
            elif key == ord(','):
                self.last_comma_time = current_time
                # Don't redraw yet — wait for possible j/k

            elif key in (ord('j'), curses.KEY_DOWN):
                # Check if comma was pressed recently
                if (current_time - self.last_comma_time) < self.COMMA_TIMEOUT:
                    # ,j → go to bottom
                    if total_lines > available_height:
                        self.offset = total_lines - available_height
                    redraw_needed = True
                else:
                    # Normal single-line down
                    if self.offset < max(0, total_lines - available_height):
                        self.offset += 1
                        redraw_needed = True

            elif key in (ord('k'), curses.KEY_UP):
                # Check if comma was pressed recently
                if (current_time - self.last_comma_time) < self.COMMA_TIMEOUT:
                    # ,k → go to top
                    self.offset = 0
                    redraw_needed = True
                else:
                    # Normal single-line up
                    if self.offset > 0:
                        self.offset -= 1
                        redraw_needed = True

            # Any other key cancels the pending comma
            elif key not in (-1, ord(','), ord('j'), ord('k'), curses.KEY_DOWN, curses.KEY_UP, 10, 11):
                self.last_comma_time = 0  # far in the past

            # Existing navigation
            if key == 3:  # Ctrl+C
                sys.exit(0)
            elif key == 27:  # ESC
                return False
            elif key in (ord("l"), ord("L"), curses.KEY_RIGHT):
                if self.idx < len(self.sequencer.lessons) - 1:
                    self.idx += 1
                    self.offset = 0
                    redraw_needed = True
            elif key in (ord("h"), ord("H"), curses.KEY_LEFT):
                if self.idx > 0:
                    self.idx -= 1
                    self.offset = 0
                    redraw_needed = True
            elif key == ord("b"):
                import os
                script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                courses_dir = os.path.join(script_dir, "courses")
                from modules.course_parser import CourseParser
                parser = CourseParser(courses_dir)
                all_courses = parser.parse_courses()
                self.bookmarks.add(all_courses, self.sequencer.name, current_lesson.name)
                try:
                    stdscr.addstr(max_y - 1, 0, "Bookmarked!           ", curses.A_BOLD)
                    stdscr.refresh()
                    curses.napms(800)
                except:
                    pass
                redraw_needed = True
            elif key in (ord("r"), ord("R")):
                rote = RoteMode(self.sequencer.name, current_lesson)
                rote.run(stdscr)
                redraw_needed = True
            elif key in (ord("t"), ord("T")):
                jump = JumpMode(self.sequencer.name, self.sequencer.lessons, self.idx)
                final_idx = jump.run(stdscr)
                if final_idx is not None:
                    if final_idx >= len(self.sequencer.lessons):
                        return True
                    self.idx = final_idx
                    self.offset = 0
                redraw_needed = True
            elif key in (ord("i"), ord("I")):
                editor = DocEditor(source_file)
                result = editor.edit_lesson(stdscr, current_lesson.name, self.idx)
                if result:
                    reloaded_lessons, course_name, new_idx = result
                    self.sequencer.lessons = reloaded_lessons
                    self.sequencer.name = course_name
                    self.idx = new_idx
                    self.offset = 0
                redraw_needed = True

            if redraw_needed:
                need_redraw = True

        return False
