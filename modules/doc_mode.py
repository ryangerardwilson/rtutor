# ~/Apps/rtutor/modules/doc_mode.py
import curses
import sys
import time
import re
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

        # For comma-then-j/k
        self.last_comma_time = 0
        self.COMMA_TIMEOUT = 0.35

        # For search
        self.last_search_term = ""
        self.search_mode = False
        self.match_lines = []        # list of all line indices that match current term
        self.current_match_idx = -1 # index in match_lines of currently shown match

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
            max_allowed_offset = max(0, total_lines - available_height)

            # Clamp offset for normal scrolling
            self.offset = max(0, min(self.offset, max_allowed_offset))

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

                # Render content
                start_line = self.offset
                visible_lines = lines[start_line:start_line + available_height]

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

                # Footer info
                counter = f"Lesson {self.idx + 1}/{len(self.sequencer.lessons)}"
                scroll_info = ""
                if total_lines > available_height:
                    top = self.offset + 1
                    bottom = min(self.offset + available_height, total_lines)
                    scroll_info = f"  [{top}-{bottom}/{total_lines}]"

                try:
                    stdscr.addstr(max_y - 2, 0, counter + scroll_info, curses.color_pair(1))
                    stdscr.clrtoeol()
                except curses.error:
                    pass

                # Bottom line - UPDATED INSTRUCTIONS
                if self.search_mode:
                    prompt = f"/{self.last_search_term}"
                    try:
                        stdscr.addstr(max_y - 1, 0, prompt[:max_x], curses.color_pair(1))
                        stdscr.clrtoeol()
                    except curses.error:
                        pass
                else:
                    instr = "l=next h=prev r=rote t=teleport i=edit b=mark Ctrl+j/k=½page ,j=end ,k=top /=search Alt+Enter=back"
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

            current_time = time.time()

            # === ENTER / EXIT SEARCH MODE ===
            if key == ord('/'):
                if self.search_mode:
                    self.search_mode = False
                    self.last_search_term = ""
                    self.match_lines = []
                    self.current_match_idx = -1
                    curses.curs_set(0)
                    stdscr.nodelay(True)
                    need_redraw = True
                else:
                    self.search_mode = True
                    self.last_search_term = ""
                    self.match_lines = []
                    self.current_match_idx = -1
                    curses.curs_set(1)
                    stdscr.nodelay(False)
                    need_redraw = True
                continue

            # === KEYS IN SEARCH MODE ===
            if self.search_mode:
                if key in (curses.KEY_ENTER, ord('\n'), ord('\r'), 10, 13):
                    term = self.last_search_term.strip()
                    if not term:
                        stdscr.nodelay(True)
                        continue

                    # If term changed or first time, recompute all matches
                    if not self.match_lines or self.last_search_term != term:
                        pattern = re.compile(re.escape(term), re.IGNORECASE)
                        self.match_lines = [i for i, line in enumerate(lines) if pattern.search(line)]
                        self.last_search_term = term
                        self.current_match_idx = -1  # will become 0 on first advance

                    # Advance to next match
                    if self.match_lines:
                        self.current_match_idx = (self.current_match_idx + 1) % len(self.match_lines)
                        match_line = self.match_lines[self.current_match_idx]
                        self.offset = match_line  # place exactly at top
                        need_redraw = True
                    else:
                        # No matches
                        stdscr.move(max_y - 1, 0)
                        stdscr.clrtoeol()
                        stdscr.addstr(max_y - 1, 0, f"No match for '{term}'", curses.A_BOLD)
                        stdscr.refresh()
                        curses.napms(800)
                        need_redraw = True

                    stdscr.nodelay(True)
                    continue

                elif key == 27:  # ESC
                    self.search_mode = False
                    self.last_search_term = ""
                    self.match_lines = []
                    self.current_match_idx = -1
                    curses.curs_set(0)
                    stdscr.nodelay(True)
                    need_redraw = True
                    continue

                elif key in (curses.KEY_BACKSPACE, 127, 8):
                    if self.last_search_term:
                        self.last_search_term = self.last_search_term[:-1]
                        stdscr.move(max_y - 1, len(self.last_search_term) + 1)
                        stdscr.clrtoeol()
                        need_redraw = True

                elif 32 <= key <= 126:
                    self.last_search_term += chr(key)
                    stdscr.addstr(chr(key))
                    need_redraw = True

                continue

            # === NORMAL MODE KEYS ===
            redraw_needed = False

            if key == 10:  # Ctrl+J
                half_page = max(1, available_height // 2)
                if self.offset < max_allowed_offset:
                    self.offset = min(self.offset + half_page, max_allowed_offset)
                    redraw_needed = True

            elif key == 11:  # Ctrl+K
                half_page = max(1, available_height // 2)
                if self.offset > 0:
                    self.offset = max(0, self.offset - half_page)
                    redraw_needed = True

            elif key == ord(','):
                self.last_comma_time = current_time

            elif key in (ord('j'), curses.KEY_DOWN):
                if (current_time - self.last_comma_time) < self.COMMA_TIMEOUT:
                    self.offset = max_allowed_offset
                    redraw_needed = True
                else:
                    if self.offset < max_allowed_offset:
                        self.offset += 1
                        redraw_needed = True

            elif key in (ord('k'), curses.KEY_UP):
                if (current_time - self.last_comma_time) < self.COMMA_TIMEOUT:
                    self.offset = 0
                    redraw_needed = True
                else:
                    if self.offset > 0:
                        self.offset -= 1
                        redraw_needed = True

            elif key not in (-1, ord(','), ord('j'), ord('k'), curses.KEY_DOWN, curses.KEY_UP, 10, 11):
                self.last_comma_time = 0

            # Navigation & other keys
            if key == 3:
                sys.exit(0)
            elif key == 27:
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
