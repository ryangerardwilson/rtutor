import curses
import sys
from .structs import Lesson
from .doc_mode import DocMode
from .boom import Boom


class LessonSequencer:
    def __init__(self, name, lessons, doc_mode=False, source_file=None):
        self.name = name  # Sequence name (e.g., "Basic Typing")
        self.lessons = lessons  # List of Lesson objects
        self.doc_mode = doc_mode
        self.source_file = source_file

    def run(self, stdscr):
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, -1)

        stdscr.bkgd(" ", curses.color_pair(1))

        if self.doc_mode:
            doc = DocMode(self)
            doc_result = doc.run(stdscr)
            if doc_result == "ordinary":
                return self._run_ordinary(stdscr)
            else:
                return doc_result

        return self._run_ordinary(stdscr)

    def _run_ordinary(self, stdscr):
        stdscr.nodelay(True)

        for lesson in self.lessons:
            stdscr.clear()
            stdscr.refresh()
            curses.curs_set(2)

            # Use rstrip() to remove final newline but keep internal empty lines
            lines = lesson.content.rstrip().splitlines() or [""]
            total_lines = len(lines)

            # Preprocess: separate tabs and typing chars
            processed_lines = []
            is_skip = []
            for line in lines:
                non_tabs = [c for c in line if c != "\t"]
                processed_lines.append(non_tabs)
                is_skip.append(line.lstrip().startswith(("#!", "//!", "--!")))

            # Virtual scrolling state
            offset = 0
            min_lines_below = 3  # Try to keep 3 lines visible below current line
            current_line = 0
            user_inputs = [[] for _ in lines]
            lesson_finished = False
            need_redraw = True

            while True:
                max_y, max_x = stdscr.getmaxyx()

                # Available rows for content: total height - header (2) - footer (2)
                available_height = max(0, max_y - 4)
                content_start_y = 2  # Title on row 0, empty row 1

                # === Smart offset adjustment — exactly like vios ===
                if total_lines > available_height:
                    visible_top = offset
                    visible_bottom = offset + available_height - 1
                    lines_below = total_lines - 1 - current_line

                    # Try to keep min_lines_below context below cursor
                    if lines_below < min_lines_below and current_line > visible_bottom - min_lines_below:
                        desired = max(0, current_line - (available_height - min_lines_below - 1))
                        offset = desired
                    elif current_line < visible_top:
                        offset = current_line
                    elif current_line >= visible_top + available_height:
                        offset = current_line - available_height + 1

                    # Clamp
                    offset = max(0, min(offset, total_lines - available_height))
                else:
                    offset = 0

                # Slice visible data
                start_idx = offset
                end_idx = min(offset + available_height, total_lines)
                visible_range = range(start_idx, end_idx)

                if need_redraw:
                    # Title
                    title = f"{self.name} | {lesson.name}"
                    try:
                        stdscr.addstr(0, 0, title[:max_x], curses.color_pair(1))
                        stdscr.clrtoeol()
                    except curses.error:
                        pass

                    # Clear row 1
                    try:
                        stdscr.move(1, 0)
                        stdscr.clrtoeol()
                    except curses.error:
                        pass

                    # Render only visible lines
                    for local_i, global_i in enumerate(visible_range):
                        row = content_start_y + local_i
                        line = lines[global_i]
                        user_input = user_inputs[global_i]
                        display_pos = 0
                        input_pos = 0

                        for char in line:
                            if char == "\t":
                                for _ in range(4):
                                    try:
                                        stdscr.addch(row, display_pos, " ", curses.color_pair(1))
                                    except:
                                        pass
                                    display_pos += 1
                            else:
                                ch = char
                                if input_pos < len(user_input):
                                    if (input_pos < len(processed_lines[global_i]) and
                                        user_input[input_pos] == processed_lines[global_i][input_pos]):
                                        ch = user_input[input_pos]
                                    else:
                                        ch = "█"
                                    input_pos += 1
                                if ch == "\n":
                                    ch = "↵"
                                try:
                                    stdscr.addch(row, display_pos, ch, curses.color_pair(1))
                                except:
                                    pass
                                display_pos += 1

                        # Extra wrong characters
                        while input_pos < len(user_input):
                            try:
                                stdscr.addch(row, display_pos, "█", curses.color_pair(1))
                            except:
                                pass
                            display_pos += 1
                            input_pos += 1

                        # Clear rest of line
                        try:
                            stdscr.move(row, display_pos)
                            stdscr.clrtoeol()
                        except:
                            pass

                    # Clear remaining rows
                    for r in range(content_start_y + (end_idx - start_idx), max_y - 2):
                        try:
                            stdscr.move(r, 0)
                            stdscr.clrtoeol()
                        except:
                            pass

                    # Stats + scroll indicator
                    typed = sum(len(ui) for i, ui in enumerate(user_inputs) if not is_skip[i])
                    total = sum(len(p) for i, p in enumerate(processed_lines) if not is_skip[i])
                    stats = f"Typed {typed}/{total} chars"

                    scroll_info = ""
                    if total_lines > available_height:
                        top = offset + 1
                        bottom = offset + (end_idx - start_idx)
                        scroll_info = f"  [{top}-{bottom}/{total_lines}]"

                    try:
                        stdscr.addstr(max_y - 2, 0, stats + scroll_info, curses.color_pair(1))
                        stdscr.clrtoeol()
                    except:
                        pass

                    # Instructions
                    instr = ("Lesson complete! Hit l for next or esc to exit"
                             if lesson_finished else "Ctrl+R → restart | ESC → quit")
                    try:
                        stdscr.addstr(max_y - 1, 0, instr, curses.color_pair(1))
                        stdscr.clrtoeol()
                    except:
                        pass

                    # Cursor position
                    if not lesson_finished:
                        cursor_row = content_start_y + (current_line - offset)
                        cursor_col = 0
                        input_pos = 0
                        for char in lines[current_line]:
                            if char == "\t":
                                cursor_col += 4
                            else:
                                if input_pos < len(user_inputs[current_line]):
                                    input_pos += 1
                                    cursor_col += 1
                                else:
                                    break
                        cursor_col += len(user_inputs[current_line]) - input_pos
                        try:
                            stdscr.move(cursor_row, cursor_col)
                        except:
                            pass
                    else:
                        curses.curs_set(0)

                    stdscr.refresh()
                    need_redraw = False

                # === Input loop ===
                changed = False
                while True:
                    key = stdscr.getch()
                    if key == -1:
                        break
                    changed = True

                    if key == 3:  # Ctrl+C
                        sys.exit(0)

                    if lesson_finished:
                        if key in (ord('l'), ord('L')):
                            break  # Next lesson
                        elif key == 27:
                            return False
                    else:
                        if key == 18:  # Ctrl+R
                            user_inputs = [[] for _ in lines]
                            current_line = 0
                            lesson_finished = False
                        elif key == 27:
                            return False
                        elif is_skip[current_line]:
                            if key in (curses.KEY_ENTER, 10, 13):
                                if current_line < total_lines - 1:
                                    current_line += 1
                        else:
                            if key in (curses.KEY_BACKSPACE, 127, 8):
                                if user_inputs[current_line]:
                                    user_inputs[current_line].pop()
                            elif key in (curses.KEY_ENTER, 10, 13):
                                if user_inputs[current_line] == processed_lines[current_line]:
                                    if current_line < total_lines - 1:
                                        current_line += 1
                            elif key == 9:  # Tab
                                cur_len = len(user_inputs[current_line])
                                req_len = len(processed_lines[current_line])
                                if cur_len < req_len:
                                    remaining = "".join(processed_lines[current_line][cur_len:])
                                    if remaining.startswith("    "):
                                        user_inputs[current_line].extend([" ", " ", " ", " "])
                            else:
                                if 32 <= key <= 126:
                                    ch = chr(key)
                                    if len(user_inputs[current_line]) < len(processed_lines[current_line]):
                                        user_inputs[current_line].append(ch)

                # Check if finished
                if all(is_skip[i] or user_inputs[i] == processed_lines[i] for i in range(total_lines)):
                    lesson_finished = True
                    changed = True

                if changed:
                    need_redraw = True

        # All lessons done
        from .boom import Boom
        boom = Boom("Press any key to exit.")
        boom.display(stdscr)
        return True
