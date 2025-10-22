import curses
import sys
from .structs import Lesson
from .ascii import boom_art


class LessonSequencer:
    def __init__(self, name, lessons):
        self.name = name  # Sequence name (e.g., "Basic Typing")
        self.lessons = lessons  # List of Lesson objects

    def run(self, stdscr):
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, -1)

        stdscr.bkgd(" ", curses.color_pair(1))
        stdscr.nodelay(True)  # Non-blocking input to batch keys

        for lesson in self.lessons:
            stdscr.clear()  # Clear once per lesson
            stdscr.refresh()
            curses.curs_set(2)  # Set block cursor for each lesson
            # Split lesson content into lines, preserving all lines including empty ones, but strip trailing/leading whitespace to avoid bogus empty lines
            lines = lesson.content.strip().splitlines()
            # For each line, store non-tab characters and tab positions
            processed_lines = []
            tab_positions = []  # List of lists, each containing tab indices for a line
            for line in lines:
                non_tabs = [c for c in line if c != "\t"]  # Characters to type
                tabs = [i for i, c in enumerate(line) if c == "\t"]  # Tab indices
                processed_lines.append(non_tabs)
                tab_positions.append(tabs)
            current_line = 0
            user_inputs = [[] for _ in lines]  # Store input for non-tab chars
            completed = False  # Track lesson completion
            lesson_finished = False  # Track if all characters have been typed over
            need_redraw = True  # Initial draw

            while not completed:
                if need_redraw:
                    # No clear here - overwrite and clear to eol
                    title = f"{self.name} | {lesson.name}"
                    try:
                        stdscr.addstr(0, 0, title, curses.color_pair(1))
                        stdscr.move(0, len(title))
                        stdscr.clrtoeol()
                    except curses.error:
                        pass

                    # Clear line 1 if needed (empty)
                    try:
                        stdscr.move(1, 0)
                        stdscr.clrtoeol()
                    except curses.error:
                        pass

                    # Display all lines, showing tabs as four spaces and preserving blank lines
                    display_row = 2
                    for i, line in enumerate(lines):
                        target_text = line
                        user_input = user_inputs[i]
                        display_pos = (
                            0  # Position in display (including tabs as 4 spaces)
                        )
                        input_pos = 0  # Position in user_input (non-tab chars only)

                        # Show target text with user input overlay
                        for j, char in enumerate(target_text):
                            if char == "\t":
                                # Display tab as 4 spaces
                                for _ in range(4):
                                    try:
                                        stdscr.addch(
                                            display_row,
                                            display_pos,
                                            " ",
                                            curses.color_pair(1),
                                        )
                                    except curses.error:
                                        pass
                                    display_pos += 1
                            else:
                                display_char = char
                                if input_pos < len(user_input):
                                    if (
                                        input_pos < len(processed_lines[i])
                                        and user_input[input_pos]
                                        == processed_lines[i][input_pos]
                                    ):
                                        display_char = user_input[input_pos]
                                    else:
                                        display_char = "█"  # Block for incorrect
                                    input_pos += 1
                                if display_char == "\n":
                                    display_char = "↵"
                                try:
                                    stdscr.addch(
                                        display_row,
                                        display_pos,
                                        display_char,
                                        curses.color_pair(1),
                                    )
                                except curses.error:
                                    pass
                                display_pos += 1
                        # Display extra inputs as blocks
                        while input_pos < len(user_input):
                            try:
                                stdscr.addch(
                                    display_row, display_pos, "█", curses.color_pair(1)
                                )
                            except curses.error:
                                pass
                            display_pos += 1
                            input_pos += 1

                        # Clear to end of line
                        try:
                            stdscr.move(display_row, display_pos)
                            stdscr.clrtoeol()
                        except curses.error:
                            pass

                        display_row += 1

                    # Get terminal dimensions
                    max_y, max_x = stdscr.getmaxyx()

                    # Clear extra lines between content and stats
                    for row in range(display_row, max_y - 2):
                        try:
                            stdscr.move(row, 0)
                            stdscr.clrtoeol()
                        except curses.error:
                            pass

                    # Display stats at bottom - 2
                    stats = f"Typed {sum(len(user_inputs[i]) for i in range(len(lines)) if processed_lines[i])}/{sum(len(line) for line in processed_lines if line)} chars"
                    try:
                        stdscr.addstr(
                            max_y - 2,
                            0,
                            stats,
                            curses.color_pair(1),
                        )
                        stdscr.move(max_y - 2, len(stats))
                        stdscr.clrtoeol()
                    except curses.error:
                        pass

                    # Display instructions at bottom - 1
                    if lesson_finished:
                        instr = "Lesson complete! Hit n for next lesson or esc to exit"
                    else:
                        instr = "Ctrl+R ->restart | ESC -> quit"
                    try:
                        stdscr.addstr(
                            max_y - 1,
                            0,
                            instr,
                            curses.color_pair(1),
                        )
                        stdscr.move(max_y - 1, len(instr))
                        stdscr.clrtoeol()
                    except curses.error:
                        pass

                    # Compute cursor column correctly, accounting for tabs and extras (only if not finished)
                    if not lesson_finished:
                        cursor_col = 0
                        input_pos = 0
                        if current_line < len(lines):
                            for char in lines[current_line]:
                                if char == "\t":
                                    cursor_col += 4
                                else:
                                    if input_pos < len(user_inputs[current_line]):
                                        input_pos += 1
                                        cursor_col += 1
                                    else:
                                        break
                        # Add columns for extra inputs
                        cursor_col += len(user_inputs[current_line]) - input_pos

                        display_row = 2 + current_line
                        try:
                            stdscr.move(display_row, cursor_col)
                        except curses.error:
                            pass
                    else:
                        curses.curs_set(0)  # Hide cursor when finished

                    stdscr.refresh()
                    need_redraw = False

                # Batch process all queued input keys without redrawing
                changed = False
                while True:
                    try:
                        key = stdscr.getch()
                        if key == -1:
                            break

                        changed = True
                        if key == 3:  # Ctrl+C
                            sys.exit(0)

                        if lesson_finished:
                            if key == ord("n") or key == ord("N"):
                                completed = True
                            elif key == 27:  # ESC
                                return False
                            # Ignore other keys
                        else:
                            if key == 18:  # Ctrl+R
                                user_inputs = [[] for _ in lines]  # Reset inputs
                                current_line = 0  # Restart lesson
                            elif key == 27:  # ESC
                                return False
                            elif key in (
                                curses.KEY_BACKSPACE,
                                127,
                                8,
                            ):  # Backspace, including Ctrl+H
                                if user_inputs[current_line]:
                                    user_inputs[current_line].pop()
                            elif key in (curses.KEY_ENTER, 10, 13):  # Enter
                                if (
                                    user_inputs[current_line]
                                    == processed_lines[current_line]
                                ):
                                    if current_line < len(lines) - 1:
                                        current_line += 1
                            elif key == 9:  # Tab key
                                if processed_lines[
                                    current_line
                                ]:  # Only allow input on non-empty lines
                                    required_len = len(processed_lines[current_line])
                                    current_len = len(user_inputs[current_line])
                                    if current_len >= required_len:
                                        pass  # Ignore if at or beyond required
                                    else:
                                        # Append four spaces for Tab key
                                        next_chars = "".join(
                                            processed_lines[current_line][current_len:]
                                        )
                                        if next_chars.startswith(
                                            "    "
                                        ):  # Check if next four chars are spaces
                                            user_inputs[current_line].extend(
                                                [" ", " ", " ", " "]
                                            )
                            else:  # Handle printable characters
                                typed_char = None
                                if 32 <= key <= 126:  # Printable ASCII
                                    typed_char = chr(key)
                                if typed_char:
                                    required_len = len(processed_lines[current_line])
                                    current_len = len(user_inputs[current_line])
                                    if current_len >= required_len:
                                        pass  # Ignore extras
                                    else:
                                        user_inputs[current_line].append(typed_char)

                    except KeyboardInterrupt:
                        sys.exit(0)
                    except curses.error:
                        pass

                # Check if lesson is finished after batch processing
                all_lines_typed = all(
                    user_inputs[i] == processed_lines[i] for i in range(len(lines))
                )
                if all_lines_typed:
                    lesson_finished = True

                if (
                    changed or lesson_finished
                ):  # Redraw if anything changed or just finished
                    need_redraw = True

            # Lesson completed successfully
            if completed:
                continue  # Move to next lesson in sequence

        # All lessons completed, display centered boom_art at top
        stdscr.clear()
        max_y, max_x = stdscr.getmaxyx()  # Get terminal dimensions
        art_lines = boom_art.splitlines()  # Split ASCII art into lines

        # Calculate content width for centering boom_art
        content_width = max(len(line) for line in art_lines)

        # Display boom_art starting 2 lines from top, centered horizontally
        start_y = 2  # Fixed offset from top
        for i, line in enumerate(art_lines):
            if line:  # Render raw line, including all spaces
                x_pos = (max_x - content_width) // 2
                if x_pos < 0:
                    line = line[:max_x]  # Truncate if line exceeds terminal width
                    x_pos = 0
                try:
                    stdscr.addstr(start_y + i, x_pos, line, curses.color_pair(1))
                except curses.error:
                    pass  # Ignore errors if art exceeds screen bounds
        try:
            stdscr.addstr(max_y - 1, 0, "Press any key to exit.", curses.color_pair(1))
        except curses.error:
            pass
        stdscr.refresh()
        stdscr.nodelay(False)  # Back to blocking for final getch
        stdscr.getch()  # Wait for keypress to exit

        return True
