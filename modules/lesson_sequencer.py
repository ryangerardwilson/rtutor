import curses
import sys
from .structs import Lesson
from .ascii import boom_art


class LessonSequencer:
    def __init__(self, name, lessons):
        self.name = name  # Sequence name (e.g., "Basic Typing")
        self.lessons = lessons  # List of Lesson objects

    def run(self, stdscr):
        curses.curs_set(2)  # Set block cursor
        curses.start_color()
        curses.init_pair(
            1, curses.COLOR_WHITE, curses.COLOR_BLACK
        )  # White for all text

        for lesson in self.lessons:
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

            while not completed:
                stdscr.clear()
                stdscr.addstr(
                    0,
                    0,
                    f"{self.name} | {lesson.name}",
                    curses.color_pair(1),
                )

                # Display all lines, showing tabs as four spaces and preserving blank lines
                display_row = 2
                for i, line in enumerate(lines):
                    target_text = line
                    user_input = user_inputs[i]
                    display_pos = 0  # Position in display (including tabs as 4 spaces)
                    input_pos = 0  # Position in user_input (non-tab chars only)

                    # Show target text with user input overlay
                    for j, char in enumerate(target_text):
                        if char == "\t":
                            # Display tab as 4 spaces
                            for _ in range(4):
                                stdscr.addch(
                                    display_row, display_pos, " ", curses.color_pair(1)
                                )
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
                            stdscr.addch(
                                display_row,
                                display_pos,
                                display_char,
                                curses.color_pair(1),
                            )
                            display_pos += 1
                    # Display extra inputs as blocks
                    while input_pos < len(user_input):
                        stdscr.addch(
                            display_row, display_pos, "█", curses.color_pair(1)
                        )
                        display_pos += 1
                        input_pos += 1
                    display_row += 1

                # Get terminal dimensions
                max_y, _ = stdscr.getmaxyx()

                # Display stats at bottom - 2
                total_chars = sum(
                    len(line) for line in processed_lines if line
                )  # Count non-tab chars in non-empty lines
                typed_chars = sum(
                    len(user_inputs[i]) for i in range(len(lines)) if processed_lines[i]
                )
                stdscr.addstr(
                    max_y - 2,
                    0,
                    f"Typed {typed_chars}/{total_chars} chars",
                    curses.color_pair(1),
                )

                # Display instructions at bottom - 1
                stdscr.addstr(
                    max_y - 1,
                    0,
                    "Ctrl+N -> next lesson | Ctrl+R ->restart | ESC -> quit",
                    curses.color_pair(1),
                )

                # Compute cursor column correctly, accounting for tabs and extras
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
                stdscr.move(display_row, cursor_col)
                stdscr.refresh()

                try:
                    key = stdscr.getch()
                    if key == 3:  # Ctrl+C
                        sys.exit(0)
                    elif key == 18:  # Ctrl+R
                        user_inputs = [[] for _ in lines]  # Reset inputs
                        current_line = 0  # Restart lesson
                    elif key == 14:  # Ctrl+N for next lesson
                        # Check if all positions have been typed over (irrespective of correctness)
                        all_lines_typed = all(
                            len(user_inputs[i]) >= len(processed_lines[i])
                            for i in range(len(lines))
                            if processed_lines[i]  # Skip empty lines
                        )
                        if all_lines_typed:
                            completed = True
                        else:
                            user_inputs = [[] for _ in lines]  # Reset inputs
                            current_line = 0  # Start over
                    elif key == 27:  # ESC
                        return False
                    elif (
                        key == curses.KEY_BACKSPACE or key == 127
                    ):  # Backspace disabled
                        pass
                    elif key in (curses.KEY_ENTER, 10, 13):  # Enter
                        if current_line < len(lines) - 1:
                            current_line += 1  # Move to next line
                        # No completion check here anymore
                    elif key == 9:  # Tab key
                        if processed_lines[
                            current_line
                        ]:  # Only allow input on non-empty lines
                            required_len = len(processed_lines[current_line])
                            current_len = len(user_inputs[current_line])
                            if (
                                current_line == len(lines) - 1
                                and current_len >= required_len
                            ):
                                pass  # Ignore on last line if at or beyond required
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
                                    # Auto-advance if this pushed us into extra error territory (non-last lines)
                                    if (
                                        len(user_inputs[current_line]) > required_len
                                        and current_line < len(lines) - 1
                                    ):
                                        current_line += 1
                    else:  # Handle printable characters
                        typed_char = None
                        if 32 <= key <= 126:  # Printable ASCII
                            typed_char = chr(key)
                        if typed_char:
                            required_len = len(processed_lines[current_line])
                            current_len = len(user_inputs[current_line])
                            if (
                                current_line == len(lines) - 1
                                and current_len >= required_len
                            ):
                                pass  # Ignore extras on last line
                            else:
                                user_inputs[current_line].append(typed_char)
                                # Auto-advance if this was an extra error at the end (non-last lines)
                                if (
                                    len(user_inputs[current_line]) > required_len
                                    and current_line < len(lines) - 1
                                ):
                                    current_line += 1

                except KeyboardInterrupt:
                    sys.exit(0)
                except curses.error:
                    pass

            # Lesson completed successfully
            if completed:
                continue  # Move to next lesson in sequence

        # All lessons completed, display centered boom_art at top
        stdscr.clear()
        max_y, max_x = stdscr.getmaxyx()  # Get terminal dimensions
        art_lines = boom_art.splitlines()  # Split ASCII art into lines

        # Calculate content width for centering boom_art
        content_width = 0
        for line in art_lines:
            if line.strip():  # Only consider non-empty lines for width
                content_width = max(content_width, len(line.strip()))

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
        stdscr.addstr(max_y - 1, 0, "Press any key to exit.", curses.color_pair(1))
        stdscr.refresh()
        stdscr.getch()  # Wait for keypress to exit

        return True
