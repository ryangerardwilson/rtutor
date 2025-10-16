# ~/Apps/rtutor/modules/lesson_sequencer.py
import curses
import sys
from .structs import Lesson
from .ascii import boom_art


class LessonSequencer:
    def __init__(self, name, lessons, error_threshold_percent=0.10):
        self.name = name  # Sequence name (e.g., "Basic Typing")
        self.lessons = lessons  # List of Lesson objects
        self.error_threshold_percent = (
            error_threshold_percent  # Minimum accuracy required (e.g., 0.10 for 90%)
        )

    def run(self, stdscr):
        curses.curs_set(2)  # Set block cursor
        curses.start_color()
        curses.init_pair(
            1, curses.COLOR_WHITE, curses.COLOR_BLACK
        )  # White for all text

        for lesson in self.lessons:
            # Split lesson content into lines, preserving all lines including empty ones
            lines = lesson.content.splitlines()
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
            total_mistakes = 0  # Track mistakes across all lines
            nav_enters = 0  # Track correct navigation Enter presses
            completed = False  # Track lesson completion

            while not completed:
                stdscr.clear()
                stdscr.addstr(
                    0,
                    0,
                    f"Course: {self.name} | Lesson: {lesson.name} (ESC to quit, Ctrl+C to exit, Ctrl+R to restart)",
                    curses.color_pair(1),
                )

                # Display all lines, showing tabs as four spaces and preserving blank lines
                display_row = 2
                for i, line in enumerate(lines):
                    if not line.strip():  # Handle blank lines
                        stdscr.addstr(display_row, 0, "", curses.color_pair(1))
                        display_row += 1
                        continue
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
                total_chars_with_nav = total_chars + nav_enters
                accuracy = (
                    (
                        100
                        * (total_chars + nav_enters - total_mistakes)
                        / total_chars_with_nav
                    )
                    if total_chars_with_nav
                    else 100
                )
                stdscr.addstr(
                    max_y - 2,
                    0,
                    f"Typed {typed_chars}/{total_chars} chars, Nav Enters: {nav_enters}, Accuracy: {accuracy:.1f}%",
                    curses.color_pair(1),
                )

                # Display "Press Enter" at bottom - 1, if applicable
                if current_line < len(lines):
                    stdscr.addstr(
                        max_y - 1,
                        0,
                        "Press Enter to submit current line or move past blank line.",
                        curses.color_pair(1),
                    )

                # Move cursor to current line's input position, no tab skipping
                cursor_pos = len(user_inputs[current_line])  # Raw input position
                display_row = 2 + current_line
                stdscr.move(display_row, cursor_pos)
                stdscr.refresh()

                try:
                    key = stdscr.getch()
                    if key == 3:  # Ctrl+C
                        sys.exit(0)
                    elif key == 18:  # Ctrl+R
                        user_inputs = [[] for _ in lines]  # Reset inputs
                        total_mistakes = 0  # Reset mistakes
                        nav_enters = 0  # Reset navigation Enters
                        current_line = 0  # Restart lesson
                    elif key == 27:  # ESC
                        return False
                    elif key == curses.KEY_BACKSPACE or key == 127:  # Backspace
                        if user_inputs[current_line] and processed_lines[current_line]:
                            user_inputs[current_line].pop()
                    elif key in (curses.KEY_ENTER, 10, 13):  # Enter
                        if current_line < len(lines) - 1:
                            nav_enters += 1
                            current_line += 1  # Move to next line
                        else:
                            # Check if all lines are fully typed
                            all_lines_typed = all(
                                len(user_inputs[i]) == len(processed_lines[i])
                                for i in range(len(lines))
                                if processed_lines[i]  # Skip empty lines
                            )
                            if all_lines_typed and accuracy >= (
                                100 - self.error_threshold_percent * 100
                            ):
                                completed = True
                            else:
                                user_inputs = [[] for _ in lines]  # Reset inputs
                                total_mistakes = 0  # Reset mistakes
                                nav_enters = 0  # Reset navigation Enters
                                current_line = 0  # Start over
                    elif key == 9:  # Tab key
                        if processed_lines[
                            current_line
                        ]:  # Only allow input on non-empty lines
                            # Append four spaces for Tab key
                            next_chars = "".join(
                                processed_lines[current_line][
                                    len(user_inputs[current_line]) :
                                ]
                            )
                            if next_chars.startswith(
                                "    "
                            ):  # Check if next four chars are spaces
                                user_inputs[current_line].extend([" ", " ", " ", " "])
                            else:
                                total_mistakes += 1  # Incorrect input
                    else:  # Handle printable characters
                        typed_char = None
                        if 32 <= key <= 126:  # Printable ASCII
                            typed_char = chr(key)
                        if (
                            typed_char
                            and processed_lines[current_line]
                            and len(user_inputs[current_line])
                            < len(processed_lines[current_line])
                        ):
                            user_inputs[current_line].append(typed_char)
                            if (
                                typed_char
                                != processed_lines[current_line][
                                    len(user_inputs[current_line]) - 1
                                ]
                            ):
                                total_mistakes += 1

                except KeyboardInterrupt:
                    sys.exit(0)
                except curses.error:
                    pass

            # Lesson completed successfully
            if completed:
                continue  # Move to next lesson in sequence

        return True
