import curses


def main(stdscr):
    curses.curs_set(1)
    stdscr.addstr("\033[1 q")  # blinking block
    stdscr.addstr(
        0, 0, "You should see a blinking block cursor. Press any key to exit."
    )
    stdscr.refresh()
    stdscr.getch()


curses.wrapper(main)
