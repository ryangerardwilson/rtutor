# RTutor: The Fork in the Road to Escape AI Slop

AI slop is a problem. Inculcating great programming taste is the antidote.
rtutor is an attempt to disseminate the antidote.

RTutor parses intentionally curated markdown files in a `courses/` directory,
turning them into interactive typing lessons. Each course has parts and
lessons, which are basically code blocks you have to type accurately - with the
objective of sequentially embedding taste across the various layers of
complexity of computer programming. Experienced and new programmers, alike, are
advised to start ground up from the most basic lessons - for this very reason.

## Featured Courses

Here are the built-in courses, each designed to beat some fundamental
programming knowledge into your skull through sheer repetition. Don't whine
about the choices; they're solid starting points.

1. `hla.md`: This one's for those who want to dip their toes into assembly
without drowning in registers and opcodes right away. High-Level Assembly (HLA)
makes it almost readable, like a gateway drug to low-level programming. Covers
basics like Hello World and variables. If you're scared of real assembly, start
here—but don't stay too long.

2. `python.md`: Python, the language that's taken over the world because it's
easy and everyone uses it for everything. This course hits the essentials:
Hello World, primitive types, loops, operators, functions, and even the walrus
operator. Great for scripting on Omarchy without compiling headaches. But
remember, performance matters—don't get too cozy.

3. `c.md`: Ah, C—the mother of all modern languages. This course drills the
core: Hello World, types, printf formatting, arithmetic, control structures,
input/output, and constants. It's raw, it's powerful, and it'll teach you how
computers really work under the hood. Mandatory suffering for any serious
programmer.

4. `classic_unix_tools.md`: Learn classics like TMUX, grep, find. Many more to
be added over time.

## Installation

On Omarchy (or any Arch-based distro worth its salt):

1. Make sure you've got Python 3 installed. It's Omarchy, so `pacman -S python`
if you're living under a rock.

2. Curses is built-in with Python's standard library on Linux, so no extra
packages needed. If it whines, check your Python setup.

3. Clone or copy the repo to `~/Apps/rtutor` (as per your path).

4. That's it. No pip crap, no virtualenvs—pure Python scripts.

If you're on some other distro, figure it out. This ain't Windows.

## Usage

Fire it up from the terminal:

    cd ~/Apps/rtutor
    ./main.py

If you're one of those tmux weirdos, use ./init.sh instead to set the damn TERM
variable right and avoid curses screwing up the colors.

Or make it executable and symlink it somewhere in your PATH if you're lazy.

- It scans courses/ for .md files.
- Picks courses, parts, lessons via a curses menu (arrow keys, Enter).
- Type the code exactly. Backspace works, Tab inserts spaces if it matches.
- Hit Enter to submit lines or skip blanks.
- Accuracy below 90%? Restart the lesson. Deal with it.
- Finish all lessons in a sequence? Boom—ASCII art. Press any key to exit.

If no courses found, it bitches and quits. Add more .md files to courses/ in
the format:

    # Course Name

    ## Part Name

    ### Lesson Name

        code block here
        multiline ok

## Adding Courses

Just drop markdown files into courses/. Structure like the examples. Lessons
are code blocks under ### headers. Parts under ##. Course name from # top
header.

No fancy parsing—keeps it simple and fast.
