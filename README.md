AI slop is a problem. Inculcating great programming taste is the antidote.
rtutor is an attempt to disseminate that antidote.

## Table of Contents

- [Preface](#preface)
- [Featured Courses](#featured-courses)
- [Installation](#installation)
- [Usage](#usage)
- [DOC and CAT modes](#doc-and-cat-modes)
- [Adding Courses](#adding-courses)

## Preface

RTutor parses intentionally curated markdown files in a courses/ directory,
turning them into interactive typing lessons. Each course has parts and
optional sections. Lessons are code blocks you must type accurately—with the
objective of embedding taste through repetition. Start from the basics. Yes,
even you.

## Featured Courses

Built-ins. Minimal excuses.

1. python.md: Hello World, primitives, loops, operators, functions, I/O. Great
   for scripting, and MVPs. Debugging tutorials.  Use python to implement the 
   Relational Model for elegant data science operations.

2. sql.md: SQL was designed as a printing language, however, for better or for
   worse it has evolved into a programming language. This has led to ugly
   queries. Learn to write readable and not-awful SQL with SOLID principles.

3. c.md: printf, types, control flow, IO, constants. The metal is cold. Learn
   it anyway.

4. unix.md: terminal typing, ls, tmux, grep, find, iwctl. More coming.

5. a.md: High-Level Assembly (HLA). Hello World, variables. Use it to ramp into
   real assembly.

## Installation

On Omarchy (or any Arch derivative):

1. Install Python 3:
   - pacman -S python

2. Put the repo somewhere sane:
   - git clone <this-repo> ~/Apps/rtutor

3. Make a proper launcher. Symlink init.sh into your PATH as rtutor:
   - chmod +x ~/Apps/rtutor/init.sh
   - mkdir -p ~/.local/bin
   - ln -sf ~/Apps/rtutor/init.sh ~/.local/bin/rtutor

4. Ensure ~/.local/bin is on PATH (fix your shell if it isn’t):
   - echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   - source ~/.bashrc
   - command -v rtutor || echo "PATH not set right"

Use /usr/local/bin with sudo if you want it system-wide. Own the consequences.

Why the symlink? Because init.sh sets TERM correctly (tmux/odd terminals) so
curses doesn’t puke colors. Use it.

If you’re on some other distro, figure it out. This ain’t Windows.

## Usage

You made the symlink—use it.

- Start interactive tutor:
  - rtutor

- Menus:
  - Scans courses/ for .md files.
  - Navigate with arrows; Enter selects.
  - Type exactly. Backspace works. Tab inserts spaces if it matches indentation.
  - Enter submits lines or skips blanks.
  - Accuracy < 90%? You restart. Cope.
  - Finish a sequence? You get ASCII art. Press any key to exit.

If no courses are found, it tells you and exits. Put .md files in courses/.

If you insist on bypassing the init script (don’t):
- ./init.sh
- ./main.py
But the supported, recommended way is: rtutor

Markdown format:

    # Course Name
    ## Part Name
    ### Section Name (optional)
    #### Lesson Name
        code block here
        multiline ok

## DOC and CAT modes 

Read-only doc viewer, direct fuzzy jump that skips menus, and a non-interactive
“cat” that spits content to stdout. Again: invoke with rtutor.

- Launch doc mode:
  - rtutor -d
  - rtutor --doc
  - -d with no args drops you into doc-mode menus.

- Fuzzy search into doc mode (interactive):
  - Pass quoted args after -d to filter by titles. Tokens map to:
    - [L] -> lesson
    - [C, L] -> course, lesson
    - [C, P, L] -> course, part, lesson
    - [C, P, S, L] -> course, part, section, lesson
  - Matching is fuzzy, case-insensitive, and punctuation-insensitive. Threshold is 70%.

Fuzzy rules:
- Single-word token matches individual words in the target title with 70%+ similarity.
- Multi-word token uses a sliding window of the same word count over the target title; any consecutive n-word window scoring 70%+ is a match.

Examples (from courses/python.md):

    rtutor -d "repl"
    # Matches lessons with a word fuzzy-matching "repl", e.g., "Lesson 1: Running the REPL"
    # under: Python > Part I: Python Basics > Section 1: Dipping Toes.
    # Also matches with: Unix > Part III: Search Utils > Main > Lesson 1: Grep, Find & Locate

    rtutor -d "python" "repl"
    # [C, L] -> Course "Python", Lesson "Lesson 1: Running the REPL".
    # Only matches with: Python > Part I: Python Basics > Section 1: Dipping Toes > Lesson 1: Running the REPL

    # Likewise, we have:
    # [C, P, L]
    # [C, P, S, L]
    rtutor -d "python" "primitives"
    rtutor -d "python" "dipping toes" "repl"

What you get (-d):
- Matches open in the doc viewer as a linear list with context like
  “Course > Part > Section > Lesson”.
- No matches? It says so and exits with code 1.

Cat mode: print the single best match to stdout (non-interactive):
- Launch cat mode:
  - rtutor -c <tokens...>
  - rtutor --cat <tokens...>
- Same token mapping and fuzzy rules as -d:
  - [L], [C, L], [C, P, L], [C, P, S, L]
- Behavior:
  - Picks one best lesson match (lesson score dominates; course/part/section break ties).
  - Prints a header and the lesson content to stdout with ANSI colors.
  - Exits 0 on success.
  - No match? Prints a “No matching lessons found for -c arguments: …” message and exits 1.
- Precedence:
  - If you pass both -c and -d with tokens, -c wins. You asked to print; it prints.
- Tokens parsing:
  - Tokens are read after the flag until the next “-” option or end of argv.
  - -c with no tokens is ignored and you’ll fall back to the normal menus. Don’t do that.

Cat mode examples:

    rtutor -c "repl"
    # Best single lesson matching "repl" across all courses.

    rtutor -c "python" "repl"
    # Best match for lesson "repl" inside course "python".

    rtutor -c "python" "dipping toes" "repl"
    # Best match constrained by course and section.

## Adding Courses

Drop markdown files into courses/. Keep it simple:

- Top-level: # Course
- Parts: ## Part
- Optional sections: ### Section
- Lessons: #### Lesson with an indented code block underneath

No fancy parsing—fast and predictable. Add content, not fluff.
