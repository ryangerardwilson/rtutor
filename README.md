# RTutor: The Fork in the Road to Escape AI Slop

AI slop is a problem. Inculcating great programming taste is the antidote.
rtutor is an attempt to disseminate that antidote.

## Table of Contents
- [Preface](#preface)
- [Featured Courses](#featured-courses)
- [Installation](#installation)
- [Usage](#usage)
- [Doc mode and fuzzy search](#doc-mode-and-fuzzy-search)
- [Adding Courses](#adding-courses)

## Preface

RTutor parses intentionally curated markdown files in a courses/ directory,
turning them into interactive typing lessons. Each course has parts and
optional sections. Lessons are code blocks you must type accurately—with the
objective of embedding taste through repetition. Start from the basics. Yes,
even you.

## Featured Courses

Built-ins. Minimal excuses.

1. a.md: High-Level Assembly (HLA). Hello World, variables. Use it to ramp into real assembly.
2. python.md: Hello World, primitives, loops, operators, functions, I/O. Great for scripting, not an excuse to ignore performance.
3. c.md: printf, types, control flow, IO, constants. The metal is cold. Learn it anyway.
4. unix.md: tmux, grep, find. More coming.

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

## Doc mode and fuzzy search

Read-only doc viewer and direct fuzzy jump that skips menus. Again: invoke with
rtutor.

- Launch doc mode:
  - rtutor -d
  - rtutor --doc

- Fuzzy search into doc mode:
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

What you get:
- Matches open in the doc viewer as a linear list with context like
  “Course > Part > Section > Lesson”.
- No matches? It says so and exits with code 1.

## Adding Courses

Drop markdown files into courses/. Keep it simple:

- Top-level: # Course
- Parts: ## Part
- Optional sections: ### Section
- Lessons: #### Lesson with an indented code block underneath

No fancy parsing—fast and predictable. Add content, not fluff.
