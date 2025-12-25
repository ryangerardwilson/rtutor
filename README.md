# RTutor

AI slop is a problem. Inculcating great programming taste is the antidote. rtutor is an attempt to disseminate that antidote.

## Table of Contents

- [Preface](#preface)
- [Featured Courses](#featured-courses)
- [Installation](#installation)
- [Usage](#usage)
- [Doc Mode Features](#doc-mode-features)
- [DOC and CAT modes](#doc-and-cat-modes)
- [Adding Courses](#adding-courses)

## Preface

rtutor parses intentionally curated Markdown files in a `courses/` directory, turning them into interactive typing lessons. Each course has parts and optional sections. Lessons are code blocks you must type accurately—with the objective of embedding taste through repetition. Start from the basics. Yes, even you.

## Featured Courses

Built-ins. Minimal excuses.

1. **python.md**: Hello World, primitives, loops, operators, functions, I/O. Great for scripting and MVPs. Debugging tutorials. Use Python to implement the Relational Model for elegant data science operations.
2. **sql.md**: SQL was designed as a printing language; it has since evolved into a programming language. This has led to ugly queries. Learn to write readable, not-awful SQL with SOLID principles.
3. **c.md**: `printf`, types, control flow, IO, constants. The metal is cold. Learn it anyway.
4. **unix.md**: Terminal typing, `ls`, `tmux`, `grep`, `find`, `iwctl`. More coming.
5. **a.md**: High-Level Assembly (HLA). Hello World, variables. Use it to ramp into real assembly.

## Installation

On Omarchy (or any Arch derivative):

1. Install Python 3:
```
sudo pacman -S python
```

2. Put the repo somewhere sane:
```
git clone <this-repo> ~/Apps/rtutor
```

3. Make a proper launcher. Symlink `init.sh` into your PATH as `rtutor`:
```
chmod +x ~/Apps/rtutor/init.sh
mkdir -p ~/.local/bin
ln -sf ~/Apps/rtutor/init.sh ~/.local/bin/rtutor
```

4. Ensure `~/.local/bin` is on PATH (fix your shell if it isn’t):
```
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
command -v rtutor || echo "PATH not set right"
```

Use `/usr/local/bin` with `sudo` if you want it system-wide. Own the consequences.

Why the symlink? Because `init.sh` sets `TERM` correctly (tmux/odd terminals) so curses doesn’t puke colors. Use it.

If you’re on some other distro, figure it out. This ain’t Windows.

## Usage

You made the symlink — use it.

Start interactive tutor:
```
rtutor
```

Menus:
- Scans `courses/` for `.md` files.
- Navigate with arrows; Enter selects.
- Type exactly. Backspace works. Tab inserts spaces if it matches indentation.
- Enter submits lines or skips blanks.
- Accuracy < 90%? You restart. Cope.
- Finish a sequence? You get ASCII art. Press any key to exit.

If no courses are found, it tells you and exits. Put `.md` files in `courses/`.

If you insist on bypassing the init script (don’t):
```
./init.sh
./main.py
```
But the supported, recommended way is: `rtutor`.

Markdown format for courses:
```
# Course Name
## Part Name
### Section Name (optional)
#### Lesson Name
    code block here
    multiline ok
```

## Doc Mode Features

Launch doc mode with:
```
rtutor -d
rtutor --doc
```

Doc mode provides a read-only viewer with powerful navigation and editing tools.

### Bookmarks
- While viewing any lesson in doc mode: press `b` to bookmark the current lesson.
- Bookmarks are persisted in `~/.config/rtutor/bookmarks.conf` in a clean, human-readable hierarchical format. Example:
```
Course: Python
Part: Part III: Data Science (ditch Excel)
Section: Section 1: Vanilla Numpy & Pandas
Lesson: Lesson 4: Modifications / Cleaning Based on Initial Inspection

Course: Unix
Part: Part I: Typing in the Terminal
Section:
Lesson: Lesson 1: Multiline Strings
```
- From the main menu in doc mode, press `b` to open the bookmark list.
- Navigate with `j`/`k`, press `Enter` or `l` to jump directly to the bookmarked lesson.
- Press `dd` (quick succession) on a selected bookmark to delete it. Slow separate `d` presses do nothing.

### In-place Editing
- While viewing a lesson in doc mode: press `i` to edit the lesson in your `$EDITOR` (defaults to `vim`).
- The source Markdown file opens at the exact lesson heading.
- Save and quit → rtutor automatically reloads the updated course.
- You stay on the edited lesson (or the closest match if the name changed).

### Other Doc Mode Keys
- `h` / `l` — previous / next lesson
- `r` — rote mode (10 reps)
- `j` — jump mode (type through entire sequence)
- `esc` — back to menu

## DOC and CAT modes

Read-only doc viewer, direct fuzzy jump that skips menus, and a non-interactive "cat" that spits content to stdout. Invoke with `rtutor`.

Doc mode:
```
rtutor -d
rtutor --doc
```
- `-d` with no args drops you into doc-mode menus.
- Pass quoted args after `-d` to filter by titles. Tokens map to:
  - `[L]` -> lesson
  - `[C, L]` -> course, lesson
  - `[C, P, L]` -> course, part, lesson
  - `[C, P, S, L]` -> course, part, section, lesson

Fuzzy matching is case-insensitive and punctuation-insensitive. Threshold: 70%.

Fuzzy rules:
- Single-word token matches individual words in the target title with >= 70% similarity.
- Multi-word token uses a sliding window of the same word count over the target title; any consecutive n-word window scoring >= 70% is a match.

Examples (from `courses/python.md`):
```
rtutor -d "repl"
# Matches lessons with a word fuzzy-matching "repl", e.g., "Lesson 1: Running the REPL"

rtutor -d "python" "repl"
# Only matches lessons titled "repl" inside course "python"

rtutor -d "python" "dipping toes" "repl"
```

What you get with `-d`:
- Matches open in the doc viewer as a linear list with context like "Course > Part > Section > Lesson".
- No matches? It says so and exits with code 1.

Cat mode: print the single best match to stdout (non-interactive):
```
rtutor -c <tokens...>
rtutor --cat <tokens...>
```
- Same token mapping and fuzzy rules as `-d`.
- Picks one best lesson match and prints it with ANSI colors.
- Exits 0 on success, 1 on no match.

## Adding Courses

Drop Markdown files into `courses/`. Keep it simple:

- Top-level: `# Course`
- Parts: `## Part`
- Optional sections: `### Section`
- Lessons: `#### Lesson` with an indented code block underneath

No fancy parsing — fast and predictable. Add content, not fluff.

