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

3. Make a proper launcher. Symlink the executable main.py into your PATH as `rtutor`:
```
chmod +x ~/Apps/rtutor/main.py
mkdir -p ~/.local/bin
ln -sf ~/Apps/rtutor/main.py ~/.local/bin/rtutor
```

4. Ensure `~/.local/bin` is on PATH (fix your shell if it isn’t):
```
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
command -v rtutor || echo "PATH not set right"
```

## Usage

You made the symlink — use it.

Start the app (doc mode is the default):
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

You can run directly (not recommended for daily use):
```
~/Apps/rtutor/main.py
```
But symlink to `~/.local/bin/rtutor` like above. 

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

Doc mode is the default now. Just run rtutor — you'll be in the read-only doc viewer with full navigation and editing tools. The old -d flag still works but is redundant unless you're using it with search tokens (see DOC and CAT modes).

Quick keys while viewing:
- b — bookmark current lesson
- i — edit the lesson in your $EDITOR (vim by default); rtutor will reload the course and try to keep your place
- h / l — previous / next lesson
- r — rote mode (10 reps)
- j — jump mode (type through entire sequence)
- / — search within the lesson (vim-style)
- v — visual select; y to copy selection (uses wl-copy if available)
- esc — back to menu

Bookmarks:
- Persisted to `~/.config/rtutor/bookmarks.conf` in a plain hierarchical format.
- From the main menu in doc mode press `b` to open and jump.
- In the bookmark list: j/k navigate, Enter or l to jump, dd (double-press d) to delete.

In-place editing:
- Press `i` on a lesson (if the source .md file is present).
- rtutor opens your editor at the lesson heading, you edit and save, and rtutor reloads and attempts to keep your location.

## DOC and CAT modes

Short and brutal: doc-mode is default. -d/--doc still accepted for compatibility; use them with tokens to do direct fuzzy searches.

Doc mode:
```
rtutor                 # launches doc-mode menus (default)
rtutor -d "token ..."  # runs a direct search and opens the matches in doc-mode viewer
```
- `rtutor` alone → menu-driven doc-mode.
- `rtutor -d` with no tokens is redundant (same as running rtutor).
- `rtutor -d "foo" "bar"` → treat tokens as [Course, Part, Section, Lesson] (fuzzy). See fuzzy rules below.
- Direct doc searches open results in the linear doc viewer. No matches → exits with code 1.

Cat mode: print the single best match to stdout (non-interactive):
```
rtutor -c <tokens...>
rtutor --cat <tokens...>
```
- Same token mapping and fuzzy rules as -d.
- Prints ANSI-colored lesson text to stdout. Exits 0 on success, 1 on no match.

Token mapping (fuzzy, case-insensitive):
- [L] -> search all courses for lesson fuzzy-matching L
- [C, L] -> course C, lesson L
- [C, P, L] -> course, part, lesson
- [C, P, S, L] -> course, part, section, lesson

Fuzzy rules:
- Single-word token matches individual words in the target with >= 70% similarity.
- Multi-word token uses a sliding window of that many words in the target; any window scoring >= 70% matches.
- Matching is punctuation-insensitive and case-insensitive.

Examples:
```
rtutor -d "repl"
rtutor -d "python" "repl"
rtutor -c "python" "dipping toes" "repl"
```

## Adding Courses

Drop Markdown files into `courses/`. Keep it simple:

- Top-level: `# Course`
- Parts: `## Part`
- Optional sections: `### Section`
- Lessons: `#### Lesson` with an indented code block underneath

No fancy parsing — fast and predictable. Add content, not fluff.

