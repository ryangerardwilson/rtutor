# RTutor


AI slop is a problem. Inculcating great programming taste is the antidote.
rtutor is an attempt to disseminate that antidote.

## Table of Contents
- [Preface](#preface)
- [Featured Courses](#featured-courses)
- [Installation](#installation)
- [Usage](#usage)
- [Doc Mode Features](#doc-mode-features)
- [Doc Mode CLI Flags](#doc-mode-cli-flags)
- [Configuration](#configuration)
- [Project Layout](#project-layout)
- [Adding Courses](#adding-courses)

## Preface

rtutor parses intentionally curated Markdown files registered through its XDG
config (`${XDG_CONFIG_HOME:-~/.config}/rt/courses/`), turning them into
interactive typing lessons. Each course has parts and optional sections.
Lessons are code blocks you must type accurately—with the objective of
embedding taste through repetition. Start from the basics. Yes, even you.

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
- Scans `${XDG_CONFIG_HOME:-~/.config}/rt/courses/` for `.md` files named `course_<id>.md`.
- Navigate with arrows; Enter selects.
- Type exactly. Backspace works. Tab inserts spaces if it matches indentation.
- Enter submits lines or skips blanks.
- Accuracy < 90%? You restart. Cope.
- Finish a sequence? You get ASCII art. Press any key to exit.

If no courses are found, it tells you and exits. Put `.md` files (named
`course_<id>.md`) in `${XDG_CONFIG_HOME:-~/.config}/rt/courses/`.

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

Doc mode is the default now. Just run rtutor — you'll be in the read-only doc
viewer with full navigation and editing tools. The old -d flag still works but
is redundant unless you're using it with search tokens (see Doc Mode CLI
Flags).

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

## Doc Mode CLI Flags

Short and brutal: doc-mode is default. `-d/--doc` is still accepted for
compatibility; use it with tokens to do direct fuzzy searches.

```
rtutor                 # launches doc-mode menus (default)
rtutor -d "token ..."  # runs a direct search and opens the matches in doc-mode viewer
rtutor -t                  # upload (train) all registered courses to Grok Collections
rtutor -q "How do I inspect a df?"  # sync courses to Grok and answer a question
```

- `rtutor` alone → menu-driven doc-mode.
- `rtutor -d` with no tokens is redundant (same as running rtutor).
- `rtutor -d "foo" "bar"` → treat tokens as [Course, Part, Section, Lesson] (fuzzy). See fuzzy rules below.
- `rtutor -t` → force a full re-upload (training) of every course to Grok Collections.
- `rtutor -q "question"` → non-interactive Q&A powered by Grok Collections (requires API keys).
- Direct doc searches open results in the linear doc viewer. No matches → exits with code 1.

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
rtutor -t
rtutor -q "show me python repl basics"
```

The former `-c/--cat` command has been retired.

## Configuration

rtutor now follows the XDG base directory spec:

- Configuration root: `${XDG_CONFIG_HOME:-~/.config}/rt/`
- Primary config file: `config.json`
- Managed course directory: `${XDG_CONFIG_HOME:-~/.config}/rt/courses/`
- Course files are flattened and stored as `course_<id>.md`

On first launch, rtutor copies the bundled seed courses into the managed course
directory and registers them in `config.json`. You can inspect and edit that
file to point at your own Markdown courses. Each course entry tracks:

- `name`: friendly label shown in menus
- `local_path`: absolute path to the Markdown file
- `xai_file_id`: last uploaded document ID (populated after syncing with Grok)

Global Grok integration settings live under the top-level `xai` key:

```
"xai": {
  "api_key": "...",           # optional, falls back to $XAI_API_KEY
  "management_key": "...",    # optional, falls back to $XAI_MANAGEMENT_API_KEY
  "collection_id": "..."      # filled automatically on first sync
}
```

Environment variables (optional, used as fallbacks if the config fields are empty):

- `XAI_API_KEY` — standard Grok API key for Responses API requests
- `XAI_MANAGEMENT_API_KEY` — Management API key with `AddFileToCollection` permission (uploading documents)

## Project Layout

The runtime Python modules live directly in the repository root (no nested
`modules/` package). Key files:

- `main.py` — CLI entry point
- `menu.py`, `lesson_sequencer.py`, `doc_mode.py`, `rote_mode.py`, `touch_type_mode.py` — interactive UI flows
- `orchestrator.py` — coordinates config loading, course ingestion, and runtime startup
- `config_manager.py`, `course_parser.py`, `bookmarks.py`, `boom.py`, `structs.py` — shared utilities and data structures
- `xai_client.py` — thin HTTP wrappers for Grok Collections and Responses APIs
- `test_config_manager.py` — lightweight regression coverage (pytest)

## Adding Courses

You control the course catalog. Point rtutor at any Markdown lesson file by
registering it:

```
rtutor --add-course "Python Basics" ~/courses/python.md
rtutor --add-course "SQL" ~/docs/sql.md
```

Each file should follow the same structure the app expects:

- Top-level: `# Course`
- Parts: `## Part`
- Optional sections: `### Section`
- Lessons: `#### Lesson` with an indented code block underneath

To remove a course, edit `config.json` directly (located at
`${XDG_CONFIG_HOME:-~/.config}/rt/config.json`).
