# RTutor

AI slop is a problem. Inculcating great programming taste is the antidote.
rtutor is an attempt to disseminate that antidote.

## Table of Contents

- [Preface](#preface)
- [Installation](#installation)
- [Usage](#usage)
- [Creating Course Markdown Files](#creating-course-markdown-files)
  - [File Location and Naming](#file-location-and-naming)
  - [Supported Heading Hierarchies](#supported-heading-hierarchies)
  - [Lesson Code Block Requirements](#lesson-code-block-requirements)
  - [Optional Skip Directives](#optional-skip-directives)
  - [Annotated Examples](#annotated-examples)
    - [Full hierarchy](#full-hierarchy)
    - [Part-only hierarchy](#part-only-hierarchy)
    - [Flat hierarchy](#flat-hierarchy)
  - [Validation and Troubleshooting](#validation-and-troubleshooting)
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

## Installation

Skip the yak shaving. Use the installer script (Linux x86_64):

```
curl -fsSL https://raw.githubusercontent.com/ryangerardwilson/rt/main/install.sh | bash
```

Optional flags:

```
curl -fsSL https://raw.githubusercontent.com/ryangerardwilson/rt/main/install.sh | \
  bash -s -- --version 0.3.0 --no-modify-path
```

The script downloads `rt-linux-x64.tar.gz`, installs to `~/.rt/app/rt`, and
drops a shim in `~/.rt/bin/rt`. It adds that directory to your PATH unless you
pass `--no-modify-path`.

## Usage

You made the symlink — use it.

Start the app (doc mode is the default):
```
rt
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
But symlink to `~/.local/bin/rt` like above.

Markdown format for courses:
```
# Course Name
## Part Name
### Section Name (optional)
#### Lesson Name
    code block here
    multiline ok
```

Need more? See [Creating Course Markdown Files](#creating-course-markdown-
files).

## Creating Course Markdown Files

rtutor only ingests Markdown that follows a strict structure. This section
explains exactly what the parser expects so your custom lessons load without
surprises.

### File Location and Naming

- Store your files wherever you want—`rt` never relocates them. The default config
  points at `${XDG_CONFIG_HOME:-~/.config}/rt/`.
- If you like convention, place courses under `${XDG_CONFIG_HOME:-~/.config}/rt/courses/`
  (create the directory yourself) and name them `course_<slug>.md`. The slug is reused
  inside `config.json`.
- Wherever you save them, record the absolute path in `config.json` (see
  [Configuration](#configuration)).

### Supported Heading Hierarchies

The parser recognises three layouts. Only the first `#` heading is treated as
the course title, so include it once at the top.

1. **Full hierarchy — course → part → section → lesson**
   ```
   # Your Course
   ## Part One
   ### Section A
   #### Lesson Name
       code…
   ```
   Use this when you want both parts _and_ sections.

2. **Part-only hierarchy — course → part → lesson**
   ```
   # Your Course
   ## Part One
   ### Lesson Name
       code…
   ```
   Sections are omitted; the parser injects a default `Main` section for you.

3. **Flat hierarchy — course → lesson**
   ```
   # Your Course
   ## Lesson Name
       code…
   ```
   Both parts and sections collapse into a single implicit `Main` grouping.

Mixing styles in the same file is unsupported. Every lesson must sit under the
appropriate parent heading (e.g., a `#### Lesson` cannot appear before a `###
Section`).

### Lesson Code Block Requirements

- Each lesson needs at least one line indented by **four spaces** or a **single tab**.
- Fenced code blocks (``` ticks) are ignored—indentation is the only signal.
- Keep narrative prose above the lesson heading. Inside the indented block, stick to
  the text you want users to type.
- Blank lines are fine; keep them indented so they remain part of the lesson.
- The parser trims trailing whitespace from each line; align indentation carefully.

### Optional Skip Directives

Lines that start with `#!`, `//!`, or `--!` are marked as **skip lines**. They
still appear in doc mode but are excluded from accuracy scoring in typing
modes. Use them for comments, instructions, or assertions users do not need to
replicate.

Example:

```
#### Echo basics
    #! read the command, then press Enter
    echo "hello"
```

### Annotated Examples

#### Full hierarchy

```
# Unix Fundamentals
## Part I — Shell Basics
### Section 1 — Navigation
#### Lesson: List files
    #! Run this in a sandbox directory
    pwd
    ls -alh

### Section 2 — Editing
#### Lesson: Open file in vim
    vim README.md
```

#### Part-only hierarchy

```
# Python Warmups
## Part 1 — Expressions
### Lesson: Print a string
    print("Hello, taste.")

### Lesson: Arithmetic
    total = 41 + 1
    print(total)
```

#### Flat hierarchy

```
# Git Touch Typing
## Lesson: Clone a repo
    git clone git@github.com:example/project.git

## Lesson: Stage files
    git status
    git add README.md
```

### Validation and Troubleshooting

1. **Preview locally** — run `rt` (doc mode) and use `/` search or bookmarks to
   confirm headings and indentation load correctly.
2. **Register the file** — add an entry to `config.json` with `name` and
   `local_path`. Use absolute paths to avoid surprises.
3. **Upload to Grok** — `rt -t` uploads new or changed files. Confirm status with
   `rt -s`.

Common parser errors:

- `Error: Multiple course names` → only one `#` heading is allowed.
- `Error: Lesson without section` → add a `### Section` before the `#### Lesson`.
- Empty lessons (no indented lines) are dropped silently; make sure every lesson
  contains code.

Run `rt` again after edits; the app reloads courses and normalises the config.

## Doc Mode Features

Doc mode is the default now. Just run `rt` — you'll be in the read-only doc
viewer with full navigation and editing tools.

Quick keys while viewing:
- b — bookmark current lesson
- i — edit the lesson in your $EDITOR (vim by default); rt will reload the course and try to keep your place
- h / l — previous / next lesson
- r — rote mode (10 reps)
- j — jump mode (type through entire sequence)
- / — search within the lesson (vim-style)
- v — visual select; y to copy selection (uses wl-copy if available)
- esc — back to menu

Bookmarks:
- Persisted to `~/.config/rt/bookmarks.conf` in a plain hierarchical format.
- From the main menu in doc mode press `b` to open and jump.
- In the bookmark list: j/k navigate, Enter or l to jump, dd (double-press d) to delete.

In-place editing:
- Press `i` on a lesson (if the source .md file is present).
- `rt` opens your editor at the lesson heading, you edit and save, and `rt` reloads and attempts to keep your location.

## Doc Mode CLI Flags

Short and brutal: doc-mode is default. Launch `rt` with no flags for the
interactive menus and lesson viewer.

```
rt                     # launches doc-mode menus (default)
rt -t                  # upload (train) all registered courses to Grok Collections
rt -s                  # show indexing status for every registered course
rt -p                  # purge all documents from the Grok collection
rt -q "query"          # query the collection
rt -h                  # show global and doc-mode help
rt -v                  # print the installed version
rt -u                  # upgrade to the latest release
```

- `rt -t` → upload (or re-upload) registered courses. Removes stray documents and skips unchanged files.
- `rt -s` → display indexing/processing status for each course file.
- `rt -p` → purge every document currently stored in the Grok collection.
- `rt -q "query"` → non-interactive Q&A powered by the collection (requires API keys).
- `rt -h` → show global usage along with doc-mode options.
- `rt -v` → show the current version.
- `rt -u` → curl the installer and upgrade in-place.

## Configuration

rt stores user configuration under `${XDG_CONFIG_HOME:-~/.config}/rt/`. You
register courses yourself by editing `config.json`. Each course entry tracks:

- `name`: friendly label shown in menus
- `local_path`: absolute path to the Markdown file (`rt` never relocates it)
- `xai_file_id`: last uploaded document ID (populated after running `-t`)

Global Grok integration settings live under the top-level `xai` key:

```
"xai": {
  "api_key": "...",           # optional, falls back to $XAI_API_KEY
  "management_key": "...",    # optional, falls back to $XAI_MANAGEMENT_API_KEY
  "collection_id": "..."      # filled automatically on first sync
}
```

Environment variables (optional, used as fallbacks if the config fields are
empty):

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

You control the course catalog. Point `rt` at any Markdown lesson file by
editing `${XDG_CONFIG_HOME:-~/.config}/rt/config.json` and adding entries under
the `courses` array. Each entry needs a friendly `name` and a `local_path`
pointing to the Markdown file. Once you've added or updated courses, run:

```
rt -t                  # upload (train) all registered courses to Grok Collections
```

For the exact Markdown structure (including skip directives and examples), see
[Creating Course Markdown Files](#creating-course-markdown-files).

To remove a course, edit `config.json` directly (located at
`${XDG_CONFIG_HOME:-~/.config}/rt/config.json`).

After registering courses:

1. Run `rt -t` to upload them to the Grok collection (only changed files are re-uploaded; stale remote files are removed).
2. Use `rt -s` to verify indexing status.
3. Ask query with `rt -q "your query"` once files show `processed`.
