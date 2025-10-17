# Unix Tool Mastery

## Part I: Tmux

### Lesson 1: Start Named Session

    tmux new -s dev  # Enter tmux; run commands.

### Lesson 2: Detach Session

    Ctrl-b d  # Exit without killing processes.

### Lesson 3: List Sessions

    tmux ls  # View active sessions.

### Lesson 4: Reattach Session

    tmux attach -t dev  # Jump back in.

### Lesson 5: Split Vertical

    Ctrl-b %  # Side-by-side panes.

### Lesson 6: Split Horizontal

    Ctrl-b "  # Top-bottom panes.

### Lesson 7: Navigate Panes

    Ctrl-b arrow  # Switch panes.

### Lesson 8: Enter Copy Mode

    Ctrl-b [  # Scroll/copy text (arrow/vi keys; q to exit).

### Lesson 9: Close Pane

    Ctrl-b x  # Kill current pane (confirm with y).

## Part II: Grep

### Lesson 1: Basic Search

    grep "kernel" dmesg  # Match lines.

### Lesson 2: Case-Insensitive

    grep -i "error" log.txt  # Ignore case.

### Lesson 3: Recursive

    grep -r "TODO" src/  # Search dirs.

### Lesson 4: Invert Match

    grep -v "^#" config  # Exclude matches.

### Lesson 5: Pipe Usage

    dmesg | grep -i "usb"  # Chain commands.

## Part III: Find

### Lesson 1: By Name

    find /home -name "*.py"  # Match names.

### Lesson 2: By Size/Type

    find . -type f -size +50M  # Files >50MB.

### Lesson 3: Exec on Results

    find /tmp -name "*.tmp" -exec rm {} \;  # Run commands.

### Lesson 4: By Mod Time

    find /backup -mtime -7  # Last 7 days.

### Lesson 5: Pipe to Tools

    find . -name "*.log" | xargs grep "error"  # Combo search.
