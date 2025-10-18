# Unix Tool Mastery

## Part I: Tmux

### Lesson 1: Sessions

    tmux new -s dev # Start
    Ctrl-b d  # Deteach 
    tmux ls  # List 
    tmux attach -t dev  # Jump back in

### Lesson 2: Panes

    Ctrl-b %  # Side-by-side
    Ctrl-b "  # Top-bottom panes.
    Ctrl-b arrow  # Switch panes.
    Ctrl-b x  # Kill current pane (confirm with y).

### Lesson 3:  Copy Mode

    Ctrl-b [  # Enter copy mode 
    arrow/hjkl # Navigate 
    q # exit
    Space # Initiate selection Mode
    ESc # Quit selection mode 

## Part II: Search Utils

### Lesson 1: Grep, Find & Locate

    # Grep searches text within files
    grep "error" system.log
    grep -r "error" logs/ # Search dirs.

    # Find searches file names
    find /project -name "*.py" # Match names/extensions.
    find . -type f -size +50M # Files >50MB.
    find /backup -mtime -7 # Last 7 days.

    # Xargs combo to find strings across multiple files of a specific extension
    find . -type f -name "*.py" | xargs grep "import pandas as pd"

    # Locate: While find seaches below ~/, locate seaches above 
    sudo updatedb && locate "nginx"

### Lesson 2: Diff and Git Diff

    # Diff compares two files line by line
    diff file1.txt file2.txt  
    diff -u file1.txt file2.txt  # Unified format, easier to read 
    diff -r dir1/ dir2/  # Recursively compare directories.

    # Git diff: 
    git diff  # Shows changes between working dir and index.
    git diff --cached  # Changes staged for commit.
    git diff HEAD  # All changes since last commit.
    git diff branch1 branch2  # Compare two branches.

## Part III: File & Directory Management

### Lesson 1: File & Dir CRUD

    # Create
    touch newfile.txt  
    mkdir newdir  
    mkdir -m 755 secure_dir  

    # Read
    ls
    ls -alh # All details
    ls -lh # Exclude hidden files
    ls -Rlh # Recursive: List subdirectories too
    ls -t  # Sort by modification time (newest first)
    ls -S  # Sort by size (largest first)
    ls *.txt  # Wildcard: List only text files
    ls -l | grep "^d"  # Pipe to grep: Show only directories

    # Update
    cp file.txt backup/  # Copy file to dir
    cp -r dir1/ dir2/  # Recursive copy for directories
    mv oldname.txt newname.txt  
    mv file.txt /new/path/  # Move to different dir
    chmod +x # Make executable
    chmod 644 script.sh  # Set octal 
    chmod -R 644 dir/  # Recursive apply
    # 600 for private files; 644 for standard files; 700 for private dirs; 755 for letting others peek/run without editing; 777 for all access
    # In unix systems, inside the /home dir, for newly created:
    # - files: 644 is default because it strikes a balance between you owning the file, the others accessing it without editing.
    # - dirs: 755 is default because it allows others to cd into your dirs as the cd command needs executable permission

    # Delete
    rm file.txt  
    rm *.py
    rm -rf dir/  

### Lesson 2: Navigating with cd and pwd

    pwd  
    cd  # Straight home
    cd /path
    cd ../..  # Up two levels
    cd ~  # Home 
    cd -  # Back to previous directory
