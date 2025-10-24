# Unix

## Part I: Vim CLI Tools

### Lesson 1: Toggle vim normal mode in Alacrity

    # This is a toggle, use the same combo to turn off vim normal mode
    Ctrl+Shift+Space

### Lesson 2: Use vim Buffers

    vim file1 file2 # Open specific files as buffers
    vim dir/* # Open all files as buffers
    vim dir/*.md # Open files of specific extensions as buffers
    vim dir/**/*.md # Same as above, but recursively. The dir should have subdirs

## Part II: Tmux

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

## Part III: Search Utils

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

## Part IV: File & Directory Management

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
    ls -1 # Display has vertical list
    ls -t  # Sort by modification time (newest first)
    ls -S  # Sort by size (largest first)
    ls *.txt  # Wildcard: List only text files
    ls -l | grep "^d"  # Pipe to grep: Show only directories

    # Update
    cp file.txt backup/  
    cp -r dir1/ dir2/  
    mv oldname.txt newname.txt  
    mv file.txt /new/path/  
    chmod 644 script.sh  # Set octal 
    chmod -R 644 dir/  # Recursive apply
    chmod +x script.sh # Make executable
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
    cd  
    cd /path
    cd ../..  # Up two levels
    cd ~  
    cd -  # Back to previous directory

## Part V: Internet

### Lesson 1: Check Interace Name

    ip link show
    # NOTE: Network interface is the logical endpoint where your hardware meets the network stack in the kernel. 
    # It's how the OS refers to your network devices—like eth0 for Ethernet or wlan0 for wireless.

### Lesson 2: Scan for networks

    sudo iw dev wlan0 scan | grep SSID
    # NOTE: SSID stands for Service Set Identifier. It's your WiFi network's human-readable name

### Lesson 3: WPA Supplicant

    # wpa_supplicant is a background process that handles the authentication and encryption for wireless networks. 

    # Add login credentials to `wpa_supplicant.conf`
    vi .wpa_supplicant.conf
    network={
        ssid="YourWiFiName"
        psk="YourPassword"
    }

    # Connect
    sudo wpa_supplicant -B -i wlan0 -c ~/.wpa_supplicant.conf
    sudo wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant/wpa_supplicant.conf

    # Test connection
    sudo dhcpcd wlan0 
    # dhcpd is a lightweight DHCP client daemon (hence the 'd' for daemon). DHCP is Dynamic Host Configuration Protocol, 
    # which automatically grabs an IP address, subnet mask, gateway, DNS servers, and all that crap from your router 
    # so you don't have to configure it statically
    ping 8.8.8.8

    # Disconnect
    sudo killall wpa_supplicant
