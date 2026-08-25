#!/bin/bash
# Watchdog for live_harbor.py - restarts if dead
# Auto-detects active X session so the GUI window opens from cron too.

SCRIPT="/home/mino/yolo/live_harbor.py"
LOG="/home/mino/yolo/watchdog.log"
USER_NAME="mino"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

find_x_env() {
    # On Wayland (Ubuntu/GDM), the real X cookie lives in /run/user/$UID/.mutter-Xwaylandauth.*
    local uid
    uid=$(id -u "$USER_NAME" 2>/dev/null)
    [ -z "$uid" ] && return 1

    # Prefer the mutter Xwayland cookie file
    local cookie
    cookie=$(ls -t /run/user/"$uid"/.mutter-Xwaylandauth.* 2>/dev/null | head -1)
    if [ -n "$cookie" ] && [ -r "$cookie" ]; then
        XAUTH_VAL="$cookie"
        DISPLAY_VAL=":0"
        return 0
    fi

    # Fallback: scrape DISPLAY/XAUTHORITY from any GUI process owned by user
    local pid
    for pid in $(pgrep -u "$USER_NAME" -f 'gnome-shell|gnome-session|Xwayland|Xorg' 2>/dev/null); do
        local env_file="/proc/$pid/environ"
        [ ! -r "$env_file" ] && continue
        local d a
        d=$(tr '\0' '\n' < "$env_file" | grep -E '^DISPLAY=' | head -1 | cut -d= -f2-)
        a=$(tr '\0' '\n' < "$env_file" | grep -E '^XAUTHORITY=' | head -1 | cut -d= -f2-)
        if [ -n "$d" ] && [ -n "$a" ] && [ -r "$a" ]; then
            DISPLAY_VAL="$d"
            XAUTH_VAL="$a"
            return 0
        fi
    done
    return 1
}

if pgrep -f "python3 $SCRIPT" > /dev/null; then
    echo "$(timestamp) - alive" >> "$LOG"
    exit 0
fi

echo "$(timestamp) - DEAD, restarting..." >> "$LOG"

if find_x_env; then
    echo "$(timestamp) - using DISPLAY=$DISPLAY_VAL XAUTHORITY=$XAUTH_VAL" >> "$LOG"
    export DISPLAY="$DISPLAY_VAL"
    export XAUTHORITY="$XAUTH_VAL"
else
    echo "$(timestamp) - no X session found, running headless" >> "$LOG"
fi

cd /home/$USER_NAME/yolo
nohup python3 "$SCRIPT" >> "$LOG" 2>&1 &

sleep 5
if pgrep -f "python3 $SCRIPT" > /dev/null; then
    echo "$(timestamp) - restarted OK (PID $(pgrep -f "python3 $SCRIPT"))" >> "$LOG"
else
    echo "$(timestamp) - restart FAILED" >> "$LOG"
fi
