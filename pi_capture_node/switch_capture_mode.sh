#!/bin/bash
# BluffSky Capture Engine Switcher (Option A <-> Legacy Allsky Backup)
# Usage:
#   ./switch_capture_mode.sh option-a
#   ./switch_capture_mode.sh legacy-backup
#   ./switch_capture_mode.sh status

set -e

MODE="${1:-status}"

case "$MODE" in
    option-a|bluffsky)
        echo "Switching to BluffSky Option A Science & Stereo Capture Daemon..."
        sudo systemctl stop allsky.service allskyperiodic.service 2>/dev/null || true
        sudo systemctl disable allsky.service allskyperiodic.service 2>/dev/null || true
        sudo systemctl mask allsky.service allskyperiodic.service 2>/dev/null || true
        if [ -f /etc/udev/rules.d/asi.rules ]; then
            sudo sed -i 's/systemctl start allsky/systemctl start bluffsky-node/g; s/systemctl stop allsky/systemctl stop bluffsky-node/g' /etc/udev/rules.d/asi.rules || true
            sudo udevadm control --reload-rules 2>/dev/null || true
        fi
        sudo systemctl unmask bluffsky-node.service 2>/dev/null || true
        sudo systemctl daemon-reload
        sudo systemctl enable bluffsky-node.service
        sudo systemctl restart bluffsky-node.service
        sleep 2
        systemctl status bluffsky-node.service --no-pager | head -n 15
        ;;
    legacy-backup|legacy|allsky)
        echo "Reverting to Legacy Allsky Backup (allsky.service)..."
        sudo systemctl stop bluffsky-node.service 2>/dev/null || true
        sudo systemctl disable bluffsky-node.service 2>/dev/null || true
        if [ -f /etc/udev/rules.d/asi.rules ]; then
            sudo sed -i 's/systemctl start bluffsky-node/systemctl start allsky/g; s/systemctl stop bluffsky-node/systemctl stop allsky/g' /etc/udev/rules.d/asi.rules || true
            sudo udevadm control --reload-rules 2>/dev/null || true
        fi
        sudo systemctl unmask allsky.service allskyperiodic.service 2>/dev/null || true
        sudo systemctl daemon-reload
        sudo systemctl enable allsky.service allskyperiodic.service 2>/dev/null || true
        sudo systemctl restart allsky.service
        sleep 2
        systemctl status allsky.service --no-pager | head -n 15
        ;;
    status|*)
        echo "=== BluffSky Option A (bluffsky-node.service) ==="
        systemctl is-active bluffsky-node.service || true
        echo "=== Legacy Allsky Backup (allsky.service) ==="
        systemctl is-active allsky.service || true
        ;;
esac
