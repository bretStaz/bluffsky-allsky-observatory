#!/usr/bin/env python3
r"""
BluffSky Allsky Workflow Health Guard & Self-Healing Automation
BluffSky • E:\allsky\anomaly_detector\allsky_workflow_guard.py

Continuously ensures the entire Allsky workflow is healthy, diagnoses root causes
for any failure or degradation, automatically fixes issues, and reports status
to Discord (#allsky).

Components Monitored & Self-Healed:
1. BluffSky Raspberry Pi 5 Camera (192.168.1.100):
   - ICMP ping reachability
   - HTTP 80 web server & status.json endpoint
   - Live image freshness (stale frame detection via Last-Modified / ETag)
   - Capture aborts & exposure parameters (overlaydebug.txt)
2. Windows Services:
   - AllSkyMonitor (AllSky Image Monitor Service -> F:\AllSky_Archive)
   - AllSkyTimelapseService (AllSky Timelapse Service)
3. Real-Time Sky Anomaly Detector Daemon (detector.py --daemon):
   - Process running status
   - Hung / deadlocked process detection (checks live_status.json update age)
   - Auto-kill & restart if hung or crashed
4. Allsky Web Portal (web_portal.py --port 8094):
   - HTTP reachability & stats API health
   - Auto-restart if offline
5. Local Vision LLM Server (LM Studio on port 1234):
   - Checks models endpoint, auto-starts LMS and loads Gemma 4 Vision if offline
6. Discord Alert Hub (http://localhost:8090/health)
7. Storage Capacity:
   - Verifies disk space on E: and F: (> 5 GB free)
"""

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
LIVE_STATUS_FILE = os.path.join(BASE_DIR, "live_status.json")
GUARD_STATUS_FILE = os.path.join(BASE_DIR, "workflow_guard_status.json")
LOG_FILE = os.path.join(BASE_DIR, "workflow_guard.log")
DISCORD_SERVER_DIR = r"E:\discord_messages"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Setup Guard Logging
logger = logging.getLogger("AllskyWorkflowGuard")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

# Cooldown tracking for Discord alerts to prevent alert storms
COOLDOWN_FILE = os.path.join(BASE_DIR, "guard_alert_cooldowns.json")
_LAST_ALERT_TIMESTAMPS: Dict[str, float] = {}
ALERT_COOLDOWN_SEC = 900  # 15 minutes between identical warnings


def _load_cooldowns() -> Dict[str, float]:
    if os.path.exists(COOLDOWN_FILE):
        try:
            with open(COOLDOWN_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cooldowns(data: Dict[str, float]):
    try:
        with open(COOLDOWN_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading {CONFIG_FILE}: {e}")
    return {}


def send_discord_guard_alert(
    title: str,
    message: str,
    level: str = "warning",
    cooldown_key: Optional[str] = None,
    force: bool = False,
) -> bool:
    """Dispatches a diagnostic / self-healing alert to Discord #allsky."""
    now = time.time()
    cooldowns = _load_cooldowns()
    if cooldown_key and not force:
        last = cooldowns.get(cooldown_key, _LAST_ALERT_TIMESTAMPS.get(cooldown_key, 0.0))
        if now - last < ALERT_COOLDOWN_SEC:
            logger.info(f"Skipping Discord alert for '{cooldown_key}' (cooldown active: {int(now - last)}s < {ALERT_COOLDOWN_SEC}s)")
            return False

    colors = {
        "info": 0x38BDF8,
        "success": 0x34D399,
        "warning": 0xFBBF24,
        "critical": 0xEF4444,
        "error": 0xEF4444,
    }
    color = colors.get(level.lower(), 0x38BDF8)
    payload = {
        "channel": "allsky",
        "title": title,
        "message": message,
        "color": color,
    }

    # 1. Try in-process direct dispatch via discord_server.py
    try:
        if os.path.isdir(DISCORD_SERVER_DIR):
            if DISCORD_SERVER_DIR not in sys.path:
                sys.path.insert(0, DISCORD_SERVER_DIR)
            from discord_server import send_to_discord

            ok, _code, _info = send_to_discord(payload)
            if ok:
                if cooldown_key:
                    _LAST_ALERT_TIMESTAMPS[cooldown_key] = now
                    cooldowns[cooldown_key] = now
                    _save_cooldowns(cooldowns)
                logger.info(f"Sent Discord guard alert: {title}")
                return True
    except Exception as e:
        logger.debug(f"Direct discord_server dispatch failed: {e}")

    # 2. Fallback to HTTP POST to local discord bridge
    try:
        resp = requests.post("http://localhost:8090/alert", json=payload, timeout=5)
        if 200 <= resp.status_code < 300:
            if cooldown_key:
                _LAST_ALERT_TIMESTAMPS[cooldown_key] = now
                cooldowns[cooldown_key] = now
                _save_cooldowns(cooldowns)
            logger.info(f"Sent Discord guard alert via HTTP: {title}")
            return True
    except Exception as e:
        logger.warning(f"Could not post guard alert to Discord bridge: {e}")

    return False


# ---------------------------------------------------------------------------
# Diagnostics & Self-Healing: Pi 5 Camera
# ---------------------------------------------------------------------------
def check_camera_pi(cfg: dict, max_retries: int = 3) -> dict:
    """
    Checks a BluffSky Raspberry Pi 5 Allsky camera node (1..N supported).
    Diagnoses connectivity, web server, camera software status, and frame freshness.
    Performs robust retries on ICMP and HTTP to avoid false positives over Wi-Fi.
    """
    from urllib.parse import urlparse as _urlparse

    cam_id = str(cfg.get("camera_id") or "cam1")
    base_url = cfg.get("allsky_base_url", "http://192.168.1.100").rstrip("/")
    parsed_host = _urlparse(base_url).hostname or "192.168.1.100"
    cam_name = str(cfg.get("camera_name") or f"BluffSky Camera {cam_id} ({parsed_host})")
    live_img_url = cfg.get("live_image_url", f"{base_url}/current/tmp/image.jpg")
    status_url = f"{base_url}/config/status.json"
    overlay_url = cfg.get("overlay_debug_url", f"{base_url}/current/tmp/overlaydebug.txt")

    diag = {
        "camera_id": cam_id,
        "name": cam_name,
        "ip": parsed_host,
        "base_url": base_url,
        "ping_ok": False,
        "http_ok": False,
        "status_str": "unknown",
        "frame_fresh": False,
        "frame_age_seconds": None,
        "exposure_str": "unknown",
        "temp_f": "unknown",
        "healthy": False,
        "diagnosis": None,
        "remedy": None,
    }

    # 1. ICMP Ping check with retries
    ping_ok = False
    for attempt in range(max_retries):
        try:
            res = subprocess.run(
                ["ping", "-n", "3", "-w", "1500", parsed_host],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=6,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if res.returncode == 0 or "TTL=" in res.stdout or "ttl=" in res.stdout:
                ping_ok = True
                break
        except Exception:
            pass
        if attempt < max_retries - 1:
            time.sleep(1.0)
    diag["ping_ok"] = ping_ok

    # 2. Query Pi status.json and live image with retries
    http_ok = False
    r_stat = None
    r_img = None
    for attempt in range(max_retries):
        if not r_stat or r_stat.status_code != 200:
            try:
                r_stat = requests.get(status_url, timeout=4)
                if r_stat.status_code == 200:
                    stat_data = r_stat.json()
                    diag["status_str"] = stat_data.get("status", "Unknown")
                    http_ok = True
            except Exception as e:
                diag["status_str"] = f"Unreachable ({e})"

        if not r_img or r_img.status_code != 200:
            try:
                r_img = requests.head(live_img_url, timeout=4)
                if r_img.status_code == 200:
                    http_ok = True
            except Exception:
                pass

        if http_ok:
            break
        if attempt < max_retries - 1:
            time.sleep(1.5)

    diag["http_ok"] = http_ok

    # If both ping and HTTP completely failed after all retries
    if not ping_ok and not http_ok:
        diag["diagnosis"] = (
            f"{cam_name} ({parsed_host}) is completely unreachable via ICMP and HTTP "
            f"after {max_retries} robust retry attempts. "
            "Likely causes: Pi is powered off, Wi-Fi is disconnected, or device has crashed."
        )
        diag["remedy"] = f"Check physical power and 5GHz Wi-Fi connection to {cam_name} ({parsed_host})."
        return diag

    # If ICMP dropped over Wi-Fi but HTTP responded, camera is verified reachable
    if not ping_ok and http_ok:
        diag["ping_ok"] = True

    # 3. Test frame freshness from image headers
    if r_img and r_img.status_code == 200:
        lm_hdr = r_img.headers.get("Last-Modified")
        if lm_hdr:
            try:
                dt_lm = parsedate_to_datetime(lm_hdr)
                now_utc = datetime.now(timezone.utc)
                age_sec = max(0.0, (now_utc - dt_lm).total_seconds())
                diag["frame_age_seconds"] = round(age_sec, 1)
                # Normal night exposure is 5-30s, day is <1s. Stale threshold is 300s (5 mins).
                diag["frame_fresh"] = age_sec <= 300.0
            except Exception:
                diag["frame_fresh"] = True
        else:
            diag["frame_fresh"] = True
    else:
        try:
            r_img2 = requests.head(live_img_url, timeout=4)
            if r_img2.status_code == 200:
                diag["http_ok"] = True
                diag["frame_fresh"] = True
            else:
                diag["http_ok"] = False
                diag["diagnosis"] = f"Pi 5 web server returned HTTP {r_img2.status_code} for live image."
                return diag
        except Exception as e:
            diag["http_ok"] = False
            diag["diagnosis"] = f"Cannot reach Pi 5 HTTP service on {live_img_url}: {e}"
            diag["remedy"] = "Verify lighttpd and allsky service status on the Pi 5."
            return diag

    # 4. Read overlay telemetry
    try:
        r_ov = requests.get(overlay_url, timeout=3)
        if r_ov.status_code == 200:
            for line in r_ov.text.splitlines():
                if line.startswith("AS_sEXPOSURE"):
                    diag["exposure_str"] = line.split(maxsplit=1)[-1].strip()
                elif line.startswith("AS_TEMPERATURE_F"):
                    diag["temp_f"] = line.split(maxsplit=1)[-1].strip()
    except Exception:
        pass

    # Evaluate Overall Health
    if diag["http_ok"]:
        if diag["status_str"].lower() not in ("running", "unknown"):
            diag["healthy"] = False
            diag["diagnosis"] = f"Pi 5 Allsky software reported non-running status: '{diag['status_str']}'."
            diag["remedy"] = f"Inspect Pi 5 Allsky status: {diag['status_str']}."
        elif not diag["frame_fresh"] and diag["frame_age_seconds"] is not None:
            diag["healthy"] = False
            diag["diagnosis"] = (
                f"Camera capture appears frozen/stalled: 'image.jpg' was last updated {int(diag['frame_age_seconds'])} seconds ago "
                f"(exposure setting: {diag['exposure_str']}). Normal update interval is < 60 seconds."
            )
            diag["remedy"] = "Pi 5 camera capture loop needs restart or camera USB re-enumeration."
        else:
            diag["healthy"] = True
            diag["diagnosis"] = f"Pi 5 camera is active and capturing smoothly (exposure: {diag['exposure_str']}, sensor temp: {diag['temp_f']}°F)."
    else:
        diag["healthy"] = False

    return diag


# ---------------------------------------------------------------------------
# Diagnostics & Self-Healing: Windows Services (AllSkyMonitor & AllSkyTimelapse)
# ---------------------------------------------------------------------------
def check_windows_services() -> Tuple[List[dict], List[str]]:
    r"""
    Checks Windows services for Allsky:
    - AllSkyMonitor: Archives live image.jpg from F:\AllSky\Archive into F:\AllSky_Archive\<Y>\<M>\<D>\
    - AllSkyTimelapseService: Generates timelapses in F:\Programming\TimelapseCreator
    Auto-restarts any stopped service.
    """
    services_to_check = [
        {"name": "AllSkyMonitor", "display": "AllSky Image Monitor Service", "desc": "Live frame archiver to F:\\AllSky_Archive"},
        {"name": "AllSkyTimelapseService", "display": "AllSky Timelapse Service", "desc": "Automated video timelapse generator"},
    ]
    results = []
    actions_taken = []

    for s_info in services_to_check:
        sname = s_info["name"]
        item = {
            "name": sname,
            "display": s_info["display"],
            "status": "Unknown",
            "healthy": False,
            "diagnosis": None,
            "action": None,
        }

        # Check status using win32serviceutil
        current_status = "Unknown"
        try:
            import win32serviceutil

            code = win32serviceutil.QueryServiceStatus(sname)[1]
            if code == 4:
                current_status = "Running"
            elif code == 1:
                current_status = "Stopped"
            elif code == 2:
                current_status = "StartPending"
            elif code == 3:
                current_status = "StopPending"
            else:
                current_status = f"State({code})"
        except Exception:
            # Fallback to PowerShell Get-Service
            try:
                ps_cmd = f"(Get-Service -Name '{sname}' -ErrorAction SilentlyContinue).Status.ToString()"
                out = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps_cmd], text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).strip()
                current_status = out or "NotFound"
            except Exception as e:
                current_status = f"Error: {e}"

        item["status"] = current_status
        if current_status == "Running":
            item["healthy"] = True
            item["diagnosis"] = f"Service '{sname}' is Running normally."
        else:
            item["healthy"] = False
            item["diagnosis"] = f"Service '{sname}' is {current_status} (expected Running). {s_info['desc']} is offline."
            logger.warning(f"Diagnosed issue: {item['diagnosis']}")

            # Self-heal: Start the service
            logger.info(f"Self-Healing: Starting Windows Service '{sname}'...")
            started_ok = False
            try:
                import win32serviceutil

                win32serviceutil.StartService(sname)
                time.sleep(2.0)
                new_code = win32serviceutil.QueryServiceStatus(sname)[1]
                started_ok = new_code == 4
            except Exception as e:
                # Fallback to PowerShell Start-Service
                try:
                    subprocess.run(["powershell", "-NoProfile", "-Command", f"Start-Service -Name '{sname}'"], timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    time.sleep(2.0)
                    ps_cmd = f"(Get-Service -Name '{sname}').Status.ToString()"
                    out = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps_cmd], text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).strip()
                    started_ok = out == "Running"
                except Exception as ex:
                    logger.error(f"Failed starting service '{sname}': {ex}")

            if started_ok:
                msg = f"Restarted Windows Service '{sname}' ({s_info['display']}) successfully."
                item["action"] = msg
                item["status"] = "Running"
                item["healthy"] = True
                actions_taken.append(msg)
                logger.info(f"Self-healed: {msg}")
            else:
                item["action"] = f"Attempted restart of '{sname}', but service failed to return to Running."
                logger.error(item["action"])

        results.append(item)

    return results, actions_taken


# ---------------------------------------------------------------------------
# Diagnostics & Self-Healing: Anomaly Detector Daemon (detector.py --daemon)
# ---------------------------------------------------------------------------
def check_anomaly_detector_daemon(cfg: dict, camera_online: bool) -> Tuple[dict, Optional[str]]:
    """
    Checks the real-time AI anomaly & meteor detector daemon.
    Diagnoses:
    1. Is process running?
    2. Is process hung / deadlocked? (Checks live_status.json update timestamp).
    Self-heals by terminating hung process if needed and respawning daemon.
    """
    daemon_pids = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or []).lower()
            if "detector.py" in cmd and "--daemon" in cmd and "allsky" in cmd:
                daemon_pids.append(proc.info["pid"])
        except Exception:
            pass

    diag = {
        "name": "Allsky AI Anomaly & Meteor Detector Daemon",
        "running": len(daemon_pids) > 0,
        "pids": daemon_pids,
        "healthy": False,
        "hung": False,
        "last_frame_processed": "unknown",
        "last_update_age_seconds": None,
        "diagnosis": None,
        "action": None,
    }

    # Inspect live_status.json
    now = time.time()
    if os.path.exists(LIVE_STATUS_FILE):
        try:
            with open(LIVE_STATUS_FILE, "r", encoding="utf-8") as f:
                ls = json.load(f)
                up_epoch = ls.get("updated_at")
                if up_epoch:
                    age = max(0.0, now - float(up_epoch))
                    diag["last_update_age_seconds"] = round(age, 1)
                diag["last_frame_processed"] = ls.get("last_frame", "none")
        except Exception:
            pass

    action_taken = None

    # Detect hung condition: Process exists, camera is online, but no update for > 600 seconds (10 mins)
    if daemon_pids and camera_online and diag["last_update_age_seconds"] is not None:
        if diag["last_update_age_seconds"] > 600.0:
            diag["hung"] = True
            diag["diagnosis"] = (
                f"Detector daemon (PIDs {daemon_pids}) is hung/deadlocked: no frames have been processed in "
                f"{int(diag['last_update_age_seconds'])} seconds while live camera is actively producing frames."
            )
            logger.warning(diag["diagnosis"])
            # Terminate hung processes
            for pid in daemon_pids:
                try:
                    p = psutil.Process(pid)
                    p.kill()
                    logger.info(f"Killed hung detector daemon PID {pid}.")
                except Exception as e:
                    logger.warning(f"Could not kill PID {pid}: {e}")
            daemon_pids = []
            diag["running"] = False

    if not daemon_pids:
        if not diag["diagnosis"]:
            diag["diagnosis"] = "Allsky AI Anomaly Detector daemon is not running (stopped or crashed)."
        logger.warning(f"Self-Healing: Spawning fresh detector daemon...")

        py_exe = sys.executable
        det_script = os.path.join(BASE_DIR, "detector.py")
        try:
            # Spawn detached process
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            subprocess.Popen(
                [py_exe, det_script, "--daemon"],
                cwd=BASE_DIR,
                creationflags=creationflags,
                close_fds=True,
            )
            time.sleep(2.5)
            # Re-verify
            new_pids = []
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmd = " ".join(proc.info["cmdline"] or []).lower()
                    if "detector.py" in cmd and "--daemon" in cmd and "allsky" in cmd:
                        new_pids.append(proc.info["pid"])
                except Exception:
                    pass

            if new_pids:
                diag["healthy"] = True
                diag["running"] = True
                diag["pids"] = new_pids
                action_taken = f"Spawned fresh Allsky AI Anomaly Detector daemon (PID {new_pids[0]})."
                diag["action"] = action_taken
                logger.info(f"Self-healed: {action_taken}")
            else:
                diag["action"] = "Spawned detector daemon, but process did not register PID."
                logger.error(diag["action"])
        except Exception as e:
            diag["action"] = f"Failed to spawn detector daemon: {e}"
            logger.error(diag["action"])
    else:
        diag["healthy"] = True
        diag["diagnosis"] = f"Detector daemon is healthy (PID {daemon_pids[0]}, processed '{diag['last_frame_processed']}' {diag['last_update_age_seconds']}s ago)."

    return diag, action_taken


# ---------------------------------------------------------------------------
# Diagnostics & Self-Healing: Web Portal (web_portal.py on port 8094)
# ---------------------------------------------------------------------------
def check_web_portal(cfg: dict) -> Tuple[dict, Optional[str]]:
    """
    Checks the Allsky Anomaly Web Portal (port 8094).
    Diagnoses responsiveness and restarts if down, ensuring only a single instance runs.
    """
    port = int(cfg.get("web_portal_port", 8094))
    url = f"http://localhost:{port}/api/version"
    diag = {
        "name": f"Allsky Anomaly Web Portal (Port {port})",
        "url": f"http://localhost:{port}/",
        "healthy": False,
        "diagnosis": None,
        "action": None,
    }

    action_taken = None
    portal_ok = False
    for _ in range(2):
        try:
            r = requests.get(url, timeout=6)
            if r.status_code == 200:
                portal_ok = True
                diag["healthy"] = True
                diag["diagnosis"] = f"Web portal is responding normally on http://localhost:{port}/."
                break
        except Exception as e:
            diag["diagnosis"] = f"Web portal on port {port} is unresponsive ({e})."
            time.sleep(1.0)

    if not portal_ok:
        logger.warning(f"Diagnosed issue: {diag['diagnosis']}. Self-healing...")
        # Terminate any hung web_portal.py processes before spawning a single clean instance
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                if (
                    len(cmdline) >= 2
                    and os.path.basename(cmdline[1]).lower() == "web_portal.py"
                    and "allsky" in cmdline[1].lower()
                ):
                    proc.kill()
            except Exception:
                pass
        time.sleep(1.0)
        py_exe = sys.executable
        portal_script = os.path.join(BASE_DIR, "web_portal.py")
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            subprocess.Popen(
                [py_exe, portal_script, "--port", str(port)],
                cwd=BASE_DIR,
                creationflags=creationflags,
                close_fds=True,
            )
            time.sleep(2.5)
            # Re-verify
            r = requests.get(url, timeout=6)
            if r.status_code == 200:
                diag["healthy"] = True
                action_taken = f"Restarted Allsky Web Portal on http://localhost:{port}/."
                diag["action"] = action_taken
                logger.info(f"Self-healed: {action_taken}")
            else:
                diag["action"] = f"Launched web portal, but /api/version returned HTTP {r.status_code}."
        except Exception as e:
            diag["action"] = f"Failed to restart web portal: {e}"
            logger.error(diag["action"])

    return diag, action_taken


# ---------------------------------------------------------------------------
# Diagnostics & Self-Healing: LM Studio Local VLM Server (Port 1234)
# ---------------------------------------------------------------------------
def check_vlm_server(cfg: dict) -> Tuple[dict, Optional[str]]:
    """Checks LM Studio local VLM server; starts server and loads Gemma 4 Vision if offline."""
    api_url = cfg.get("vlm_api_url", "http://localhost:1234/v1/chat/completions")
    models_url = api_url.replace("/chat/completions", "/models")
    preferred = cfg.get("vlm_preferred_model", "qwen3.5-122b-a10b")

    diag = {
        "name": "Local VLM (LM Studio - Qwen 3.5 122B Vision on RTX 8000)",
        "models_url": models_url,
        "healthy": False,
        "loaded_model": None,
        "diagnosis": None,
        "action": None,
    }
    action_taken = None

    try:
        r = requests.get(models_url, timeout=2.5)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                diag["healthy"] = True
                diag["loaded_model"] = data[0].get("id")
                diag["diagnosis"] = f"VLM server is online with model '{diag['loaded_model']}'."
                return diag, None
    except Exception:
        pass

    # Server down or no model loaded
    diag["diagnosis"] = "Local VLM server (port 1234) is offline or has no vision model loaded."
    logger.warning(f"Diagnosed issue: {diag['diagnosis']}")

    lms_exe = os.path.expanduser(r"~\.lmstudio\bin\lms.exe")
    if os.path.exists(lms_exe):
        logger.info(f"Self-Healing: Starting LM Studio server and loading '{preferred}'...")
        try:
            subprocess.run([lms_exe, "server", "start"], timeout=12, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            subprocess.run([lms_exe, "load", preferred, "--gpu", "max", "-c", "32768", "--parallel", "2", "-y"], timeout=120, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            r = requests.get(models_url, timeout=4)
            if r.status_code == 200:
                diag["healthy"] = True
                action_taken = f"Started LM Studio server and loaded vision model '{preferred}' on RTX 8000."
                diag["action"] = action_taken
                logger.info(f"Self-healed: {action_taken}")
            else:
                diag["action"] = "Invoked LMS server start, but models endpoint still unready."
        except Exception as e:
            diag["action"] = f"Failed to start LMS: {e}"
            logger.error(diag["action"])
    else:
        diag["action"] = f"lms.exe not found at {lms_exe}."

    return diag, action_taken


# ---------------------------------------------------------------------------
# Diagnostics & Self-Healing: IoT Portal (port 8095) & Discord Agent Daemon
# ---------------------------------------------------------------------------
def check_iot_portal() -> Tuple[dict, Optional[str]]:
    """Checks Home Wi-Fi & IoT Portal (port 8095) and auto-restarts if offline."""
    port = 8095
    url = f"http://localhost:{port}/"
    diag = {
        "name": f"Home IoT & Sense Portal (Port {port})",
        "url": url,
        "healthy": False,
        "diagnosis": None,
        "action": None,
    }
    action_taken = None
    for _ in range(2):
        try:
            r = requests.get(url, timeout=4)
            r_api = requests.get(f"http://localhost:{port}/api/state", timeout=6)
            if r.status_code == 200 and r_api.status_code == 200:
                diag["healthy"] = True
                diag["diagnosis"] = f"IoT Portal is responding normally on {url}."
                return diag, None
            diag["diagnosis"] = f"IoT Portal returned HTTP {r.status_code} (/api/state HTTP {r_api.status_code})."
        except Exception as e:
            diag["diagnosis"] = f"IoT Portal on port {port} is unresponsive ({e})."
            time.sleep(1.0)

    logger.warning(f"Diagnosed issue: {diag['diagnosis']}. Self-healing...")
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            if "iot_portal_server.py" in cmdline:
                proc.kill()
        except Exception:
            pass
    time.sleep(1.0)

    iot_dir = r"E:\iot_portal"
    iot_script = os.path.join(iot_dir, "iot_portal_server.py")
    pyw_exe = sys.executable
    if not os.path.exists(pyw_exe):
        pyw_exe = sys.executable

    if os.path.exists(iot_script):
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            subprocess.Popen(
                [pyw_exe, iot_script],
                cwd=iot_dir,
                creationflags=creationflags,
                close_fds=True,
            )
            time.sleep(3.0)
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                diag["healthy"] = True
                action_taken = f"Restarted Home IoT & Sense Portal on {url}."
                diag["action"] = action_taken
                logger.info(f"Self-healed: {action_taken}")
        except Exception as e:
            diag["action"] = f"Failed to restart IoT Portal: {e}"
            logger.error(diag["action"])

    return diag, action_taken


def check_discord_agent_daemon() -> Tuple[dict, Optional[str]]:
    """Checks the Discord AI Command Center daemon (E:\\Agent\\discord_agent.py --daemon) and self-heals."""
    agent_script = r"E:\Agent\discord_agent.py"
    pids = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or []).lower()
            if "discord_agent.py" in cmd and "--daemon" in cmd:
                pids.append(proc.info["pid"])
        except Exception:
            pass

    diag = {
        "name": "Discord AI Command Center Daemon (#ask-agent)",
        "running": len(pids) > 0,
        "pids": pids,
        "healthy": len(pids) > 0,
        "diagnosis": f"Discord Agent daemon is running (PID {pids[0]})." if pids else "Discord Agent daemon is not running.",
        "action": None,
    }
    action_taken = None
    if not pids and os.path.exists(agent_script):
        logger.warning("Self-Healing: Spawning Discord AI Command Center daemon...")
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            subprocess.Popen(
                [sys.executable, agent_script, "--daemon"],
                cwd=r"E:\Agent",
                creationflags=creationflags,
                close_fds=True,
            )
            time.sleep(2.5)
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmd = " ".join(proc.info.get("cmdline") or []).lower()
                    if "discord_agent.py" in cmd and "--daemon" in cmd:
                        pids.append(proc.info["pid"])
                except Exception:
                    pass
            if pids:
                diag["running"] = True
                diag["pids"] = pids
                diag["healthy"] = True
                action_taken = f"Started Discord AI Command Center daemon (PID {pids[0]}) and caught up on #ask-agent."
                diag["action"] = action_taken
                logger.info(f"Self-healed: {action_taken}")
        except Exception as e:
            diag["action"] = f"Failed to start Discord Agent daemon: {e}"
            logger.error(diag["action"])

    return diag, action_taken


# ---------------------------------------------------------------------------
# Diagnostics: Storage Free Space (E: and F:)
# ---------------------------------------------------------------------------
def check_storage() -> dict:
    """Verifies available storage on E: (models/code) and F: (raw Allsky archive/timelapses)."""
    drives = {}
    healthy = True
    diag_msgs = []

    for d in ["E:\\", "F:\\"]:
        try:
            usage = shutil.disk_usage(d)
            free_gb = usage.free / (1024**3)
            total_gb = usage.total / (1024**3)
            drives[d] = {
                "free_gb": round(free_gb, 1),
                "total_gb": round(total_gb, 1),
                "percent_free": round((usage.free / usage.total) * 100, 1),
            }
            if free_gb < 5.0:
                healthy = False
                diag_msgs.append(f"Drive {d} has critically low disk space: {free_gb:.1f} GB remaining.")
        except Exception:
            pass

    return {
        "healthy": healthy,
        "drives": drives,
        "diagnosis": "Disk capacity healthy (> 5 GB free on all archive drives)." if healthy else " ".join(diag_msgs),
    }


# ---------------------------------------------------------------------------
# Main Workflow Health Evaluation & Self-Healing Cycle
# ---------------------------------------------------------------------------
def run_guard_cycle(trigger_discord_alerts: bool = True) -> dict:
    """
    Executes a complete health assessment of the Allsky workflow,
    diagnoses any issues, executes self-healing procedures, and records telemetry.
    """
    start_t = time.time()
    cfg = load_config()

    report = {
        "timestamp": datetime.now().isoformat(),
        "overall_healthy": True,
        "camera_pi": {},
        "camera_nodes": [],
        "windows_services": [],
        "detector_daemon": {},
        "web_portal": {},
        "vlm_server": {},
        "storage": {},
        "healed_actions": [],
        "unresolved_issues": [],
        "duration_sec": 0.0,
    }

    # 1. Check 1..N Pi 5 Camera Nodes
    cam_diags: List[dict] = []
    try:
        import detector
        enabled_cams = detector.get_configured_cameras(cfg, enabled_only=True)
        for cam_entry in enabled_cams:
            cam_cfg = detector.get_effective_camera_cfg(cfg, cam_entry)
            cd = check_camera_pi(cam_cfg)
            cam_diags.append(cd)
    except Exception:
        cam_diags = [check_camera_pi(cfg)]

    if not cam_diags:
        cam_diags = [check_camera_pi(cfg)]

    cam_diag = cam_diags[0]
    report["camera_pi"] = cam_diag
    report["camera_nodes"] = cam_diags
    for cd in cam_diags:
        if not cd["healthy"]:
            report["overall_healthy"] = False
            report["unresolved_issues"].append(f"Camera {cd.get('camera_id', 'cam1')} ({cd.get('ip')}): {cd.get('diagnosis')}")

    # 2. Check Windows Services (AllSkyMonitor & AllSkyTimelapseService)
    svc_res, svc_actions = check_windows_services()
    report["windows_services"] = svc_res
    report["healed_actions"].extend(svc_actions)
    for s in svc_res:
        if not s["healthy"]:
            report["overall_healthy"] = False
            report["unresolved_issues"].append(f"Windows Service: {s['name']} is {s['status']}.")

    # 3. Check Anomaly Detector Daemon
    camera_capturing = any(cd.get("healthy", False) and cd.get("frame_fresh", False) for cd in cam_diags)
    det_diag, det_action = check_anomaly_detector_daemon(cfg, camera_online=camera_capturing)
    report["detector_daemon"] = det_diag
    if det_action:
        report["healed_actions"].append(det_action)
    if not det_diag["healthy"]:
        report["overall_healthy"] = False
        report["unresolved_issues"].append(f"Detector Daemon: {det_diag.get('diagnosis')}")

    # 4. Check Web Portal on 8094
    wp_diag, wp_action = check_web_portal(cfg)
    report["web_portal"] = wp_diag
    if wp_action:
        report["healed_actions"].append(wp_action)
    if not wp_diag["healthy"]:
        report["overall_healthy"] = False
        report["unresolved_issues"].append(f"Web Portal: {wp_diag.get('diagnosis')}")

    # 5. Check Local VLM
    vlm_diag, vlm_action = check_vlm_server(cfg)
    report["vlm_server"] = vlm_diag
    if vlm_action:
        report["healed_actions"].append(vlm_action)

    # 6. Check IoT & Sense Portal (port 8095) and Discord Agent Daemon
    iot_diag, iot_action = check_iot_portal()
    report["iot_portal"] = iot_diag
    if iot_action:
        report["healed_actions"].append(iot_action)

    da_diag, da_action = check_discord_agent_daemon()
    report["discord_agent"] = da_diag
    if da_action:
        report["healed_actions"].append(da_action)

    # 7. Check Storage
    stor_diag = check_storage()
    report["storage"] = stor_diag
    if not stor_diag["healthy"]:
        report["overall_healthy"] = False
        report["unresolved_issues"].append(stor_diag["diagnosis"])

    report["duration_sec"] = round(time.time() - start_t, 2)

    # Save snapshot to workflow_guard_status.json
    try:
        with open(GUARD_STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write {GUARD_STATUS_FILE}: {e}")

    # Dispatch Discord alerts if self-healing occurred or unresolved issues exist
    if trigger_discord_alerts:
        if report["healed_actions"]:
            actions_list = "\n".join(f"• {a}" for a in report["healed_actions"])
            send_discord_guard_alert(
                title="🛡️ Allsky Workflow Guard: Self-Healing Performed",
                message=(
                    f"The automated Allsky health guard diagnosed an issue and executed self-healing:\n\n"
                    f"{actions_list}\n\n"
                    f"**Current Status:** All healed components have been verified and restored."
                ),
                level="success",
                cooldown_key="allsky_guard_healed",
                force=True,
            )

        if report["unresolved_issues"]:
            issues_list = "\n".join(f"• {issue}" for issue in report["unresolved_issues"])
            send_discord_guard_alert(
                title="⚠️ Allsky Workflow Issue Diagnosed",
                message=(
                    f"The automated Allsky health guard diagnosed issues requiring attention:\n\n"
                    f"{issues_list}\n\n"
                    f"**Automated Guard Telemetry:** Pi 5 Ping: `{cam_diag['ping_ok']}`, "
                    f"HTTP: `{cam_diag['http_ok']}`, Detector: `{det_diag['running']}`, Web: `{wp_diag['healthy']}`."
                ),
                level="warning",
                cooldown_key="allsky_guard_unresolved",
            )

    return report


# ---------------------------------------------------------------------------
# Windows Task Scheduler Management
# ---------------------------------------------------------------------------
TASK_NAME = "AllSkyWorkflowGuard"


def install_scheduled_task() -> bool:
    """Installs a recurring Windows Scheduled Task running every 2 minutes indefinitely."""
    pyw_exe = sys.executable
    if not os.path.exists(pyw_exe):
        pyw_exe = sys.executable

    script_path = os.path.abspath(__file__)
    task_cmd = f'"{pyw_exe}" "{script_path}" --once'

    cmd = [
        "schtasks",
        "/Create",
        "/SC",
        "MINUTE",
        "/MO",
        "2",
        "/TN",
        TASK_NAME,
        "/TR",
        task_cmd,
        "/F",
    ]
    logger.info(f"Installing Scheduled Task '{TASK_NAME}'...")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode == 0:
        logger.info(f"SUCCESS: Created Windows Scheduled Task '{TASK_NAME}' (runs every 2 mins).")
        return True
    else:
        logger.error(f"Failed to create task: {res.stderr or res.stdout}")
        return False


def uninstall_scheduled_task() -> bool:
    """Removes the Windows Scheduled Task cleanly."""
    cmd = ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode == 0:
        logger.info(f"SUCCESS: Removed Windows Scheduled Task '{TASK_NAME}'.")
        return True
    else:
        logger.warning(f"Could not remove task (may not exist): {res.stderr or res.stdout}")
        return False


def run_daemon_loop(interval_sec: int = 60):
    """Runs continuous background monitoring loop."""
    logger.info(f"Starting Allsky Workflow Health Guard Daemon (interval={interval_sec}s)...")
    try:
        while True:
            try:
                run_guard_cycle(trigger_discord_alerts=True)
            except Exception as e:
                logger.error(f"Error in guard cycle: {e}")
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        logger.info("Guard daemon terminated by user.")


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BluffSky Allsky Workflow Health Guard & Self-Healing Automation")
    parser.add_argument("--once", action="store_true", help="Run a single health check & self-healing cycle")
    parser.add_argument("--daemon", action="store_true", help="Run continuous background watchdog daemon")
    parser.add_argument("--interval", type=int, default=60, help="Interval in seconds for daemon mode (default 60s)")
    parser.add_argument("--install-task", action="store_true", help="Install Windows Scheduled Task (runs every 2 mins)")
    parser.add_argument("--uninstall-task", action="store_true", help="Uninstall Windows Scheduled Task")
    parser.add_argument("--status", action="store_true", help="Output current status from workflow_guard_status.json")
    args = parser.parse_args()

    if args.install_task:
        ok = install_scheduled_task()
        sys.exit(0 if ok else 1)

    if args.uninstall_task:
        ok = uninstall_scheduled_task()
        sys.exit(0 if ok else 1)

    if args.status:
        if os.path.exists(GUARD_STATUS_FILE):
            with open(GUARD_STATUS_FILE, "r", encoding="utf-8") as f:
                print(f.read())
        else:
            rep = run_guard_cycle(trigger_discord_alerts=False)
            print(json.dumps(rep, indent=2))
        return

    if args.daemon:
        run_daemon_loop(interval_sec=args.interval)
        return

    # Default: Run once
    rep = run_guard_cycle(trigger_discord_alerts=True)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
