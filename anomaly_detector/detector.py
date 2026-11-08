import argparse
import base64
from collections import deque
from datetime import datetime, timedelta, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
HISTORY_FILE = os.path.join(BASE_DIR, "detections_history.json")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
MASK_FILE = os.path.join(BASE_DIR, "sky_mask.png")
MASK_PREVIEW_FILE = os.path.join(BASE_DIR, "sky_mask_preview.jpg")
REF_DAY_FILE = os.path.join(BASE_DIR, "reference_day.jpg")
LOG_FILE = os.path.join(BASE_DIR, "detector.log")
LIVE_STATUS_FILE = os.path.join(BASE_DIR, "live_status.json")
LATEST_LIVE_FRAME_FILE = os.path.join(BASE_DIR, "latest_live_frame.jpg")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logger = logging.getLogger("AllskyDetector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

# Category Priorities & Display Metadata
CATEGORY_PRIORITY = {
    "fireball": 10,
    "meteor": 9,
    "sky_phenomenon": 8,
    "space_object": 7,
    "transient_flash": 6,
    "aircraft": 5,
    "day_anomaly": 4,
    "day_aircraft": 3,
    "day_bird": 2,
}

CATEGORY_BADGES = {
    "fireball": ("🔥 Bright Fireball / Bolide Detected!", "critical", (0, 110, 255)),       # Bright Orange-Red (BGR)
    "meteor": ("🌠 Shooting Star / Meteor Detected", "success", (120, 255, 80)),            # Luminous Green-Cyan
    "space_object": ("🛰️ Satellite / Space Object Detected", "info", (255, 220, 60)),      # Sky Cyan
    "transient_flash": ("✨ Optical Space Flash / Glint", "info", (255, 180, 220)),         # Magenta-White
    "sky_phenomenon": ("🌌 Luminous Sky Phenomenon", "warning", (180, 255, 120)),           # Aurora Green
    "aircraft": ("✈️ Night Aircraft Transit", "info", (80, 200, 255)),                      # Amber-Gold
    "day_anomaly": ("☀️ Daytime Sky Anomaly", "warning", (0, 215, 255)),                    # Gold
    "day_aircraft": ("🛩️ Daytime Aircraft / Contrail", "info", (255, 200, 50)),            # Blue-Cyan
    "day_bird": ("🦅 Soaring Bird / Raptor in Sky Dome", "info", (140, 255, 140)),          # Mint Green
}

CARDINALS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

_SKY_MASK_CACHE: Dict[str, Any] = {"mask": None, "erosion": None, "max_r": None, "by_cam": {}}
_YOLO_MODELS: Dict[str, Any] = {}
_LAST_ALERT_TIMES: Dict[str, float] = {}
_RECENT_TRACKS: List[Dict[str, Any]] = []  # Tracks recent moving aircraft/satellites across consecutive frames
_RECENT_CAMERA_DETECTIONS: List[Dict[str, Any]] = []  # Cross-camera coincidence & stereo parallax buffer (1..N cameras)


def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        example_file = os.path.join(BASE_DIR, "config.example.json")
        if os.path.exists(example_file):
            try:
                import shutil
                shutil.copyfile(example_file, CONFIG_FILE)
            except Exception:
                pass
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config.json: {e}")
    return {}


def get_configured_cameras(cfg: dict, enabled_only: bool = True) -> List[Dict[str, Any]]:
    """
    Returns the list of 1..N configured camera nodes.
    If `cfg["cameras"]` is omitted or empty, synthesizes a single camera node (`cam1`)
    from the top-level config fields for 100% backward compatibility.
    """
    raw_cams = cfg.get("cameras")
    if not isinstance(raw_cams, list) or len(raw_cams) == 0:
        base_url = str(cfg.get("allsky_base_url", "http://192.168.1.100")).rstrip("/")
        raw_cams = [
            {
                "id": "cam1",
                "node_id": "bluffsky-pi1",
                "name": cfg.get("camera_name", "BluffSky Allsky (ASI676MC)"),
                "enabled": True,
                "allsky_base_url": base_url,
                "live_image_url": cfg.get("live_image_url", f"{base_url}/current/tmp/image.jpg"),
                "overlay_debug_url": cfg.get("overlay_debug_url", f"{base_url}/current/tmp/overlaydebug.txt"),
                "telemetry_api_url": f"{base_url}:8096/api/status",
                "camera_lat": cfg.get("camera_lat", 45.0000),
                "camera_lon": cfg.get("camera_lon", -100.0000),
                "camera_alt_ft": cfg.get("camera_alt_ft", 130.0),
                "east_offset_m": 0.0,
                "north_offset_m": 0.0,
                "up_offset_m": 0.0,
            }
        ]
    out: List[Dict[str, Any]] = []
    for idx, c_item in enumerate(raw_cams, start=1):
        if not isinstance(c_item, dict):
            continue
        cam_id = str(c_item.get("id") or f"cam{idx}").strip()
        enabled = bool(c_item.get("enabled", True))
        if enabled_only and not enabled:
            continue
        base_u = str(c_item.get("allsky_base_url") or cfg.get("allsky_base_url", "http://192.168.1.100")).rstrip("/")
        entry = dict(c_item)
        entry["id"] = cam_id
        entry["node_id"] = str(c_item.get("node_id") or f"bluffsky-pi{idx}")
        entry["name"] = str(c_item.get("name") or f"Allsky Camera #{idx} ({cam_id})")
        entry["enabled"] = enabled
        entry["allsky_base_url"] = base_u
        entry["live_image_url"] = str(c_item.get("live_image_url") or f"{base_u}/current/tmp/image.jpg")
        entry["overlay_debug_url"] = str(c_item.get("overlay_debug_url") or f"{base_u}/current/tmp/overlaydebug.txt")
        entry["telemetry_api_url"] = str(c_item.get("telemetry_api_url") or f"{base_u}:8096/api/status")
        out.append(entry)
    return out


def get_effective_camera_cfg(cfg: dict, cam_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Merges global config with camera-specific calibration, URLs, and GPS/baseline offsets."""
    merged = dict(cfg)
    merged.update(cam_entry)
    merged["camera_id"] = str(cam_entry.get("id") or "cam1")
    merged["camera_name"] = str(cam_entry.get("name") or cfg.get("camera_name", "Allsky Camera"))
    return merged


def get_camera_live_frame_path(cam_id: str) -> str:
    """Returns the local NVMe cached live frame path for `cam_id`."""
    if not cam_id or cam_id == "cam1":
        return LATEST_LIVE_FRAME_FILE
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", cam_id)
    return os.path.join(BASE_DIR, f"latest_live_frame_{safe_id}.jpg")


def match_multi_camera_stereo(
    cand: Dict[str, Any],
    cam_cfg: Dict[str, Any],
    frame_ts: float,
) -> Optional[Dict[str, Any]]:
    """
    Correlates detections across 1..N cameras within a 12-second window.
    When a second camera (`cam_B != cam_A`) observes the same event simultaneously,
    computes angular parallax separation and (when physical baseline offsets are known)
    triangulates 3D slant range and altitude.
    """
    global _RECENT_CAMERA_DETECTIONS
    cam_id = str(cam_cfg.get("camera_id") or "cam1")
    cam_name = str(cam_cfg.get("camera_name") or cam_id)
    alt_deg = float(cand.get("alt_deg") or 0.0)
    az_deg = float(cand.get("az_deg") or 0.0)
    alt_r = math.radians(alt_deg)
    az_r = math.radians(az_deg)
    # ENU unit vector u = [East, North, Up]
    ux = math.cos(alt_r) * math.sin(az_r)
    uy = math.cos(alt_r) * math.cos(az_r)
    uz = math.sin(alt_r)
    east_m = float(cam_cfg.get("east_offset_m") or 0.0)
    north_m = float(cam_cfg.get("north_offset_m") or 0.0)
    up_m = float(cam_cfg.get("up_offset_m") or 0.0)

    now_ts = float(frame_ts or time.time())
    _RECENT_CAMERA_DETECTIONS = [d for d in _RECENT_CAMERA_DETECTIONS if abs(now_ts - d["ts"]) <= 25.0]

    best_match: Optional[Dict[str, Any]] = None
    best_ang_diff = 35.0  # Max plausible parallax angle (deg) for atmospheric/orbital objects

    for prev in _RECENT_CAMERA_DETECTIONS:
        if prev["camera_id"] == cam_id:
            continue
        dt = abs(now_ts - prev["ts"])
        if dt > 12.0:
            continue
        dot = max(-1.0, min(1.0, ux * prev["ux"] + uy * prev["uy"] + uz * prev["uz"]))
        parallax_deg = math.degrees(math.acos(dot))
        if parallax_deg <= best_ang_diff:
            best_ang_diff = parallax_deg
            bx = prev["east_m"] - east_m
            by = prev["north_m"] - north_m
            bz = prev["up_m"] - up_m
            baseline_m = math.sqrt(bx * bx + by * by + bz * bz)
            triangulated_alt_km = None
            slant_range_km = None
            if baseline_m > 1.0 and parallax_deg >= 0.015:
                # Cross products ||b x u_B|| / ||u_A x u_B||
                cx_bu = by * prev["uz"] - bz * prev["uy"]
                cy_bu = bz * prev["ux"] - bx * prev["uz"]
                cz_bu = bx * prev["uy"] - by * prev["ux"]
                num = math.sqrt(cx_bu * cx_bu + cy_bu * cy_bu + cz_bu * cz_bu)
                cx_uu = uy * prev["uz"] - uz * prev["uy"]
                cy_uu = uz * prev["ux"] - ux * prev["uz"]
                cz_uu = ux * prev["uy"] - uy * prev["ux"]
                den = max(1e-6, math.sqrt(cx_uu * cx_uu + cy_uu * cy_uu + cz_uu * cz_uu))
                r_m = num / den
                slant_range_km = round(r_m / 1000.0, 2)
                triangulated_alt_km = round((r_m * max(0.05, uz)) / 1000.0, 2)
            best_match = {
                "stereo_corroborated": True,
                "corroborated_by_camera_id": prev["camera_id"],
                "corroborated_by_camera_name": prev["camera_name"],
                "time_delta_sec": round(dt, 3),
                "parallax_deg": round(parallax_deg, 4),
                "baseline_m": round(baseline_m, 2),
                "slant_range_km": slant_range_km,
                "triangulated_alt_km": triangulated_alt_km,
            }

    _RECENT_CAMERA_DETECTIONS.append(
        {
            "camera_id": cam_id,
            "camera_name": cam_name,
            "ts": now_ts,
            "ux": ux,
            "uy": uy,
            "uz": uz,
            "east_m": east_m,
            "north_m": north_m,
            "up_m": up_m,
            "category": cand.get("category"),
        }
    )
    return best_match


def save_config(cfg: dict) -> None:
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save config.json: {e}")


def load_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"processed_frames": {}, "recent_events": []}


def save_history(history: dict) -> None:
    try:
        if "recent_events" in history and len(history["recent_events"]) > 500:
            history["recent_events"] = history["recent_events"][-500:]
        if "processed_frames" in history and len(history["processed_frames"]) > 6000:
            keys = list(history["processed_frames"].keys())[-4500:]
            history["processed_frames"] = {k: history["processed_frames"][k] for k in keys}
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save detections_history.json: {e}")


def update_live_status(status_update: dict) -> None:
    try:
        cur = {}
        if os.path.exists(LIVE_STATUS_FILE):
            try:
                with open(LIVE_STATUS_FILE, "r", encoding="utf-8") as f:
                    cur = json.load(f)
            except Exception:
                cur = {}
        cur.update(status_update)
        cur["updated_at"] = time.time()
        cur["updated_iso"] = datetime.now().isoformat()
        with open(LIVE_STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(cur, f, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Sky Dome Mask Calibration & Astronomical Alt/Az Coordinate Mapping
# ---------------------------------------------------------------------------
def ensure_sky_mask(cfg: dict, shape: Tuple[int, int] = (3552, 3552)) -> np.ndarray:
    """
    Builds or loads the pixel-accurate unobstructed Sky Dome Mask (uint8, 255=sky, 0=obstacles).
    Excludes the house roofline/windows (top-left), fir trees around the horizon, saplings,
    ground lights, and outer fisheye border.
    """
    global _SKY_MASK_CACHE
    h, w = shape[:2]
    cam_id = str(cfg.get("camera_id") or "cam1")
    erosion_px = int(cfg.get("mask_erosion_px", 32))
    max_r = int(cfg.get("max_sky_radius_px", 1530))
    cx, cy = cfg.get("mask_center", [1718, 1796])
    cache_key = (cam_id, h, w, erosion_px, max_r, int(cx), int(cy))

    by_cam = _SKY_MASK_CACHE.setdefault("by_cam", {})
    if _SKY_MASK_CACHE.get("mask") is None:
        by_cam.clear()
    if cache_key in by_cam:
        return by_cam[cache_key]

    ref_day_path = REF_DAY_FILE if cam_id == "cam1" else os.path.join(BASE_DIR, f"reference_day_{cam_id}.jpg")
    if not os.path.exists(ref_day_path):
        ref_day_path = REF_DAY_FILE
    mask_out_path = MASK_FILE if cam_id == "cam1" else os.path.join(BASE_DIR, f"sky_mask_{cam_id}.png")

    sky_final = None
    if os.path.exists(ref_day_path):
        day = cv2.imread(ref_day_path)
        if day is not None:
            if day.shape[:2] != (h, w):
                day = cv2.resize(day, (w, h), interpolation=cv2.INTER_AREA)
            b, g, r = cv2.split(day)
            sky_raw = ((b > 75) & (g > 60)).astype(np.uint8) * 255
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky_raw)
            if num_labels > 1:
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                sky_cc = (labels == largest).astype(np.uint8) * 255
                cnts, _ = cv2.findContours(sky_cc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filled = np.zeros_like(sky_cc)
                cv2.drawContours(filled, cnts, -1, 255, -1)
                obstacles = ((b < 55) & (r < 55)).astype(np.uint8) * 255
                sky_clean = cv2.bitwise_and(filled, cv2.bitwise_not(obstacles))
                num_l2, lbls2, st2, _ = cv2.connectedComponentsWithStats(sky_clean)
                if num_l2 > 1:
                    largest2 = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
                    sky_clean = (lbls2 == largest2).astype(np.uint8) * 255

                circle_mask = np.zeros_like(sky_clean)
                cv2.circle(circle_mask, (int(cx), int(cy)), int(max_r), 255, -1)
                sky_clean = cv2.bitwise_and(sky_clean, circle_mask)

                k_size = max(3, erosion_px * 2 + 1)
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
                sky_eroded = cv2.erode(sky_clean, kern)
                sky_eroded = cv2.GaussianBlur(sky_eroded, (31, 31), 0)
                _, sky_final = cv2.threshold(sky_eroded, 128, 255, cv2.THRESH_BINARY)

    if sky_final is None:
        # Geometric fallback if reference_day.jpg is missing
        sky_final = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(sky_final, (int(cx), int(cy)), int(max_r * 0.88), 255, -1)
        # Mask out house sector in top-left
        house_poly = np.array([[0, 0], [int(w * 0.56), 0], [int(w * 0.48), int(h * 0.25)], [int(w * 0.22), int(h * 0.35)], [0, int(h * 0.42)]], np.int32)
        cv2.fillPoly(sky_final, [house_poly], 0)

    try:
        cv2.imwrite(mask_out_path, sky_final)
        if os.path.exists(ref_day_path) and not os.path.exists(MASK_PREVIEW_FILE):
            preview = cv2.imread(ref_day_path)
            if preview is not None:
                cnts_p, _ = cv2.findContours(sky_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(preview, cnts_p, -1, (0, 255, 160), 6)
                preview_small = cv2.resize(preview, (1024, 1024), interpolation=cv2.INTER_AREA)
                cv2.imwrite(MASK_PREVIEW_FILE, preview_small, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    except Exception as e:
        logger.warning(f"Failed saving sky mask preview: {e}")

    by_cam[cache_key] = sky_final
    _SKY_MASK_CACHE["mask"] = sky_final
    _SKY_MASK_CACHE["erosion"] = erosion_px
    _SKY_MASK_CACHE["max_r"] = max_r
    return sky_final


def pixel_to_alt_az(x: float, y: float, cfg: dict, shape: Tuple[int, int] = (3552, 3552)) -> Tuple[float, float, str]:
    """
    Converts pixel (x, y) on the circular fisheye sensor to:
      (altitude_deg [0..90], azimuth_deg [0..360], cardinal_str)
    using the 22-star 3D Spherical Tilt + 5th-Order Radial Fisheye calibration.
    """
    h, w = shape[:2]
    cx, cy = cfg.get("zenith_center", [1750.95, 1690.25])
    r_horiz = float(cfg.get("horizon_radius_px", 2007.74))
    bottom_az = float(cfg.get("bottom_azimuth_deg", 265.055))
    tilt_x = math.radians(float(cfg.get("tilt_x_deg", 6.292)))
    tilt_y = math.radians(float(cfg.get("tilt_y_deg", -0.947)))
    k2 = float(cfg.get("radial_k2", -0.14857))
    k3 = float(cfg.get("radial_k3", 0.03125))

    scale_x = w / 3552.0
    scale_y = h / 3552.0
    cx_s = float(cx) * scale_x
    cy_s = float(cy) * scale_y
    r_horiz_s = max(1.0, r_horiz * ((scale_x + scale_y) * 0.5))

    dx = (float(x) - cx_s) / r_horiz_s
    dy = (float(y) - cy_s) / r_horiz_s
    r_norm = math.hypot(dx, dy)

    # Invert 5th-order fisheye radial mapping r_norm = u + k2 * u^3 + k3 * u^5 via Newton-Raphson
    u = min(1.18, max(0.0, r_norm))
    if abs(k2) > 1e-6 or abs(k3) > 1e-6:
        for _ in range(5):
            f_val = u + k2 * (u ** 3) + k3 * (u ** 5) - r_norm
            df_val = max(0.15, 1.0 + 3.0 * k2 * (u ** 2) + 5.0 * k3 * (u ** 4))
            u = max(0.0, min(1.25, u - f_val / df_val))

    theta = u * (math.pi / 2.0)
    if r_norm > 1e-9:
        vx_c = math.sin(theta) * (dx / r_norm)
        vy_c = math.sin(theta) * (dy / r_norm)
    else:
        vx_c, vy_c = 0.0, 0.0
    vz_c = math.cos(theta)

    # Undo Ry(tilt_y) -> Rx(tilt_x) -> R0(bottom_az)
    cty, sty = math.cos(tilt_y), math.sin(tilt_y)
    vx1 = cty * vx_c - sty * vz_c
    vy1 = vy_c
    vz1 = sty * vx_c + cty * vz_c

    ctx, stx = math.cos(tilt_x), math.sin(tilt_x)
    vx0 = vx1
    vy0 = ctx * vy1 + stx * vz1
    vz0 = -stx * vy1 + ctx * vz1

    phi = math.radians(bottom_az)
    cb, sb = math.cos(phi), math.sin(phi)
    east = cb * vx0 + sb * vy0
    north = -sb * vx0 + cb * vy0
    up = vz0

    alt = max(0.0, min(90.0, math.degrees(math.asin(max(-1.0, min(1.0, up))))))
    az = math.degrees(math.atan2(east, north)) % 360.0
    idx = int(round(az / 22.5)) % 16
    return round(alt, 1), round(az, 1), CARDINALS[idx]


def compute_ephemeris_sun_moon(
    ts_utc: float,
    cfg: dict,
    shape: Tuple[int, int] = (3552, 3552),
) -> Dict[str, Any]:
    """
    Computes exact astronomical topocentric (Alt, Az) and 3D fisheye sensor pixel (x, y)
    for both the Moon and the Sun at `ts_utc`, even when hidden behind clouds or trees.
    """
    out = {
        "moon_alt_deg": -90.0,
        "moon_az_deg": 0.0,
        "moon_phase": 0.0,
        "moon_px": (-9999, -9999),
        "sun_alt_deg": -90.0,
        "sun_az_deg": 0.0,
        "sun_px": (-9999, -9999),
    }
    try:
        import ephem
        from datetime import timezone
        import adsb_tracker

        obs = ephem.Observer()
        obs.lat = str(cfg.get("camera_lat", 45.0000))
        obs.lon = str(cfg.get("camera_lon", -100.0000))
        obs.elevation = float(cfg.get("camera_alt_ft", 130.0)) * 0.3048
        obs.date = datetime.fromtimestamp(float(ts_utc), tz=timezone.utc)

        moon = ephem.Moon(obs)
        m_alt = math.degrees(float(moon.alt))
        m_az = math.degrees(float(moon.az)) % 360.0
        m_phase = float(moon.phase)
        mx, my = adsb_tracker.altaz_to_pixel(max(-6.0, m_alt), m_az, cfg, shape)

        sun = ephem.Sun(obs)
        s_alt = math.degrees(float(sun.alt))
        s_az = math.degrees(float(sun.az)) % 360.0
        sx, sy = adsb_tracker.altaz_to_pixel(max(-6.0, s_alt), s_az, cfg, shape)

        out.update({
            "moon_alt_deg": round(m_alt, 1),
            "moon_az_deg": round(m_az, 1),
            "moon_phase": round(m_phase, 1),
            "moon_phase_pct": round(m_phase, 1),
            "moon_px": (mx, my),
            "sun_alt_deg": round(s_alt, 1),
            "sun_az_deg": round(s_az, 1),
            "sun_px": (sx, sy),
        })
    except Exception as e:
        logger.debug(f"Ephemeris Sun/Moon computation skipped: {e}")
    return out


# ---------------------------------------------------------------------------
# Allsky Pi 5 HTTP Fetching & Telemetry Parser
# ---------------------------------------------------------------------------
def fetch_overlay_metadata(session: requests.Session, cfg: dict) -> Dict[str, str]:
    url = cfg.get("overlay_debug_url", "http://192.168.1.100/current/tmp/overlaydebug.txt")
    meta: Dict[str, str] = {}
    try:
        r = session.get(url, timeout=4)
        if r.status_code == 200:
            for line in r.text.splitlines():
                line_s = line.strip()
                if not line_s:
                    continue
                parts = re.split(r"\s{2,}|\t+", line_s, maxsplit=1)
                if len(parts) == 2:
                    meta[parts[0].strip()] = parts[1].strip()
    except Exception as e:
        logger.debug(f"Could not fetch overlaydebug.txt: {e}")
    return meta


def parse_frame_timestamp(filename: str, fallback_meta: Optional[Dict[str, str]] = None) -> Tuple[str, str, int, str]:
    """
    Returns (iso_datetime, date_str YYYY-MM-DD, hour_int, display_str).
    Matches Allsky filenames like image-20261015010004.jpg.
    """
    m = re.search(r"image-(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", filename)
    if m:
        try:
            dt = datetime(
                int(m.group(1)),
                int(m.group(2)),
                int(m.group(3)),
                int(m.group(4)),
                int(m.group(5)),
                int(m.group(6)),
            )
            return dt.isoformat(), dt.strftime("%Y-%m-%d"), dt.hour, dt.strftime("%Y-%m-%d %I:%M:%S %p")
        except Exception:
            pass
    now = datetime.now()
    return now.isoformat(), now.strftime("%Y-%m-%d"), now.hour, now.strftime("%Y-%m-%d %I:%M:%S %p")


def get_frame_exposure_window(
    filename: str,
    exp_sec: float,
    fallback_end_ts: float,
    overlay_meta: Optional[Dict[str, str]] = None,
) -> Tuple[float, float]:
    """
    Returns the exact Unix timestamp window `(t_exp_start, t_exp_end)` when the Allsky camera
    shutter was open for `filename`.
    When Option A microsecond telemetry (`AS_EXPOSURE_START_US`, `AS_EXPOSURE_END_US`) is present
    in `overlay_meta` for the current frame, uses those exact microsecond timestamps directly.
    Otherwise falls back to the Allsky filename timestamp (`STARTING EXPOSURE`).
    """
    if overlay_meta:
        cur_img = os.path.basename(overlay_meta.get("CURRENT_IMAGE", ""))
        if not cur_img or cur_img == os.path.basename(filename):
            s_us = overlay_meta.get("AS_EXPOSURE_START_US")
            e_us = overlay_meta.get("AS_EXPOSURE_END_US")
            if s_us and e_us:
                try:
                    t_s = int(s_us) / 1_000_000.0
                    t_e = int(e_us) / 1_000_000.0
                    if t_e > t_s > 1_700_000_000.0:
                        return t_s, t_e
                except Exception:
                    pass
    dur = max(0.05, float(exp_sec))
    if "-sim-" not in filename:
        m = re.search(r"image-(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", filename)
        if m:
            try:
                dt = datetime(
                    int(m.group(1)),
                    int(m.group(2)),
                    int(m.group(3)),
                    int(m.group(4)),
                    int(m.group(5)),
                    int(m.group(6)),
                )
                t_start = dt.timestamp() + 0.3
                t_end = t_start + dur
                return t_start, t_end
            except Exception:
                pass
    return fallback_end_ts - dur, fallback_end_ts


def infer_day_or_night(img: np.ndarray, sky_mask: np.ndarray, meta: Dict[str, str]) -> str:
    """
    Determines whether the frame is NIGHT or DAY using both Allsky's DAY_OR_NIGHT telemetry
    and actual sky dome luminance/saturation (so twilight transitions are handled gracefully).
    """
    tod = (meta.get("DAY_OR_NIGHT") or "").upper()
    # Compute mean brightness inside the unobstructed sky dome
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mean_sky = float(cv2.mean(gray, mask=sky_mask)[0])
    gain_str = meta.get("AS_GAIN", "")
    try:
        gain = float(gain_str)
    except Exception:
        gain = 0.0

    if tod == "NIGHT":
        return "NIGHT"
    if tod == "DAY":
        # During late dusk before Allsky switches DAY_OR_NIGHT to NIGHT, if sky dome is dark (< 48) or gain >= 100, treat as NIGHT
        if mean_sky < 48.0 or gain >= 100.0:
            return "NIGHT"
        return "DAY"
    return "NIGHT" if (mean_sky < 65.0 or gain >= 100.0) else "DAY"


# ---------------------------------------------------------------------------
# Rolling Temporal Sky Buffer
# ---------------------------------------------------------------------------
class RollingSkyBuffer:
    """
    Maintains the last N frames (separately for NIGHT and DAY modes) so we can compute:
      1. Immediate frame-to-frame high-pass innovation (t vs t-1)
      2. Multi-frame median high-pass background (rejecting hot pixels, stars, planets, and persistent structures)
      3. Previous raw frame for Side-by-Side Before/During VLM & PiP cards
    """

    def __init__(self, maxlen: int = 5):
        self.maxlen = maxlen
        self.frames: deque = deque(maxlen=maxlen)  # list of dicts: {filename, mode, img, gray, hp, timestamp}

    def clear(self):
        self.frames.clear()

    def get_compatible_history(self, mode: str, shape: Tuple[int, int]) -> List[dict]:
        return [f for f in self.frames if f["mode"] == mode and f["gray"].shape == shape]

    def push(self, filename: str, mode: str, img: np.ndarray, gray: np.ndarray, hp: np.ndarray, ts: float = None):
        self.frames.append({
            "filename": filename,
            "mode": mode,
            "img": img,
            "gray": gray,
            "hp": hp,
            "timestamp": ts or time.time(),
        })


# ---------------------------------------------------------------------------
# Stage-1 Night-Time Anomaly Detection Engine (Meteors, Fireballs, Satellites, Planes, Flashes)
# ---------------------------------------------------------------------------
def compute_highpass(gray: np.ndarray, bg_kernel: int = 35) -> Tuple[np.ndarray, np.ndarray]:
    k = bg_kernel if bg_kernel % 2 == 1 else bg_kernel + 1
    bg = cv2.medianBlur(gray, k)
    hp = cv2.subtract(gray, bg)
    return hp, bg


def analyze_streak_profile(
    pts_xy: np.ndarray,
    hp_innovation: np.ndarray,
    raw_gray: np.ndarray,
    raw_bgr: np.ndarray,
) -> Dict[str, Any]:
    """
    Given the pixel coordinates (N x 2, [x, y]) of a candidate anomaly in the temporal high-pass innovation:
      - Fits the principal axis via PCA (eigenvalues -> linearity R^2, angle_deg, projected span length_px, width_px)
      - Samples the 1D intensity profile along the streak to count periodic strobe peaks (aircraft vs smooth meteor/satellite)
      - Computes peak innovation delta, peak raw brightness, integrated flux, and RGB color ratios.
    """
    n_pts = len(pts_xy)
    if n_pts < 5:
        return {"valid": False}

    mean_xy = np.mean(pts_xy, axis=0)
    centered = pts_xy - mean_xy
    cov = np.cov(centered.T)
    if cov.ndim < 2:
        return {"valid": False}

    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns eigenvalues in ascending order: eigvals[1] is major, eigvals[0] is minor
    lam_minor = max(1e-6, float(eigvals[0]))
    lam_major = max(1e-6, float(eigvals[1]))
    linearity = lam_major / (lam_major + lam_minor)

    major_vec = eigvecs[:, 1]  # unit vector along streak
    minor_vec = eigvecs[:, 0]

    proj_major = centered @ major_vec
    proj_minor = centered @ minor_vec

    p_min, p_max = float(np.percentile(proj_major, 1)), float(np.percentile(proj_major, 99))
    w_min, w_max = float(np.percentile(proj_minor, 5)), float(np.percentile(proj_minor, 95))

    length_px = max(1.0, p_max - p_min)
    width_px = max(1.5, w_max - w_min)
    aspect = length_px / width_px
    angle_deg = math.degrees(math.atan2(float(major_vec[1]), float(major_vec[0]))) % 180.0

    # Endpoints in image coordinates
    p1 = mean_xy + p_min * major_vec
    p2 = mean_xy + p_max * major_vec

    # Sample 1D profile along the principal axis to distinguish smooth meteor vs blinking aircraft strobes
    num_samples = max(12, min(120, int(round(length_px / 3.0))))
    ts = np.linspace(p_min, p_max, num_samples)
    sample_pts = mean_xy[None, :] + ts[:, None] * major_vec[None, :]
    h, w = raw_gray.shape[:2]
    sx = np.clip(np.round(sample_pts[:, 0]).astype(int), 0, w - 1)
    sy = np.clip(np.round(sample_pts[:, 1]).astype(int), 0, h - 1)

    # Take max in a 3x3 neighborhood along the axis
    hp_pad = cv2.dilate(hp_innovation, np.ones((3, 3), np.uint8))
    profile = hp_pad[sy, sx].astype(np.float32)

    # Smooth slightly and count prominent local maxima (strobe flashes)
    if len(profile) >= 7:
        k_smooth = np.array([0.2, 0.6, 0.2], dtype=np.float32)
        prof_s = np.convolve(profile, k_smooth, mode="same")
    else:
        prof_s = profile

    prof_max = float(np.max(prof_s)) if len(prof_s) > 0 else 1.0
    prof_min = float(np.min(prof_s)) if len(prof_s) > 0 else 0.0
    fill_ratio = float(np.mean(prof_s > max(12.0, prof_max * 0.18))) if len(prof_s) > 0 else 0.0
    strobe_peaks = 0
    deep_gaps = 0
    if len(prof_s) >= 9 and prof_max > 15.0:
        peak_thresh = prof_max * 0.38
        gap_thresh = prof_max * 0.22
        in_gap = False
        in_peak = False
        for i in range(1, len(prof_s) - 1):
            if prof_s[i] > peak_thresh:
                if not in_peak and prof_s[i] >= prof_s[i - 1] and prof_s[i] >= prof_s[i + 1]:
                    strobe_peaks += 1
                    in_peak = True
            else:
                in_peak = False
            if prof_s[i] < gap_thresh:
                if not in_gap:
                    deep_gaps += 1
                    in_gap = True
            else:
                in_gap = False

    xs_i = np.clip(pts_xy[:, 0].astype(int), 0, w - 1)
    ys_i = np.clip(pts_xy[:, 1].astype(int), 0, h - 1)
    vals_hp = hp_innovation[ys_i, xs_i]
    vals_raw = raw_gray[ys_i, xs_i]
    vals_bgr = raw_bgr[ys_i, xs_i].astype(np.float32)

    peak_delta = float(np.max(vals_hp))
    mean_delta = float(np.mean(vals_hp))
    peak_raw = float(np.max(vals_raw))
    flux = float(np.sum(vals_hp))

    # Color tint analysis (for green fireballs / aurora vs red/green wingtip strobes)
    mean_b, mean_g, mean_r = [float(x) for x in np.mean(vals_bgr, axis=0)]
    chroma_spread = max(abs(mean_r - mean_g), abs(mean_g - mean_b), abs(mean_r - mean_b))

    return {
        "valid": True,
        "center": (float(mean_xy[0]), float(mean_xy[1])),
        "p1": (float(p1[0]), float(p1[1])),
        "p2": (float(p2[0]), float(p2[1])),
        "length_px": round(length_px, 1),
        "width_px": round(width_px, 1),
        "aspect": round(aspect, 2),
        "linearity": round(linearity, 3),
        "fill_ratio": round(fill_ratio, 3),
        "angle_deg": round(angle_deg, 1),
        "strobe_peaks": strobe_peaks,
        "deep_gaps": deep_gaps,
        "peak_delta": round(peak_delta, 1),
        "mean_delta": round(mean_delta, 1),
        "peak_raw": round(peak_raw, 1),
        "flux": round(flux, 1),
        "chroma_spread": round(chroma_spread, 1),
        "area_px": n_pts,
    }


def merge_collinear_clusters(clusters: List[Dict[str, Any]], max_dist_px: float = 320.0, angle_tol_deg: float = 12.0) -> List[Dict[str, Any]]:
    """
    Merges nearby collinear streak clusters (such as segmented aircraft trails or fragmented meteor trails)
    into a single composite track so they are analyzed and boxed as one cohesive event.
    """
    if len(clusters) <= 1:
        return [{"pts_xy": c["pts_xy"], "sub_clusters": 1, "is_thin_streak": bool(c.get("is_thin_streak"))} for c in clusters]

    def ang_diff(a: float, b: float) -> float:
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    def cluster_key_pts(c: Dict[str, Any]) -> List[Tuple[float, float]]:
        pts = [c["center"]]
        if c.get("p1"):
            pts.append(c["p1"])
        if c.get("p2"):
            pts.append(c["p2"])
        return pts

    # Seed groups starting from the longest/clearest streak segments first
    order = sorted(range(len(clusters)), key=lambda k: clusters[k].get("length_px", 0.0), reverse=True)
    used = [False] * len(clusters)
    merged = []

    for i in order:
        if used[i]:
            continue
        base = clusters[i]
        group_members = [base]
        group_pts = [base["pts_xy"]]
        used[i] = True

        gcx, gcy = base["center"]
        gang = base.get("angle_deg")
        glen = float(base.get("length_px", 0.0))

        # If base is a compact point strobe (< 25 px), check if 2+ other unused clusters form a collinear strobe dot train with it
        if glen < 25.0:
            for j in order:
                if used[j] or math.hypot(clusters[j]["center"][0] - gcx, clusters[j]["center"][1] - gcy) > max_dist_px:
                    continue
                vj = math.degrees(math.atan2(clusters[j]["center"][1] - gcy, clusters[j]["center"][0] - gcx)) % 180.0
                vj_rad = math.radians(vj)
                for k in order:
                    if k == j or used[k] or math.hypot(clusters[k]["center"][0] - gcx, clusters[k]["center"][1] - gcy) > max_dist_px * 1.8:
                        continue
                    vk = math.degrees(math.atan2(clusters[k]["center"][1] - gcy, clusters[k]["center"][0] - gcx)) % 180.0
                    p_dist = abs(-(clusters[k]["center"][0] - gcx) * math.sin(vj_rad) + (clusters[k]["center"][1] - gcy) * math.cos(vj_rad))
                    if ang_diff(vj, vk) <= 7.0 and p_dist <= 12.0:
                        gang = vj
                        glen = 30.0
                        break
                if glen >= 25.0:
                    break

        changed = True
        while changed:
            changed = False
            for j in order:
                if used[j]:
                    continue
                cand = clusters[j]
                cand_pts = cluster_key_pts(cand)
                min_gap = min(
                    math.hypot(pa[0] - pb[0], pa[1] - pb[1])
                    for m in group_members
                    for pa in cluster_key_pts(m)
                    for pb in cand_pts
                )
                if min_gap > max_dist_px:
                    continue

                cx2, cy2 = cand["center"]
                vec_angle = math.degrees(math.atan2(cy2 - gcy, cx2 - gcx)) % 180.0
                ang2 = cand.get("angle_deg")
                clen = float(cand.get("length_px", 0.0))

                aligned = False
                if gang is not None and ang2 is not None:
                    gang_rad = math.radians(gang)
                    perp_dist = abs(-(cx2 - gcx) * math.sin(gang_rad) + (cy2 - gcy) * math.cos(gang_rad))
                    on_line = (ang_diff(gang, vec_angle) <= angle_tol_deg) or (glen >= 25.0 and perp_dist <= 20.0)
                    if on_line:
                        if ang_diff(gang, ang2) <= angle_tol_deg:
                            aligned = True
                        elif max(glen, clen) >= 25.0 and min(glen, clen) < 25.0:
                            aligned = True

                if aligned:
                    group_members.append(cand)
                    group_pts.append(cand["pts_xy"])
                    used[j] = True
                    changed = True
                    all_now = np.vstack(group_pts)
                    mean_xy = np.mean(all_now, axis=0)
                    gcx, gcy = float(mean_xy[0]), float(mean_xy[1])
                    cov = np.cov((all_now - mean_xy).T)
                    if cov.ndim == 2:
                        _, eigvecs = np.linalg.eigh(cov)
                        mv = eigvecs[:, 1]
                        gang = math.degrees(math.atan2(float(mv[1]), float(mv[0]))) % 180.0
                        proj = (all_now - mean_xy) @ mv
                        glen = float(np.percentile(proj, 99) - np.percentile(proj, 1))

        all_pts = np.vstack(group_pts)
        merged.append({
            "pts_xy": all_pts,
            "sub_clusters": len(group_members),
            "is_thin_streak": any(m.get("is_thin_streak") for m in group_members),
        })

    return merged


def match_or_update_recent_track(center: Tuple[float, float], angle_deg: float, now_ts: float, cfg: dict) -> Tuple[bool, int, str, dict]:
    """
    Checks if a detected linear streak/object is a continuation of a multi-frame moving object
    (like an airplane or slow satellite crossing the sky over several 30s frames).
    Returns (is_continuation, frame_hit_count, prior_category, trk_dict).
    """
    global _RECENT_TRACKS
    dedup_sec = float(cfg.get("track_dedup_seconds", 180.0))
    dedup_rad = float(cfg.get("track_dedup_radius_px", 450.0))

    # Prune expired tracks
    _RECENT_TRACKS = [t for t in _RECENT_TRACKS if (now_ts - t["last_ts"]) <= dedup_sec]

    cx, cy = center
    for trk in _RECENT_TRACKS:
        # Multi-frame tracks must originate from a prior frame in time (at least 2.0s earlier)
        if (now_ts - trk["last_ts"]) < 2.0:
            continue
        dist = math.hypot(cx - trk["cx"], cy - trk["cy"])
        if 15.0 <= dist <= dedup_rad:
            # Check angle alignment between track displacement and streak orientation
            move_ang = math.degrees(math.atan2(cy - trk["cy"], cx - trk["cx"])) % 180.0
            d_ang = abs(move_ang - angle_deg) % 180.0
            d_ang = min(d_ang, 180.0 - d_ang)
            if d_ang <= 22.0 or abs(angle_deg - trk["angle_deg"]) <= 20.0:
                prev_cat = trk.get("category", "")
                trk["cx"], trk["cy"] = cx, cy
                trk["angle_deg"] = angle_deg
                trk["last_ts"] = now_ts
                trk["hits"] += 1
                return True, trk["hits"], prev_cat, trk

    new_trk = {
        "cx": cx,
        "cy": cy,
        "angle_deg": angle_deg,
        "last_ts": now_ts,
        "hits": 1,
        "category": "",
    }
    _RECENT_TRACKS.append(new_trk)
    return False, 1, "", new_trk


def detect_night_anomalies(
    img: np.ndarray,
    gray: np.ndarray,
    hp: np.ndarray,
    bg_smooth: np.ndarray,
    sky_mask: np.ndarray,
    history_frames: List[dict],
    cfg: dict,
    now_ts: float,
) -> List[Dict[str, Any]]:
    """
    Primary Night-Time Anomaly Engine:
    Detects shooting stars (meteors), bright fireballs, satellites/space objects, aircraft trails,
    optical space flashes, and luminous sky phenomena while canceling out stars, planets, the Moon,
    hot pixels, and drifting clouds.
    """
    h, w = gray.shape[:2]

    # 1. Build dynamic exclusion mask for the Moon, bright planet cores, or exterior roofline floodlight glare
    valid_mask = sky_mask.copy()
    if not history_frames:
        # First frame buffered on startup acts as the baseline star-field reference for subsequent frames
        return []

    prev_gray = history_frames[-1]["gray"]
    prev_bg = cv2.medianBlur(prev_gray, 35)
    cloud_shift = cv2.absdiff(bg_smooth, prev_bg)
    sky_bg_median = float(np.median(bg_smooth[sky_mask > 0])) if np.count_nonzero(sky_mask) > 0 else 45.0

    # 1a. Astronomical Ephemeris Lunar Mask & Backlit Cloud-Scatter Shield
    ephem_info = compute_ephemeris_sun_moon(now_ts, cfg, (h, w))
    moon_alt = ephem_info["moon_alt_deg"]
    moon_phase = ephem_info["moon_phase"]
    mx, my = ephem_info["moon_px"]
    if moon_alt > -6.0 and moon_phase >= 4.0 and mx > -1000:
        # Always punch out the true lunar disk + inner aureole even if partially behind clouds/trees
        r_core = int(160 + 2.2 * moon_phase)
        cv2.circle(valid_mask, (int(mx), int(my)), r_core, 0, -1)
        # Within the wider lunar cloud-scatter halo, suppress any illuminated or shifting cloud bank
        r_scatter = int(320 + 5.5 * moon_phase)
        lunar_zone = np.zeros_like(valid_mask)
        cv2.circle(lunar_zone, (int(mx), int(my)), r_scatter, 255, -1)
        lunar_cloud_glare = (lunar_zone > 0) & ((bg_smooth > sky_bg_median + 6.0) | (cloud_shift > 2))
        if np.any(lunar_cloud_glare):
            lunar_glare_dil = cv2.dilate(
                lunar_cloud_glare.astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
            )
            valid_mask = cv2.bitwise_and(valid_mask, cv2.bitwise_not(lunar_glare_dil))

    sat_cores = (gray > 235).astype(np.uint8) * 255
    sat_prev = (prev_gray > 220)
    sat_prev_dilated = cv2.dilate(sat_prev.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    sat_cores = ((sat_cores > 0) & sat_prev_dilated).astype(np.uint8) * 255
    if np.count_nonzero(sat_cores) > 20:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sat_cores)
        moon_cores = np.zeros_like(sat_cores)
        for i in range(1, num_labels):
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            asp = max(cw, ch) / max(1, min(cw, ch))
            # Do not punch huge 141px holes around tiny pinprick stars (< 12 px area)
            if stats[i, cv2.CC_STAT_AREA] < 12 and asp < 2.0:
                continue
            moon_cores[labels == i] = 255

        if np.count_nonzero(moon_cores) > 20:
            moon_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (141, 141))
            moon_mask = cv2.dilate(moon_cores, moon_dilate)
            valid_mask = cv2.bitwise_and(valid_mask, cv2.bitwise_not(moon_mask))

        # Extra glare buffer around the upper-left house floodlight & roofline sector (x < 2350, y < 1150)
        roof_glare = np.zeros_like(sat_cores)
        roof_glare[:1150, :2100] = sat_cores[:1150, :2100]
        if np.count_nonzero(roof_glare) > 40:
            roof_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (301, 301))
            valid_mask = cv2.bitwise_and(valid_mask, cv2.bitwise_not(cv2.dilate(roof_glare, roof_dilate)))
            # Also suppress illuminated/drifting clouds adjacent to the roofline glare (y < 920, x < 2500)
            roof_cloud = np.zeros_like(valid_mask)
            roof_cloud[:920, :2500] = (
                ((bg_smooth[:920, :2500] > sky_bg_median + 9.0) & (cloud_shift[:920, :2500] > 3))
                | (cloud_shift[:920, :2500] > 5)
            ).astype(np.uint8) * 255
            if np.any(roof_cloud):
                roof_cloud_dil = cv2.dilate(roof_cloud, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
                valid_mask = cv2.bitwise_and(valid_mask, cv2.bitwise_not(roof_cloud_dil))

    # 1b. Mask protruding tree branches (lower-right pine branch, lower-left foliage, extreme bottom horizon)
    valid_mask[2180:2860, 2620:3200] = 0
    valid_mask[2000:, :520] = 0
    valid_mask[3090:, :] = 0

    # 1c. Overcast / Fog Shield: count sharp pinpoint stars in the central sky dome (800:2750, 800:2750)
    dome_hp_sub = hp[800:2750, 800:2750]
    _, star_bin = cv2.threshold(dome_hp_sub, 18, 255, cv2.THRESH_BINARY)
    n_s, _, st_s, _ = cv2.connectedComponentsWithStats(star_bin)
    stars_dome = sum(
        1 for idx_s in range(1, n_s)
        if 1 <= st_s[idx_s, cv2.CC_STAT_AREA] <= 24
        and st_s[idx_s, cv2.CC_STAT_WIDTH] <= 7
        and st_s[idx_s, cv2.CC_STAT_HEIGHT] <= 7
    )
    if stars_dome < 25:
        return []

    # 2. Compute Temporal High-Pass Innovation against previous frame(s) + rolling median
    # Dilating prior high-pass by 9x9 absorbs 30-90s of sidereal star rotation (~2-6 px) + seeing jitter + hot pixels!
    star_guard_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    prev_hp = history_frames[-1]["hp"]
    ref_guard = cv2.dilate(prev_hp, star_guard_kern)
    if len(history_frames) >= 2:
        med_hp = np.median(np.stack([f["hp"] for f in history_frames], axis=0), axis=0).astype(np.uint8)
        ref_guard = np.maximum(ref_guard, cv2.dilate(med_hp, star_guard_kern))
    hp_innovation = cv2.subtract(hp, ref_guard)

    # Strongly suppress diffuse cloud brightenings where the 35x35 smooth background shifted between frames
    cloud_penalty = np.where(cloud_shift > 5, cloud_shift.astype(np.float32) * 1.65, cloud_shift.astype(np.float32) * 0.40)
    hp_innovation = cv2.subtract(hp_innovation, np.clip(cloud_penalty, 0, 255).astype(np.uint8))

    hp_innovation = cv2.bitwise_and(hp_innovation, valid_mask)

    # 3. Adaptive noise floor estimation inside valid sky dome
    mean_Arr, std_Arr = cv2.meanStdDev(hp_innovation, mask=valid_mask)
    noise_mean = float(mean_Arr[0][0])
    noise_std = max(1.5, float(std_Arr[0][0]))

    k_sigma = float(cfg.get("night_k_sigma", 4.8))
    min_peak_delta = float(cfg.get("night_min_peak_delta", 26.0))
    thresh_val = max(min_peak_delta * 0.62, noise_mean + k_sigma * noise_std, 12.0)

    _, bin_img = cv2.threshold(hp_innovation, thresh_val, 255, cv2.THRESH_BINARY)

    # Gentle morphological bridge (5x5 close) to connect closely spaced pixels along a faint streak
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, close_k, iterations=1)

    contours, _ = cv2.findContours(bin_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    raw_clusters = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        roi_bin = bin_closed[y : y + bh, x : x + bw]
        ys_local, xs_local = np.where(roi_bin > 0)
        if len(xs_local) < 8:
            continue
        pts_xy = np.column_stack((xs_local + x, ys_local + y))
        prof = analyze_streak_profile(pts_xy, hp_innovation, gray, img)
        if not prof.get("valid"):
            continue
        if prof["peak_delta"] < min_peak_delta * 0.85:
            continue
        # Reject diffuse cloud puffs before collinear merging so scattered cloud fragments never chain together
        pad_c = 28
        cx1, cy1 = max(0, x - pad_c), max(0, y - pad_c)
        cx2, cy2 = min(w, x + bw + pad_c), min(h, y + bh + pad_c)
        cs_patch = cloud_shift[cy1:cy2, cx1:cx2]
        bg_patch = bg_smooth[cy1:cy2, cx1:cx2]
        if cs_patch.size > 0:
            cs_mean_c = float(np.mean(cs_patch))
            cs_p90_c = float(np.percentile(cs_patch, 90))
            bg_std_c = float(np.std(bg_patch))
            if (cs_mean_c > 7.0 or cs_p90_c > 14.0 or bg_std_c > 14.0) and not (
                prof["peak_delta"] >= 145.0 and prof["width_px"] <= 7.5 and prof["linearity"] >= 0.96
            ):
                continue
        prof["pts_xy"] = pts_xy
        raw_clusters.append(prof)

    # 3b. Thin-Streak Hysteresis Pass (background-leveled frame difference + 3x3 Gaussian ridge detector):
    # Captures steady, moderate-brightness LEO satellite transits (50-250 px long, razor-straight)
    # in calm sky where 9x9 high-pass subtraction attenuates thin ridges.
    prev_gray_f = history_frames[-1]["gray"].astype(np.float32)
    diff_signed = gray.astype(np.float32) - prev_gray_f
    bg_shift = cv2.blur(diff_signed, (51, 51))
    clean_diff = np.clip(diff_signed - bg_shift, 0, 255).astype(np.uint8)
    thin_star_mask = cv2.dilate(
        (prev_hp > 10).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    clean_diff[thin_star_mask > 0] = 0
    # Zero out clean_diff anywhere the 35x35 cloud background shifted (> 3 ADU) so clouds never trigger 3b
    clean_diff[cloud_shift > 3] = 0
    clean_diff = cv2.bitwise_and(clean_diff, valid_mask)
    clean_s = cv2.GaussianBlur(clean_diff, (3, 3), 0)
    _, bin_thin = cv2.threshold(clean_s, 8, 255, cv2.THRESH_BINARY)
    close_k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bin_thin_closed = cv2.morphologyEx(bin_thin, cv2.MORPH_CLOSE, close_k7, iterations=1)
    thin_contours, _ = cv2.findContours(bin_thin_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in thin_contours:
        if cv2.contourArea(cnt) < 40:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if math.hypot(bw, bh) < 50.0:
            continue
        cx_t, cy_t = x + bw * 0.5, y + bh * 0.5
        if any(math.hypot(cx_t - rc["center"][0], cy_t - rc["center"][1]) < 45.0 for rc in raw_clusters):
            continue
        cs_roi = cloud_shift[max(0, y - 20) : min(h, y + bh + 20), max(0, x - 20) : min(w, x + bw + 20)]
        bg_roi = bg_smooth[max(0, y - 20) : min(h, y + bh + 20), max(0, x - 20) : min(w, x + bw + 20)]
        if cs_roi.size > 0 and (float(np.mean(cs_roi)) > 2.5 or float(np.std(bg_roi)) > 5.5):
            continue
        roi_bin = bin_thin_closed[y : y + bh, x : x + bw]
        ys_local, xs_local = np.where(roi_bin > 0)
        if len(xs_local) < 40 or len(xs_local) > 2500:
            continue
        pts_xy = np.column_stack((xs_local + x, ys_local + y))
        prof = analyze_streak_profile(pts_xy, clean_diff, gray, img)
        if not prof.get("valid"):
            continue
        if (
            prof["length_px"] >= 50.0
            and prof["linearity"] >= 0.968
            and prof["aspect"] >= 7.0
            and prof["width_px"] <= 8.5
            and prof.get("fill_ratio", 0.0) >= 0.72
            and prof["peak_delta"] >= 18.0
        ):
            prof["pts_xy"] = pts_xy
            prof["is_thin_streak"] = True
            raw_clusters.append(prof)

    # Merge collinear dot/dash chains (e.g. blinking plane strobes or segmented meteor wake)
    merged_groups = merge_collinear_clusters(raw_clusters, max_dist_px=320.0, angle_tol_deg=12.0)
    bin_combined = cv2.bitwise_or(bin_closed, bin_thin_closed)

    candidates: List[Dict[str, Any]] = []
    meteor_min_len = float(cfg.get("meteor_min_length_px", 36.0))
    meteor_min_aspect = float(cfg.get("meteor_min_aspect", 3.6))
    meteor_min_lin = float(cfg.get("meteor_min_linearity", 0.88))
    space_min_len = float(cfg.get("space_object_min_length_px", 22.0))
    fireball_peak = float(cfg.get("fireball_min_peak_val", 205.0))
    fireball_flux = float(cfg.get("fireball_min_flux", 22000.0))
    flash_min_snr = float(cfg.get("transient_flash_min_snr", 7.5))
    flash_min_delta = float(cfg.get("transient_flash_min_delta", 55.0))

    for grp in merged_groups:
        pts_xy = grp["pts_xy"]
        sub_clusters = grp["sub_clusters"]
        is_thin = bool(grp.get("is_thin_streak"))
        prof = analyze_streak_profile(pts_xy, clean_diff if is_thin else hp_innovation, gray, img)
        if not prof.get("valid"):
            continue

        length_px = prof["length_px"]
        width_px = prof["width_px"]
        aspect = prof["aspect"]
        linearity = prof["linearity"]
        fill_ratio = prof.get("fill_ratio", 1.0)
        peak_delta = prof["peak_delta"]
        peak_raw = prof["peak_raw"]
        flux = prof["flux"]
        strobe_peaks = prof["strobe_peaks"]
        deep_gaps = prof["deep_gaps"]
        snr = (peak_delta - noise_mean) / noise_std

        # Bounding box with padding
        x_min, y_min = np.min(pts_xy, axis=0)
        x_max, y_max = np.max(pts_xy, axis=0)
        bbox = [float(x_min), float(y_min), float(x_max), float(y_max)]

        # Local Cloud-Coherence & PSF Sharpness Veto across candidate neighborhood
        pad_v = 32
        vx1, vy1 = max(0, int(x_min) - pad_v), max(0, int(y_min) - pad_v)
        vx2, vy2 = min(w, int(x_max) + pad_v), min(h, int(y_max) + pad_v)
        local_cs = cloud_shift[vy1:vy2, vx1:vx2]
        local_bg = bg_smooth[vy1:vy2, vx1:vx2]
        local_cs_mean = float(np.mean(local_cs)) if local_cs.size > 0 else 0.0
        local_cs_p90 = float(np.percentile(local_cs, 90)) if local_cs.size > 0 else 0.0
        local_bg_std = float(np.std(local_bg)) if local_bg.size > 0 else 0.0

        if (local_cs_mean > 6.5 or local_cs_p90 > 13.5 or local_bg_std > 13.5) and not (
            peak_delta >= 150.0 and width_px <= 7.5 and linearity >= 0.965 and fill_ratio >= 0.78
        ):
            continue

        # Hough line verification inside bounding box for linear candidates
        has_hough_line = False
        if length_px >= space_min_len:
            pad = 6
            rx1, ry1 = max(0, int(x_min) - pad), max(0, int(y_min) - pad)
            rx2, ry2 = min(w, int(x_max) + pad), min(h, int(y_max) + pad)
            roi_b = bin_combined[ry1:ry2, rx1:rx2]
            lines = cv2.HoughLinesP(
                roi_b,
                1,
                np.pi / 180.0,
                threshold=max(10, int(min(30, length_px * 0.22))),
                minLineLength=max(12, int(length_px * 0.38)),
                maxLineGap=25 if sub_clusters > 1 else 10,
            )
            has_hough_line = lines is not None and len(lines) > 0

        category = None
        subtype = ""
        confidence = 0.0
        active_trk = None

        # Case A: Linear Streak / Track (Meteor, Fireball, Aircraft, Satellite/Space Object)
        if (linearity >= meteor_min_lin and aspect >= meteor_min_aspect and length_px >= space_min_len) or (
            sub_clusters >= 2 and linearity >= 0.92 and length_px >= 45.0
        ):
            if not has_hough_line and linearity < 0.94:
                continue

            is_multi_frame_track, track_hits, prior_category, active_trk = match_or_update_recent_track(prof["center"], prof["angle_deg"], now_ts, cfg)

            # Classify between Aircraft, Fireball, Meteor, and Space Object / Satellite
            min_fill = 0.35 if (sub_clusters >= 2 and linearity >= 0.94) else 0.65
            # A real fireball is a continuous plasma streak (never a sparse chain of disconnected cloud puffs)
            min_fireball_fill = 0.72

            if (
                not is_thin
                and sub_clusters == 1
                and length_px >= 50.0
                and width_px <= 18.0
                and fill_ratio >= min_fireball_fill
                and (peak_raw >= fireball_peak or flux >= fireball_flux)
                and peak_delta >= 110.0
            ):
                category = "fireball"
                subtype = "Bright Fireball / Bolide Streak"
                confidence = min(0.99, 0.72 + 0.03 * snr)
            elif (
                (sub_clusters >= 3 and strobe_peaks >= 2)
                or strobe_peaks >= 3
                or (is_multi_frame_track and strobe_peaks >= 2)
            ):
                if fill_ratio < 0.28 and strobe_peaks < 3:
                    continue
                if not cfg.get("detect_aircraft", True):
                    continue
                category = "aircraft"
                subtype = "Night Aircraft Strobe Trail"
                confidence = min(0.98, 0.55 + 0.04 * snr + 0.05 * min(5, strobe_peaks))
            elif is_multi_frame_track and track_hits >= 2:
                if prior_category in ("fireball", "meteor"):
                    category = "meteor"
                    subtype = "Persistent Meteor Smoke Train / Ionization Wake"
                    confidence = min(0.98, 0.70 + 0.03 * snr)
                else:
                    # Smooth non-strobing object moving steadily across consecutive frames -> Satellite / Orbital Pass
                    category = "space_object"
                    subtype = f"Orbital Satellite Pass (Frame #{track_hits})"
                    confidence = min(0.96, 0.60 + 0.04 * snr)
            elif (
                (length_px >= meteor_min_len or (length_px >= 24.0 and snr >= 25.0 and peak_delta >= 75.0 and aspect >= 4.5))
                and fill_ratio >= min_fill
                and width_px <= 12.0
                and strobe_peaks <= 2
                and deep_gaps <= 2
                and peak_delta >= min_peak_delta
            ):
                category = "meteor"
                subtype = "Shooting Star / Meteor Streak"
                confidence = min(0.97, 0.58 + 0.035 * snr + 0.0015 * length_px)
            elif length_px >= space_min_len and width_px <= 10.5 and fill_ratio >= 0.62 and linearity >= 0.91 and peak_delta >= min_peak_delta:
                category = "space_object"
                subtype = "Satellite / Space Object Flare Streak"
                confidence = min(0.93, 0.52 + 0.035 * snr)
            elif length_px >= 50.0 and width_px <= 8.5 and fill_ratio >= 0.68 and linearity >= 0.965 and aspect >= 6.0 and peak_delta >= max(16.0, min_peak_delta * 0.62):
                category = "space_object"
                subtype = "Orbital Satellite Streak"
                confidence = min(0.92, 0.50 + 0.04 * snr)

        # Case B: High-SNR Compact Point Flash / Geostationary Glint (stellar PSF in calm, cloud-free sky)
        elif (
            history_frames
            and not is_thin
            and snr >= flash_min_snr
            and peak_delta >= flash_min_delta
            and 10 <= prof["area_px"] <= 165
            and width_px <= 8.5
            and aspect <= 2.3
            and local_cs_mean <= 3.8
            and local_bg_std <= 9.0
        ):
            category = "transient_flash"
            subtype = "High-Contrast Optical Sky Flash / Space Glint"
            confidence = min(0.94, 0.50 + 0.04 * snr)

        if category:
            if active_trk is not None:
                active_trk["category"] = category
            alt_deg, az_deg, cardinal = pixel_to_alt_az(prof["center"][0], prof["center"][1], cfg, (h, w))
            candidates.append({
                "category": category,
                "subtype": subtype,
                "conf": round(float(confidence), 3),
                "snr": round(float(snr), 1),
                "bbox": bbox,
                "center": prof["center"],
                "p1": prof["p1"],
                "p2": prof["p2"],
                "length_px": length_px,
                "width_px": prof["width_px"],
                "aspect": aspect,
                "linearity": linearity,
                "fill_ratio": fill_ratio,
                "angle_deg": prof["angle_deg"],
                "strobe_peaks": strobe_peaks,
                "deep_gaps": deep_gaps,
                "chroma_spread": prof.get("chroma_spread", 0.0),
                "peak_delta": peak_delta,
                "peak_raw": peak_raw,
                "flux": flux,
                "alt_deg": alt_deg,
                "az_deg": az_deg,
                "cardinal": cardinal,
            })

    # 4. Macro Phenomenon Check: Aurora / Sudden Sky Flash
    if cfg.get("detect_sky_phenomena", True) and history_frames and not candidates:
        prev_gray = history_frames[-1]["gray"]
        mean_now = float(cv2.mean(gray, mask=valid_mask)[0])
        mean_prev = float(cv2.mean(prev_gray, mask=valid_mask)[0])
        if (mean_now - mean_prev) >= 38.0 and mean_now >= 75.0:
            alt_deg, az_deg, cardinal = 90.0, 0.0, "ZENITH"
            candidates.append({
                "category": "sky_phenomenon",
                "subtype": f"Sudden Full-Sky Illumination Flash (+{mean_now - mean_prev:.0f} lum)",
                "conf": 0.85,
                "snr": round((mean_now - mean_prev) / max(1.0, noise_std), 1),
                "bbox": [w * 0.25, h * 0.25, w * 0.75, h * 0.75],
                "center": (w / 2.0, h / 2.0),
                "length_px": 0.0,
                "aspect": 1.0,
                "linearity": 0.0,
                "peak_delta": round(mean_now - mean_prev, 1),
                "peak_raw": round(mean_now, 1),
                "flux": 0.0,
                "alt_deg": alt_deg,
                "az_deg": az_deg,
                "cardinal": cardinal,
            })

    candidates.sort(
        key=lambda d: (CATEGORY_PRIORITY.get(d["category"], 1), d["snr"], d.get("length_px", 0.0)),
        reverse=True,
    )
    return candidates[:4]


# ---------------------------------------------------------------------------
# Stage-1 Daytime Anomaly Detection Engine
# ---------------------------------------------------------------------------
def get_day_yolo_models(cfg: dict):
    global _YOLO_MODELS
    if not cfg.get("day_enable_yolo", True):
        return None, None
    if "loaded" not in _YOLO_MODELS:
        try:
            from ultralytics import YOLO
            import torch

            device = f"cuda:{cfg.get('gpu_device', 0)}" if torch.cuda.is_available() else "cpu"
            coco_path = cfg.get("yolo_coco_model", "yolo11x.pt")
            if os.path.exists(coco_path):
                logger.info(f"Loading Daytime Sky YOLO11x model ({coco_path}) on {device}...")
                _YOLO_MODELS["coco"] = YOLO(coco_path)
                _YOLO_MODELS["device"] = device
            else:
                _YOLO_MODELS["coco"] = None
                _YOLO_MODELS["device"] = "cpu"
        except Exception as e:
            logger.warning(f"Daytime YOLO initialization skipped: {e}")
            _YOLO_MODELS["coco"] = None
            _YOLO_MODELS["device"] = "cpu"
        _YOLO_MODELS["loaded"] = True
    return _YOLO_MODELS.get("coco"), _YOLO_MODELS.get("device", "cpu")


def detect_day_anomalies(
    img: np.ndarray,
    gray: np.ndarray,
    hp: np.ndarray,
    sky_mask: np.ndarray,
    history_frames: List[dict],
    cfg: dict,
    bg_smooth: Optional[np.ndarray] = None,
    now_ts: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Daytime Sky Anomaly Detector:
    1. Dynamically masks the saturated solar disk + circumsolar glare envelope + astronomical ephem.Sun() disk.
    2. Runs GPU YOLO11x inside the unobstructed sky dome for aircraft, helicopters, birds/raptors, kites/balloons,
       while rejecting false triggers on drifting cumulus/cirrus cloud edges.
    """
    h, w = gray.shape[:2]
    valid_mask = sky_mask.copy()

    # Mask out the Sun and circumsolar glare (pixels > 235 brightness dilated + ephem.Sun position)
    sun_pixels = (gray > 235).astype(np.uint8) * 255
    sun_dil_px = int(cfg.get("day_sun_mask_dilation_px", 210))
    if np.count_nonzero(sun_pixels) > 0:
        k_sun = max(31, sun_dil_px | 1)
        sun_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sun, k_sun))
        sun_glare_mask = cv2.dilate(sun_pixels, sun_kern)
        valid_mask = cv2.bitwise_and(valid_mask, cv2.bitwise_not(sun_glare_mask))

    ephem_info = compute_ephemeris_sun_moon(now_ts, cfg, (h, w))
    if ephem_info.get("sun_alt_deg", -90.0) > -4.0 and ephem_info.get("sun_px") is not None:
        sx, sy = ephem_info["sun_px"]
        cv2.circle(valid_mask, (sx, sy), 340, 0, -1)

    if bg_smooth is None:
        _, bg_smooth = compute_highpass(gray, bg_kernel=35)

    cloud_shift = np.zeros_like(gray, dtype=np.uint8)
    if history_frames:
        prev_gray = history_frames[-1]["gray"]
        prev_bg = cv2.GaussianBlur(prev_gray, (35, 35), 0)
        cloud_shift = cv2.absdiff(bg_smooth, prev_bg)

    candidates: List[Dict[str, Any]] = []

    # 1. GPU YOLO check on masked sky dome (center crop of sky dome for high resolution)
    coco_model, device = get_day_yolo_models(cfg)
    if coco_model is not None:
        try:
            sky_only = cv2.bitwise_and(img, img, mask=sky_mask)
            conf_t = float(cfg.get("day_yolo_conf", 0.38))
            res = coco_model.predict([sky_only], conf=conf_t, imgsz=1280, device=device, verbose=False)[0]
            if res.boxes is not None:
                names = res.names
                for box in res.boxes:
                    cls_id = int(box.cls[0].item())
                    lbl = names[cls_id].lower()
                    if lbl not in ("airplane", "aeroplane", "bird", "kite", "frisbee", "sports ball"):
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    # Must be well inside the valid non-sun sky mask
                    if valid_mask[int(cy), int(cx)] == 0:
                        continue
                    bw, bh = x2 - x1, y2 - y1
                    if bw * bh > (w * h * 0.08):
                        continue
                    # Reject diffuse cloud patches misclassified by YOLO
                    rx1, ry1 = max(0, int(cx - 45)), max(0, int(cy - 45))
                    rx2, ry2 = min(w, int(cx + 45)), min(h, int(cy + 45))
                    local_cs = float(np.mean(cloud_shift[ry1:ry2, rx1:rx2])) if history_frames else 0.0
                    local_bg_s = float(np.std(bg_smooth[ry1:ry2, rx1:rx2]))
                    if local_cs > 12.0 or (local_bg_s > 14.0 and max(bw, bh) > 85.0):
                        continue
                    conf_val = float(box.conf[0].item())
                    alt_deg, az_deg, cardinal = pixel_to_alt_az(cx, cy, cfg, (h, w))
                    if lbl in ("airplane", "aeroplane"):
                        cat, sub = "day_aircraft", "Daytime Aircraft in Sky Dome"
                    elif lbl == "bird":
                        cat, sub = "day_bird", "Soaring Bird / Raptor in Sky Dome"
                    else:
                        cat, sub = "day_anomaly", f"Daytime Aerial Object ({lbl})"

                    candidates.append({
                        "category": cat,
                        "subtype": sub,
                        "conf": round(conf_val, 3),
                        "snr": round(conf_val * 15.0, 1),
                        "bbox": [x1, y1, x2, y2],
                        "center": (cx, cy),
                        "length_px": round(max(bw, bh), 1),
                        "width_px": round(min(bw, bh), 1),
                        "aspect": round(max(bw, bh) / max(1.0, min(bw, bh)), 2),
                        "linearity": 0.0,
                        "peak_delta": 0.0,
                        "peak_raw": float(gray[int(cy), int(cx)]),
                        "flux": 0.0,
                        "alt_deg": alt_deg,
                        "az_deg": az_deg,
                        "cardinal": cardinal,
                    })
        except Exception as e:
            logger.debug(f"Daytime YOLO inference error: {e}")

    candidates.sort(key=lambda d: (CATEGORY_PRIORITY.get(d["category"], 1), d["conf"]), reverse=True)
    return candidates[:3]


# ---------------------------------------------------------------------------
# Stage-2 Local Vision LLM (LM Studio on Quadro RTX 8000) Verification
# ---------------------------------------------------------------------------
def ensure_vlm_server(cfg: dict) -> bool:
    api_url = cfg.get("vlm_api_url", "http://localhost:1234/v1/chat/completions")
    models_url = api_url.replace("/chat/completions", "/models")
    try:
        r = requests.get(models_url, timeout=2)
        if r.status_code == 200 and r.json().get("data"):
            return True
    except Exception:
        pass

    if not cfg.get("vlm_auto_start_lms", True):
        return False

    lms_exe = os.path.expanduser(r"~\.lmstudio\bin\lms.exe")
    if not os.path.exists(lms_exe):
        return False

    preferred = cfg.get("vlm_preferred_model", "qwen3.5-122b-a10b")
    try:
        logger.info(f"Starting LM Studio server and loading vision model '{preferred}'...")
        subprocess.run([lms_exe, "server", "start"], timeout=15, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        subprocess.run([lms_exe, "load", preferred, "--gpu", "max", "-c", "32768", "--parallel", "2", "-y"], timeout=120, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        r = requests.get(models_url, timeout=5)
        return r.status_code == 200
    except Exception as e:
        logger.warning(f"Could not auto-start LM Studio VLM server: {e}")
        return False


def build_vlm_comparison_card(
    curr_img: np.ndarray,
    prev_img: Optional[np.ndarray],
    cand: Dict[str, Any],
) -> np.ndarray:
    """
    Creates a side-by-side 2-panel high-resolution crop (Left: Previous Frame [Before], Right: Current Frame [Event])
    with corner reticle ticks so the Vision LLM can clearly see the transient streak/object without the box covering it.
    """
    h, w = curr_img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in cand["bbox"]]
    cx, cy = int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))
    span = max(x2 - x1, y2 - y1, 260)
    half = int(min(max(span * 1.15, 220), 520))

    cx1 = max(0, cx - half)
    cy1 = max(0, cy - half)
    cx2 = min(w, cx + half)
    cy2 = min(h, cy + half)

    crop_curr = curr_img[cy1:cy2, cx1:cx2].copy()
    if prev_img is not None and prev_img.shape == curr_img.shape:
        crop_prev = prev_img[cy1:cy2, cx1:cx2].copy()
    else:
        crop_prev = np.zeros_like(crop_curr)

    target_sz = 480
    crop_curr_r = cv2.resize(crop_curr, (target_sz, target_sz), interpolation=cv2.INTER_CUBIC)
    crop_prev_r = cv2.resize(crop_prev, (target_sz, target_sz), interpolation=cv2.INTER_CUBIC)

    # Draw subtle corner brackets around the anomaly in the right panel (Event)
    scale_x = target_sz / float(max(1, cx2 - cx1))
    scale_y = target_sz / float(max(1, cy2 - cy1))
    bx1 = max(8, int((x1 - cx1) * scale_x) - 10)
    by1 = max(42, int((y1 - cy1) * scale_y) - 10)
    bx2 = min(target_sz - 8, int((x2 - cx1) * scale_x) + 10)
    by2 = min(target_sz - 8, int((y2 - cy1) * scale_y) + 10)
    cv2.rectangle(crop_curr_r, (bx1, by1), (bx2, by2), (0, 255, 180), 1, cv2.LINE_AA)

    # Top banners
    cv2.rectangle(crop_prev_r, (0, 0), (target_sz, 34), (20, 25, 40), -1)
    cv2.putText(crop_prev_r, "BEFORE (Previous Frame)", (12, 23), cv2.FONT_HERSHEY_DUPLEX, 0.58, (200, 215, 235), 1, cv2.LINE_AA)

    cv2.rectangle(crop_curr_r, (0, 0), (target_sz, 34), (15, 45, 35), -1)
    cv2.putText(crop_curr_r, "EVENT (Current Frame)", (12, 23), cv2.FONT_HERSHEY_DUPLEX, 0.58, (120, 255, 190), 1, cv2.LINE_AA)

    divider = np.full((target_sz, 6, 3), (60, 80, 110), dtype=np.uint8)
    return np.hstack([crop_prev_r, divider, crop_curr_r])


def query_local_vlm_sky(
    curr_img: np.ndarray,
    prev_img: Optional[np.ndarray],
    cand: Dict[str, Any],
    mode: str,
    meta: Dict[str, str],
    cfg: dict,
    similar_examples: Optional[List[Dict[str, Any]]] = None,
) -> Optional[dict]:
    """
    Sends the Before vs. Event comparison crop card + astronomical telemetry to the local Vision LLM
    on the Quadro RTX 8000 to verify and describe the sky event.
    """
    if not cfg.get("vlm_enabled", True):
        return None
    if not ensure_vlm_server(cfg):
        return None

    card = build_vlm_comparison_card(curr_img, prev_img, cand)
    ok, buf = cv2.imencode(".jpg", card, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return None
    b64_img = base64.b64encode(buf.tobytes()).decode("ascii")

    exp_str = meta.get("AS_sEXPOSURE") or f"{float(meta.get('AS_EXPOSURE_US', 0)) / 1e6:.2f}s"
    gain_str = meta.get("AS_GAIN", "N/A")

    few_shot_block = ""
    if similar_examples:
        try:
            import feedback_learner
            few_shot_block = feedback_learner.format_vlm_few_shot_block(similar_examples)
        except Exception:
            few_shot_block = ""

    prompt = (
        f"You are an expert astronomical & all-sky camera analyzer reviewing a {mode} event from a 3552x3552 fisheye Allsky camera.\n"
        f"The attached image shows two side-by-side zoomed crops of the exact same sky sector:\n"
        f"- LEFT: BEFORE (previous frame)\n"
        f"- RIGHT: EVENT (current frame, with a thin green box around the new transient anomaly)\n\n"
        f"Stage-1 Computer Vision Metrics:\n"
        f"- Candidate Type: {cand['subtype']} (category='{cand['category']}')\n"
        f"- Sky Position: Altitude {cand['alt_deg']}°, Azimuth {cand['az_deg']}° ({cand['cardinal']})\n"
        f"- Streak Length: {cand.get('length_px', 0)} px, Aspect Ratio: {cand.get('aspect', 1)}, Linearity: {cand.get('linearity', 0)}, Fill Ratio: {cand.get('fill_ratio', 1)}\n"
        f"- Strobe Peaks Along Track: {cand.get('strobe_peaks', 0)}, Deep Gaps: {cand.get('deep_gaps', 0)}, SNR: {cand.get('snr', 0)}σ\n"
        f"- Camera Exposure: {exp_str}, Gain: {gain_str}\n"
        f"{few_shot_block}\n"
        "Respond ONLY with a valid JSON object:\n"
        "{\n"
        '  "is_interesting": true or false (false ONLY if the green box in the RIGHT panel is clearly just an ordinary cloud edge, static tree branch, house roof, lens flare, or hot pixel noise with no real streak/flash/object),\n'
        '  "category": "meteor" | "fireball" | "space_object" | "aircraft" | "transient_flash" | "sky_phenomenon" | "day_aircraft" | "day_bird" | "day_anomaly" | "false_positive",\n'
        '  "subject": "Short descriptive title, e.g. \'Bright Shooting Star (Meteor)\', \'Fireball / Bolide Streak\', \'LEO Satellite Pass\', \'Night Aircraft Strobe Trail\', \'High-Altitude Space Glint\', \'Soaring Hawk / Bird\'",\n'
        '  "description": "One concise, informative sentence describing what appeared in the sky and its visual characteristics."\n'
        "}\n"
    )

    api_url = cfg.get("vlm_api_url", "http://localhost:1234/v1/chat/completions")
    model_name = cfg.get("vlm_preferred_model", "qwen3.5-122b-a10b")
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                ],
            }
        ],
        "reasoning_effort": "none",
        "temperature": 0.1,
        "max_tokens": 450,
    }

    try:
        t0 = time.time()
        resp = requests.post(api_url, json=payload, timeout=25)
        if resp.status_code != 200:
            logger.warning(f"VLM HTTP {resp.status_code}: {resp.text[:180]}")
            return None
        msg = resp.json()["choices"][0]["message"]
        content = msg.get("content") or msg.get("reasoning_content") or ""
        content_clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        m = re.search(r"\{.*\}", content_clean, flags=re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            parsed["vlm_latency_sec"] = round(time.time() - t0, 2)
            return parsed
    except Exception as e:
        logger.warning(f"VLM sky verification skipped: {e}")
    return None


# ---------------------------------------------------------------------------
# Annotated Snapshot Renderer with Picture-in-Picture (PiP) Corner Insets
# ---------------------------------------------------------------------------
def render_annotated_snapshot(
    curr_img: np.ndarray,
    prev_img: Optional[np.ndarray],
    detections: List[Dict[str, Any]],
    mode: str,
    frame_time_str: str,
    meta: Dict[str, str],
    cfg: dict,
    out_size: int = 1920,
    object_index: int = 1,
    objects_in_frame: int = 1,
) -> np.ndarray:
    """
    Renders a rich 1920x1920 annotated Allsky snapshot:
      - Full fisheye sky view with glowing target reticle + compass/altitude badge
      - Top HUD telemetry bar
      - Bottom-Right Black Corner PiP Inset: 2.5x Zoom of Current Frame (EVENT)
      - Bottom-Left Black Corner PiP Inset: 2.5x Zoom of Previous Frame (BEFORE)
    """
    h0, w0 = curr_img.shape[:2]
    scale = out_size / float(w0)
    canvas = cv2.resize(curr_img, (out_size, out_size), interpolation=cv2.INTER_AREA)

    primary = detections[0]
    cat = primary["category"]
    badge_title, _lvl, color = CATEGORY_BADGES.get(cat, ("🌌 Sky Anomaly", "info", (120, 255, 180)))

    # 1. Draw target reticle & label for each detection on the main dome
    occupied_rects: List[Tuple[int, int, int, int]] = []

    def _overlaps_any(rx1: int, ry1: int, rx2: int, ry2: int) -> bool:
        for ox1, oy1, ox2, oy2 in occupied_rects:
            if not (rx2 < ox1 or rx1 > ox2 or ry2 < oy1 or ry1 > oy2):
                return True
        return False

    for det_idx, det in enumerate(detections):
        is_primary_target = det_idx == 0
        _, _, d_color = CATEGORY_BADGES.get(det["category"], ("", "", (120, 255, 180)))
        adsb_m = det.get("adsb_match")
        orb_m = det.get("orbital_match")
        if orb_m:
            # Distinct Cyan-Gold border for identified orbital satellites / deep-space flashes
            d_color = (255, 225, 0) if not orb_m.get("is_classified") else (235, 115, 255)
        if not is_primary_target and objects_in_frame > 1:
            # Muted context styling for other objects in the same frame (which have their own separate detection cards)
            d_color = (148, 163, 184)

        x1, y1, x2, y2 = [int(round(v * scale)) for v in det["bbox"]]
        cx_b = int(round(0.5 * (x1 + x2)))
        cy_b = int(round(0.5 * (y1 + y2)))
        pad = 18
        min_half = 26
        bx1 = max(10, min(x1 - pad, cx_b - min_half))
        by1 = max(54, min(y1 - pad, cy_b - min_half))
        bx2 = min(out_size - 10, max(x2 + pad, cx_b + min_half))
        by2 = min(out_size - 10, max(y2 + pad, cy_b + min_half))
        occupied_rects.append((bx1, by1, bx2, by2))

        # Outer glow + crisp box around the event so the flash/streak is clearly framed
        box_thick = 3 if (is_primary_target and objects_in_frame > 1) else (1 if not is_primary_target else 2)
        cv2.rectangle(canvas, (bx1 - 1, by1 - 1), (bx2 + 1, by2 + 1), (16, 20, 28), box_thick + 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), d_color, box_thick, cv2.LINE_AA)

        traj_m = adsb_m or orb_m
        traj_color = (148, 163, 184) if (not is_primary_target and objects_in_frame > 1) else ((0, 215, 255) if adsb_m else d_color)
        if traj_m and traj_m.get("seg_start_px") and traj_m.get("seg_end_px"):
            sx1, sy1 = [int(round(v * scale)) for v in traj_m["seg_start_px"]]
            sx2, sy2 = [int(round(v * scale)) for v in traj_m["seg_end_px"]]
            seg_len = math.hypot(sx2 - sx1, sy2 - sy1)
            if seg_len >= 4:
                ux = (sx2 - sx1) / seg_len
                uy = (sy2 - sy1) / seg_len
                if orb_m and not adsb_m:
                    # Draw directional vector outside the target box so the streak/flash inside the box is 100% unobstructed
                    box_rad = max(bx2 - cx_b, by2 - cy_b) + 4
                    ax1 = int(round(cx_b + ux * box_rad))
                    ay1 = int(round(cy_b + uy * box_rad))
                    ax2 = int(round(cx_b + ux * (box_rad + 22)))
                    ay2 = int(round(cy_b + uy * (box_rad + 22)))
                    cv2.arrowedLine(canvas, (ax1, ay1), (ax2, ay2), (16, 20, 28), 4, cv2.LINE_AA, tipLength=0.45)
                    cv2.arrowedLine(canvas, (ax1, ay1), (ax2, ay2), traj_color, 2, cv2.LINE_AA, tipLength=0.45)
                else:
                    # Extend arrowhead slightly ahead of the leading point so the aircraft track is clear
                    hx2 = int(round(sx2 + ux * 18.0))
                    hy2 = int(round(sy2 + uy * 18.0))
                    shaft_layer = canvas.copy()
                    cv2.circle(shaft_layer, (sx1, sy1), 3, traj_color, -1, cv2.LINE_AA)
                    cv2.line(shaft_layer, (sx1, sy1), (hx2, hy2), traj_color, 2, cv2.LINE_AA)
                    cv2.addWeighted(shaft_layer, 0.44, canvas, 0.56, 0, canvas)
                    hx1 = int(round(sx2 - ux * min(10.0, seg_len * 0.3)))
                    hy1 = int(round(sy2 - uy * min(10.0, seg_len * 0.3)))
                    cv2.arrowedLine(canvas, (hx1, hy1), (hx2, hy2), (16, 20, 28), 4, cv2.LINE_AA, tipLength=0.55)
                    cv2.arrowedLine(canvas, (hx1, hy1), (hx2, hy2), traj_color, 2, cv2.LINE_AA, tipLength=0.55)

        header_text = (orb_m.get("box_header") if orb_m else None) or str(det.get("subject") or det["subtype"])
        raw_subj = re.sub(r"[^\x20-\x7E]+", "", header_text).strip()
        if objects_in_frame > 1:
            if is_primary_target:
                raw_subj = f"[OBJECT {object_index}/{objects_in_frame}] {raw_subj}"
            else:
                raw_subj = f"[OTHER OBJECT IN FRAME] {raw_subj[:28]}"
        caption = f"{raw_subj}  |  Alt {det['alt_deg']:.0f}deg {det['cardinal']}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_DUPLEX, 0.56, 1)
        py1 = max(56, by1 - th - 12)
        py2 = py1 + th + 10
        min_x_top = 442 if py2 >= (out_size - 440) else 10
        max_x_top = (out_size - 442 - tw - 16) if py2 >= (out_size - 440) else (out_size - tw - 24)
        px1 = max(min_x_top, min(bx1, max_x_top))
        if _overlaps_any(px1, py1, px1 + tw + 14, py2):
            # Try placing below the box or nudging further up
            alt_py1 = min(out_size - th - 16, by2 + 6)
            alt_py2 = alt_py1 + th + 10
            if not _overlaps_any(px1, alt_py1, px1 + tw + 14, alt_py2):
                py1, py2 = alt_py1, alt_py2
            else:
                py1 = max(56, py1 - (th + 16))
                py2 = py1 + th + 10
        occupied_rects.append((px1, py1, px1 + tw + 14, py2))
        cv2.rectangle(canvas, (px1, py1), (px1 + tw + 14, py2), d_color, -1)
        cv2.putText(canvas, caption, (px1 + 7, py2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.56, (12, 16, 24), 1, cv2.LINE_AA)

        if not is_primary_target and objects_in_frame > 1:
            sub_lbl = ""
        elif adsb_m:
            sub_lbl = re.sub(
                r"[^\x20-\x7E]+",
                "|",
                f"ADS-B: {adsb_m.get('callsign')} ({adsb_m.get('model_name') or adsb_m.get('type_code') or 'Aircraft'}) "
                f"| {int(adsb_m.get('alt_ft', 0)):,} ft | {adsb_m.get('gs_kt', 0):.0f} kt | {adsb_m.get('slant_miles', 0):.1f} mi",
            ).strip()
        elif orb_m:
            sub_lbl = re.sub(
                r"[^\x20-\x7E]+",
                "|",
                f"SAT: {orb_m.get('name')} (NORAD #{orb_m.get('norad_id')}) | {orb_m.get('orbit_regime') or orb_m.get('type_desc')} "
                f"| Alt {orb_m.get('orbit_alt_km', 0):,.0f} km ({orb_m.get('orbit_alt_miles', 0):,.0f} mi) | {orb_m.get('speed_mph', 0):,} mph",
            ).strip()
        elif det["category"] in ("space_object", "transient_flash"):
            sub_lbl = (
                f"ORBITAL / FLASH TAG: Uncataloged High-Altitude Object | "
                f"El {det['alt_deg']:.1f}deg Az {det['az_deg']:.0f}deg ({det['cardinal']}) | SNR {det.get('snr', 0):.1f}sigma"
            )
        else:
            sub_lbl = ""

        if sub_lbl:
            (stw, sth), _ = cv2.getTextSize(sub_lbl, cv2.FONT_HERSHEY_DUPLEX, 0.47, 1)
            spy1 = min(out_size - sth - 16, by2 + 6)
            spy2 = spy1 + sth + 10
            min_x_sub = 442 if spy2 >= (out_size - 440) else 10
            max_x_sub = (out_size - 442 - stw - 16) if spy2 >= (out_size - 440) else (out_size - stw - 20)
            spx1 = max(min_x_sub, min(bx1, max_x_sub))
            if _overlaps_any(spx1, spy1, spx1 + stw + 14, spy2):
                # Place directly above top header pill if space below is occupied by another detection
                alt_spy2 = max(66, py1 - 4)
                alt_spy1 = alt_spy2 - (sth + 10)
                if not _overlaps_any(spx1, alt_spy1, spx1 + stw + 14, alt_spy2):
                    spy1, spy2 = alt_spy1, alt_spy2
            occupied_rects.append((spx1, spy1, spx1 + stw + 14, spy2))
            cv2.rectangle(canvas, (spx1, spy1), (spx1 + stw + 14, spy2), (16, 24, 38), -1)
            cv2.rectangle(canvas, (spx1, spy1), (spx1 + stw + 14, spy2), traj_color, 1)
            cv2.putText(canvas, sub_lbl, (spx1 + 7, spy2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.47, traj_color, 1, cv2.LINE_AA)

    # 2. Extract 2.5x High-Res Zoom Insets centered strictly on `primary` (`detections[0]`)
    px1, py1, px2, py2 = [int(round(v)) for v in primary["bbox"]]
    pcx, pcy = int(round((px1 + px2) / 2.0)), int(round((py1 + py2) / 2.0))
    span = max(px2 - px1, py2 - py1, 220)
    half = int(min(max(span * 0.85, 180), 420))

    rx1, ry1 = max(0, pcx - half), max(0, pcy - half)
    rx2, ry2 = min(w0, pcx + half), min(h0, pcy + half)

    inset_sz = 410
    margin = 22

    # Bottom-Right Inset: EVENT (Current Frame Zoom focused on primary)
    crop_now = curr_img[ry1:ry2, rx1:rx2].copy()
    if crop_now.size > 0:
        zoom_now = cv2.resize(crop_now, (inset_sz, inset_sz), interpolation=cv2.INTER_CUBIC)
        # Draw target box inside zoom_now for the primary target
        for det_z_idx, det_z in enumerate(detections[:1]):
            db1, db2, db3, db4 = [int(round(v)) for v in det_z["bbox"]]
            if db3 < rx1 or db1 > rx2 or db4 < ry1 or db2 > ry2:
                continue
            zx1 = int((db1 - rx1) * (inset_sz / float(max(1, rx2 - rx1)))) - 12
            zy1 = int((db2 - ry1) * (inset_sz / float(max(1, ry2 - ry1)))) - 12
            zx2 = int((db3 - rx1) * (inset_sz / float(max(1, rx2 - rx1)))) + 12
            zy2 = int((db4 - ry1) * (inset_sz / float(max(1, ry2 - ry1)))) + 12
            zcx, zcy = (zx1 + zx2) // 2, (zy1 + zy2) // 2
            zx1, zy1 = min(zx1, zcx - 24), min(zy1, zcy - 24)
            zx2, zy2 = max(zx2, zcx + 24), max(zy2, zcy + 24)
            _, _, det_z_col = CATEGORY_BADGES.get(det_z["category"], ("", "", (120, 255, 180)))
            if det_z.get("orbital_match"):
                det_z_col = (255, 225, 0)
            cv2.rectangle(
                zoom_now,
                (max(6, zx1), max(36, zy1)),
                (min(inset_sz - 6, zx2), min(inset_sz - 6, zy2)),
                det_z_col,
                2,
                cv2.LINE_AA,
            )
            z_tag = (
                f"SAT: {det_z['orbital_match']['name']} [#{det_z['orbital_match']['norad_id']}]"
                if det_z.get("orbital_match")
                else ("METEOR" if det_z["category"] == "meteor" else det_z["category"].upper())
            )
            ty_z = max(48, zy1 - 5) if zy1 > 54 else min(inset_sz - 10, zy2 + 16)
            cv2.putText(
                zoom_now,
                re.sub(r"[^\x20-\x7E]+", "", z_tag)[:34],
                (max(8, min(inset_sz - 180, zx1)), ty_z),
                cv2.FONT_HERSHEY_DUPLEX,
                0.42,
                det_z_col,
                1,
                cv2.LINE_AA,
            )
        z_col = (255, 225, 0) if primary.get("orbital_match") else color
        cv2.rectangle(zoom_now, (0, 0), (inset_sz, 32), (18, 24, 38), -1)
        obj_tag = f"OBJ {object_index}/{objects_in_frame} " if objects_in_frame > 1 else ""
        if primary.get("orbital_match"):
            z_title = f"2.5x ZOOM: {obj_tag}{primary['orbital_match']['name']} (#{primary['orbital_match']['norad_id']})"
        elif primary.get("adsb_match"):
            z_title = f"2.5x ZOOM: {obj_tag}{primary['adsb_match'].get('callsign') or 'AIRCRAFT'}"
        else:
            z_title = f"2.5x ZOOM: {obj_tag}{primary['category'].upper()} ({primary.get('length_px', 0):.0f}px, {primary.get('snr', 0):.1f}sigma)"
        cv2.putText(
            zoom_now,
            re.sub(r"[^\x20-\x7E]+", "", z_title)[:44],
            (10, 22),
            cv2.FONT_HERSHEY_DUPLEX,
            0.48,
            z_col,
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(zoom_now, (0, 0), (inset_sz - 1, inset_sz - 1), z_col, 2)

        iy1 = out_size - inset_sz - margin
        ix1 = out_size - inset_sz - margin
        canvas[iy1 : iy1 + inset_sz, ix1 : ix1 + inset_sz] = zoom_now

    # Bottom-Left Inset: BEFORE (Previous Frame Zoom)
    if prev_img is not None and prev_img.shape == curr_img.shape:
        crop_prev = prev_img[ry1:ry2, rx1:rx2].copy()
        if crop_prev.size > 0:
            zoom_prev = cv2.resize(crop_prev, (inset_sz, inset_sz), interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(zoom_prev, (0, 0), (inset_sz, 32), (18, 24, 38), -1)
            cv2.putText(
                zoom_prev,
                "2.5x ZOOM: BEFORE (Prior Frame)",
                (10, 22),
                cv2.FONT_HERSHEY_DUPLEX,
                0.50,
                (180, 195, 215),
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(zoom_prev, (0, 0), (inset_sz - 1, inset_sz - 1), (100, 116, 139), 2)

            iy1 = out_size - inset_sz - margin
            ix1 = margin
            canvas[iy1 : iy1 + inset_sz, ix1 : ix1 + inset_sz] = zoom_prev

    # 3. Top HUD Header Bar
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (out_size, 48), (12, 18, 32), -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)

    exp_s = meta.get("AS_sEXPOSURE") or f"{float(meta.get('AS_EXPOSURE_US', 0)) / 1e6:.2f}s"
    gain_s = meta.get("AS_GAIN", "0")
    hud_left = f"BLUFFSKY ALLSKY ({mode})  |  {frame_time_str}  |  Exp: {exp_s}  Gain: {gain_s}"
    hud_obj = f"OBJ {object_index}/{objects_in_frame}  |  " if objects_in_frame > 1 else ""
    hud_right = f"{hud_obj}Alt {primary['alt_deg']:.1f}deg  Az {primary['az_deg']:.0f}deg ({primary['cardinal']})"
    cv2.putText(canvas, hud_left, (18, 32), cv2.FONT_HERSHEY_DUPLEX, 0.68, (241, 245, 249), 1, cv2.LINE_AA)
    (rw, _), _ = cv2.getTextSize(hud_right, cv2.FONT_HERSHEY_DUPLEX, 0.68, 1)
    cv2.putText(canvas, hud_right, (out_size - rw - 18, 32), cv2.FONT_HERSHEY_DUPLEX, 0.68, color, 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Discord Alert Dispatch (to channel 'allsky' via E:\discord_messages)
# ---------------------------------------------------------------------------
def is_remarkable_discord_space_event(event: Dict[str, Any], cfg: dict) -> Tuple[bool, str]:
    """
    Filters Discord notifications so ONLY remarkable space & astronomical events
    (fireballs, prominent meteors, space stations, classified/military satellites,
    rocket bodies, deep-space MEO/GEO/HEO glints, and high-SNR uncataloged transients)
    ping Discord, while all routine events (aircraft, daytime, routine Starlinks) are
    still logged to the home website.
    """
    if not cfg.get("discord_remarkable_space_only", True):
        return True, "all_enabled"

    category = str(event.get("category") or "")
    mode = str(event.get("mode") or "NIGHT")
    snr = float(event.get("snr") or 0.0)
    length_px = float(event.get("length_px") or 0.0)
    linearity = float(event.get("linearity") or 0.0)
    alt_deg = float(event.get("alt_deg") or 0.0)

    # 1. Never ping Discord for daytime events or aircraft
    if mode != "NIGHT" or category in ("aircraft", "day_aircraft", "day_bird", "day_anomaly"):
        return False, f"routine category '{category}' ({mode})"

    feat = {}
    raw_fj = event.get("features_json")
    if isinstance(raw_fj, str) and raw_fj.strip():
        try:
            feat = json.loads(raw_fj)
        except Exception:
            feat = {}
    elif isinstance(raw_fj, dict):
        feat = raw_fj

    if feat.get("adsb_flight"):
        return False, "ADS-B aircraft transit (logged to website only)"

    orb = feat.get("orbital_object") or {}
    orb_name = str(orb.get("name") or event.get("subject") or "").upper()
    orb_type = str(orb.get("obj_type") or "").lower()
    orb_alt_km = float(orb.get("orbit_alt_km") or 0.0)

    # 2. Suppress routine commercial mega-constellations (Starlink, OneWeb, Kuiper, Qianfan) on Discord
    if orb_type == "starlink" or any(k in orb_name for k in ("STARLINK", "ONEWEB", "KUIPER", "QIANFAN")):
        return False, f"routine commercial constellation pass ({orb.get('name', 'Constellation')})"

    # 3. Fireballs / Bolides are always remarkable
    if category == "fireball":
        return True, "bright fireball / bolide"

    # 4. Natural Meteors / Shooting Stars: require crisp, unmistakable trajectory
    if category == "meteor":
        if snr >= 10.0 and length_px >= 35.0 and linearity >= 0.94 and alt_deg >= 14.0:
            return True, f"prominent meteor streak (len={length_px:.0f}px, SNR={snr:.1f}σ)"
        return False, f"faint/short meteor candidate (len={length_px:.0f}px, SNR={snr:.1f}σ — website only)"

    # 5. Space Objects (Orbital streaks)
    if category == "space_object":
        if orb:
            if orb_type in ("station", "classified", "rocket_body") or orb_alt_km >= 1400.0:
                return True, f"remarkable cataloged orbital object ({orb.get('name')} [{orb_type}, {orb_alt_km:.0f}km])"
            if snr >= 14.0 and length_px >= 50.0:
                return True, f"bright cataloged satellite pass ({orb.get('name')}, SNR={snr:.1f}σ)"
            return False, f"routine LEO satellite pass ({orb.get('name')} — website only)"
        else:
            # Uncataloged space object streak
            if snr >= 14.0 and length_px >= 45.0 and linearity >= 0.96 and alt_deg >= 18.0:
                return True, f"high-SNR uncataloged space streak (len={length_px:.0f}px, SNR={snr:.1f}σ)"
            return False, f"faint uncataloged streak (len={length_px:.0f}px, SNR={snr:.1f}σ — website only)"

    # 6. Transient Flashes & Sky Phenomena
    if category in ("transient_flash", "sky_phenomenon"):
        w_px = float(feat.get("width_px") or event.get("width_px") or 99.0)
        if orb:
            if (orb_type == "classified" or orb_alt_km >= 1500.0) and snr >= 14.0 and w_px <= 6.5:
                return True, f"deep-space / classified orbital flash ({orb.get('name')}, {orb_alt_km:.0f}km)"
            return False, f"routine LEO specular glint ({orb.get('name')} — website only)"
        else:
            if snr >= 18.0 and alt_deg >= 20.0 and w_px <= 5.5 and length_px <= 14.0:
                return True, f"sharp high-SNR uncataloged optical flash (SNR={snr:.1f}σ, w={w_px:.1f}px)"
            return False, f"diffuse/moderate flash (SNR={snr:.1f}σ, w={w_px:.1f}px — website only)"

    return False, f"non-remarkable category '{category}'"


def send_discord_sky_alert(event: Dict[str, Any], cfg: dict) -> bool:
    if not cfg.get("send_discord_alerts", True):
        return False

    category = event["category"]
    if category == "aircraft" and not cfg.get("alert_on_aircraft", False):
        logger.info("Skipping Discord alert for night aircraft (alert_on_aircraft=False).")
        return False

    is_rem, rem_reason = is_remarkable_discord_space_event(event, cfg)
    if not is_rem:
        logger.info(f"Skipping Discord alert (logged to website only): {rem_reason}")
        return False

    cooldowns = cfg.get("cooldown_seconds") or {}
    cd_sec = int(cooldowns.get(category, 60))
    now = time.time()
    if category in _LAST_ALERT_TIMES and (now - _LAST_ALERT_TIMES[category]) < cd_sec:
        logger.info(f"Cooldown active for '{category}' ({int(now - _LAST_ALERT_TIMES[category])}s < {cd_sec}s), skipping Discord ping.")
        return False

    badge_title, level, _ = CATEGORY_BADGES.get(category, ("🌌 Sky Anomaly Detected", "info", (120, 255, 180)))
    title = f"{badge_title} — {event.get('subject') or event['subtype']}"

    lines = []
    if event.get("vlm_description"):
        lines.append(f"**AI Sky Analysis:** {event['vlm_description']}")
    lines.append(
        f"**Sky Coordinates:** Altitude `{event['alt_deg']}°` • Azimuth `{event['az_deg']}° ({event['cardinal']})`"
    )
    if event.get("length_px", 0) > 0:
        lines.append(
            f"**Telemetry:** Streak `{event['length_px']} px` • SNR `{event['snr']}σ` • Linearity `{event.get('linearity', 0):.2f}`"
        )
    else:
        lines.append(f"**Telemetry:** SNR `{event['snr']}σ` • Peak Δ `{event.get('peak_delta', 0)}`")

    lines.append(
        f"**Capture:** `{event['mode']}` Mode • Exp `{event.get('exposure_str', 'N/A')}` • Gain `{event.get('gain_str', 'N/A')}`"
    )
    lines.append(f"**Time:** {event['frame_time_display']} (`{event['frame_filename']}`)")
    lines.append(
        "💬 *Reply `plane`, `meteor`, `fireball`, `satellite`, `flash`, or `false` (or react ☄️ 🛰️ ✈️ ❌) to train AI*"
    )

    channel = cfg.get("discord_channel", "allsky")
    payload = {
        "app": "BluffSky Allsky AI",
        "channel": channel,
        "title": title,
        "message": "\n".join(lines),
        "level": level,
        "file_path": event.get("snapshot_path"),
    }

    alert_url = cfg.get("discord_alert_url", "http://localhost:8090/alert")
    try:
        discord_dir = r"E:\discord_messages"
        if os.path.isdir(discord_dir):
            if discord_dir not in sys.path:
                sys.path.insert(0, discord_dir)
            from discord_server import send_to_discord
            ok, _code, info = send_to_discord(payload)
            if ok:
                if payload.get("discord_message_id"):
                    event["discord_message_id"] = str(payload["discord_message_id"])
                _LAST_ALERT_TIMES[category] = now
                logger.info(f"Sent Allsky Discord alert (channel='{channel}'): {title}")
                return True
            logger.warning(f"Direct send_to_discord returned ({_code}): {info}, trying HTTP POST")

        webhook_url = str(cfg.get("discord_webhook_url") or cfg.get("allsky_webhook_url") or "").strip()
        if webhook_url.startswith("https://discord"):
            snap_fp = event.get("snapshot_path")
            embed = {"title": title, "description": "\n".join(lines), "color": 3447003}
            if snap_fp and os.path.isfile(snap_fp):
                with open(snap_fp, "rb") as sf:
                    w_resp = requests.post(
                        webhook_url + ("&wait=true" if "?" in webhook_url else "?wait=true"),
                        data={"payload_json": json.dumps({"username": "BluffSky Allsky AI", "embeds": [embed]})},
                        files={"file": (os.path.basename(snap_fp), sf, "image/jpeg")},
                        timeout=20,
                    )
            else:
                w_resp = requests.post(
                    webhook_url + ("&wait=true" if "?" in webhook_url else "?wait=true"),
                    json={"username": "BluffSky Allsky AI", "embeds": [embed]},
                    timeout=15,
                )
            if 200 <= w_resp.status_code < 300:
                try:
                    event["discord_message_id"] = str(w_resp.json().get("id") or "")
                except Exception:
                    pass
                _LAST_ALERT_TIMES[category] = now
                logger.info(f"Sent Allsky Discord alert via direct Webhook: {title}")
                return True

        resp = requests.post(alert_url, json=payload, timeout=15)
        if 200 <= resp.status_code < 300:
            _LAST_ALERT_TIMES[category] = now
            logger.info(f"Sent Allsky Discord alert via HTTP (channel='{channel}'): {title}")
            return True
        logger.error(f"Discord bridge HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"Failed sending Allsky Discord alert: {e}")
    return False


# ---------------------------------------------------------------------------
# End-to-End Single Frame Processing Pipeline
# ---------------------------------------------------------------------------
def process_frame_array(
    img: np.ndarray,
    frame_filename: str,
    meta: Dict[str, str],
    buffer: RollingSkyBuffer,
    cfg: dict,
    send_discord: bool = True,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Runs the full Allsky Anomaly Detection pipeline on a decoded BGR image (3552x3552).
    Updates the RollingSkyBuffer and returns an event dict if an interesting sky event is found.
    """
    t0 = time.time()
    history = load_history()
    cam_id = str(cfg.get("camera_id") or "cam1")
    cam_name = str(cfg.get("camera_name") or f"Allsky Camera ({cam_id})")
    hist_frame_key = frame_filename if cam_id == "cam1" else f"{cam_id}_{frame_filename}"

    if not force and hist_frame_key in history.get("processed_frames", {}):
        return None

    h, w = img.shape[:2]
    sky_mask = ensure_sky_mask(cfg, (h, w))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    bg_kernel = int(cfg.get("night_bg_kernel", 35))
    hp, bg_smooth = compute_highpass(gray, bg_kernel=bg_kernel)

    mode = infer_day_or_night(img, sky_mask, meta)
    compat_history = buffer.get_compatible_history(mode, (h, w))
    prev_img = compat_history[-1]["img"] if compat_history else None

    try:
        exp_sec = max(1.0, float(meta.get("AS_EXPOSURE_US", 20000000 if mode == "NIGHT" else 50000)) / 1e6)
    except Exception:
        exp_sec = 20.0 if mode == "NIGHT" else 0.5
    exp_start_ts, exp_end_ts = get_frame_exposure_window(frame_filename, exp_sec, t0, overlay_meta=meta)
    frame_mid_ts = 0.5 * (exp_start_ts + exp_end_ts)

    ephem_info = compute_ephemeris_sun_moon(frame_mid_ts, cfg, (h, w))
    if not meta.get("AS_MOON_ELEVATION") and ephem_info.get("moon_alt_deg") is not None:
        meta["AS_MOON_ELEVATION"] = f"{ephem_info['moon_alt_deg']:.1f}"
    if not meta.get("AS_MOON_ILLUMINATION") and ephem_info.get("moon_phase_pct") is not None:
        meta["AS_MOON_ILLUMINATION"] = f"{ephem_info['moon_phase_pct']:.0f}"

    candidates: List[Dict[str, Any]] = []
    if mode == "NIGHT" and cfg.get("night_detection_enabled", True):
        candidates = detect_night_anomalies(
            img=img,
            gray=gray,
            hp=hp,
            bg_smooth=bg_smooth,
            sky_mask=sky_mask,
            history_frames=compat_history,
            cfg=cfg,
            now_ts=frame_mid_ts,
        )
    elif mode == "DAY" and cfg.get("day_detection_enabled", True):
        candidates = detect_day_anomalies(
            img=img,
            gray=gray,
            hp=hp,
            sky_mask=sky_mask,
            history_frames=compat_history,
            cfg=cfg,
            bg_smooth=bg_smooth,
            now_ts=frame_mid_ts,
        )

    # Always push current frame into the rolling buffer for the next frame's comparison
    buffer.push(frame_filename, mode, img, gray, hp, ts=t0)

    mean_sky = round(float(cv2.mean(gray, mask=sky_mask)[0]), 1)
    exp_str = meta.get("AS_sEXPOSURE") or f"{float(meta.get('AS_EXPOSURE_US', 0)) / 1e6:.3f}s"
    gain_str = str(meta.get("AS_GAIN", "0"))

    adsb_summary = {}
    if cfg.get("adsb_enabled", True):
        try:
            import adsb_tracker
            snap_adsb = adsb_tracker.get_live_adsb_snapshot()
            adsb_summary = {
                "adsb_receiver_ok": snap_adsb.get("local_receiver_ok", False),
                "adsb_in_range": snap_adsb.get("aircraft_in_range_count", 0),
                "adsb_in_dome": snap_adsb.get("aircraft_in_dome_count", 0),
                "adsb_flights": [
                    f"{a.get('callsign')} ({int(a.get('alt_ft', 0)):,}ft, {a.get('alt_deg', 0):.0f}° {a.get('cardinal')})"
                    for a in snap_adsb.get("aircraft_in_dome", [])[:5]
                ],
            }
        except Exception as e:
            logger.debug(f"ADS-B status snapshot skipped: {e}")

    status_payload = {
        "camera_id": cam_id,
        "camera_name": cam_name,
        "last_frame": frame_filename,
        "mode": mode,
        "mean_sky_brightness": mean_sky,
        "exposure": exp_str,
        "gain": gain_str,
        "temperature_f": meta.get("AS_TEMPERATURE_F", ""),
        "moon_elevation": meta.get("AS_MOON_ELEVATION", ""),
        "moon_illumination": meta.get("AS_MOON_ILLUMINATION", ""),
        "buffer_depth": len(compat_history) + 1,
    }
    status_payload.update(adsb_summary)
    try:
        cur_ls = {}
        if os.path.exists(LIVE_STATUS_FILE):
            with open(LIVE_STATUS_FILE, "r", encoding="utf-8") as lf:
                cur_ls = json.load(lf)
        cam_map = cur_ls.get("cameras") if isinstance(cur_ls.get("cameras"), dict) else {}
        cam_map[cam_id] = dict(status_payload, updated_at=time.time(), updated_iso=datetime.now().isoformat())
        status_payload["cameras"] = cam_map
    except Exception:
        pass
    update_live_status(status_payload)

    if not candidates:
        history.setdefault("processed_frames", {})[hist_frame_key] = {
            "status": "clean",
            "camera_id": cam_id,
            "mode": mode,
            "timestamp": t0,
        }
        save_history(history)
        return None

    # Stage 1.2: Live ADS-B Aircraft Transponder Trajectory Matching (ADBSTracker 192.168.1.50)
    # Prevents aircraft from ever being misclassified as meteors/satellites and labels the exact flight!
    if cfg.get("adsb_enabled", True):
        try:
            import adsb_tracker
            exp_start_ts, exp_end_ts = get_frame_exposure_window(frame_filename, exp_sec, t0, overlay_meta=meta)
            for cand in candidates:
                m_ac = adsb_tracker.match_detection_to_adsb(
                    cand,
                    cfg=cfg,
                    exposure_sec=exp_sec,
                    frame_timestamp=exp_end_ts,
                    exposure_start_ts=exp_start_ts,
                    exposure_end_ts=exp_end_ts,
                    shape=(h, w),
                )
                if m_ac:
                    cand["adsb_match"] = m_ac
                    cand["category"] = "aircraft" if mode == "NIGHT" else "day_aircraft"
                    cand["subtype"] = m_ac.get("subject_line") or f"ADS-B Flight {m_ac.get('callsign')}"
                    cand["subject"] = m_ac.get("subject_line") or cand["subtype"]
                    cand["adsb_description"] = m_ac.get("description_line") or ""
                    cand["conf"] = 0.99
                    logger.info(
                        f"[{mode}] {frame_filename}: Matched sky detection at ({cand['alt_deg']}°, {cand['az_deg']}°) "
                        f"to live ADS-B flight {m_ac.get('callsign')} ({m_ac.get('model_name')}) "
                        f"[dist={m_ac.get('match_dist_px')}px, sep={m_ac.get('match_ang_deg')}°]!"
                    )

            # If a long night strobe trail was split into multiple candidates that matched the same aircraft
            # (or if an un-matchedaircraft fragment is collinear with a matched aircraft), merge them into one box:
            if len(candidates) > 1:
                consolidated: List[Dict[str, Any]] = []
                for cand in candidates:
                    m_ac = cand.get("adsb_match")
                    merged_into_existing = False
                    for prev in consolidated:
                        pm_ac = prev.get("adsb_match")
                        same_hex = m_ac and pm_ac and m_ac.get("hex") and m_ac.get("hex") == pm_ac.get("hex")
                        collinear_frag = (
                            pm_ac
                            and not m_ac
                            and cand.get("category") in ("aircraft", "space_object", "meteor")
                            and prev.get("angle_deg") is not None
                            and cand.get("angle_deg") is not None
                            and min(abs(prev["angle_deg"] - cand["angle_deg"]) % 180.0, 180.0 - (abs(prev["angle_deg"] - cand["angle_deg"]) % 180.0)) <= 12.0
                            and math.hypot(cand["center"][0] - prev["center"][0], cand["center"][1] - prev["center"][1]) <= 420.0
                        )
                        if same_hex or collinear_frag:
                            b1, b2 = prev["bbox"], cand["bbox"]
                            prev["bbox"] = [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]
                            cx_u = 0.5 * (prev["bbox"][0] + prev["bbox"][2])
                            cy_u = 0.5 * (prev["bbox"][1] + prev["bbox"][3])
                            prev["center"] = (cx_u, cy_u)
                            prev["alt_deg"], prev["az_deg"], prev["cardinal"] = pixel_to_alt_az(cx_u, cy_u, cfg, (h, w))
                            prev["length_px"] = round(math.hypot(prev["bbox"][2] - prev["bbox"][0], prev["bbox"][3] - prev["bbox"][1]), 1)
                            # Re-orient trajectory arrow across the full combined trail span along ADS-B velocity vector
                            adsb_s = pm_ac.get("adsb_seg_start_px") or pm_ac.get("seg_start_px")
                            adsb_e = pm_ac.get("adsb_seg_end_px") or pm_ac.get("seg_end_px")
                            pts_all = [p for c_item in (prev, cand) for p in (c_item.get("p1"), c_item.get("p2"), c_item.get("center")) if p]
                            if len(pts_all) >= 2 and adsb_s and adsb_e:
                                vx = float(adsb_e[0] - adsb_s[0])
                                vy = float(adsb_e[1] - adsb_s[1])
                                pts_sorted = sorted(pts_all, key=lambda pt: pt[0] * vx + pt[1] * vy)
                                prev["p1"] = pts_sorted[0]
                                prev["p2"] = pts_sorted[-1]
                                pm_ac["seg_start_px"] = [int(round(pts_sorted[0][0])), int(round(pts_sorted[0][1]))]
                                pm_ac["seg_end_px"] = [int(round(pts_sorted[-1][0])), int(round(pts_sorted[-1][1]))]
                            merged_into_existing = True
                            break
                    if not merged_into_existing:
                        consolidated.append(cand)
                candidates = consolidated
        except Exception as e:
            logger.warning(f"ADS-B candidate matching error: {e}")

    # Suppress any stray point-flash candidate that lies inside or adjacent to a detected aircraft box
    if len(candidates) > 1:
        ac_boxes = [c["bbox"] for c in candidates if c.get("adsb_match") or c.get("category") in ("aircraft", "day_aircraft")]
        if ac_boxes:
            filtered_cands = []
            for cand in candidates:
                if not cand.get("adsb_match") and cand.get("category") == "transient_flash" and cand.get("length_px", 0) < 22.0:
                    cx_f, cy_f = cand["center"]
                    inside_ac = any(
                        (ab[0] - 65 <= cx_f <= ab[2] + 65) and (ab[1] - 65 <= cy_f <= ab[3] + 65)
                        for ab in ac_boxes
                    )
                    if inside_ac:
                        continue
                filtered_cands.append(cand)
            if filtered_cands:
                candidates = filtered_cands

    # Stage 1.3: Orbital Satellite, Spent Rocket Body & Space Glint SGP4 Matching (CelesTrak + McCants Classified)
    if mode == "NIGHT" and cfg.get("orbital_tracking_enabled", True):
        try:
            import orbital_tracker
            exp_start_ts, exp_end_ts = get_frame_exposure_window(frame_filename, exp_sec, t0, overlay_meta=meta)
            for cand in candidates:
                if cand.get("adsb_match") or cand.get("strobe_peaks", 0) >= 4:
                    continue
                m_sat = orbital_tracker.match_detection_to_satellite(
                    cand,
                    cfg=cfg,
                    exposure_sec=exp_sec,
                    frame_timestamp=exp_end_ts,
                    exposure_start_ts=exp_start_ts,
                    exposure_end_ts=exp_end_ts,
                    shape=(h, w),
                )
                if m_sat:
                    cand["orbital_match"] = m_sat
                    if m_sat.get("match_mode") == "streak":
                        cand["category"] = "space_object"
                    else:
                        cand["category"] = "transient_flash"
                    cand["subtype"] = m_sat.get("subject_line") or f"Satellite {m_sat.get('name')}"
                    cand["subject"] = m_sat.get("subject_line") or cand["subtype"]
                    cand["orbital_description"] = m_sat.get("description_line") or ""
                    cand["conf"] = 0.96
                    logger.info(
                        f"[{mode}] {frame_filename}: Matched sky {m_sat.get('match_mode')} at ({cand['alt_deg']}°, {cand['az_deg']}°) "
                        f"to orbital object {m_sat.get('name')} (NORAD #{m_sat.get('norad_id')} — {m_sat.get('type_desc')}) "
                        f"[dist={m_sat.get('match_dist_px')}px, sep={m_sat.get('match_ang_deg')}°, alt={m_sat.get('orbit_alt_km')}km]!"
                    )
        except Exception as e:
            logger.warning(f"Orbital satellite matching error: {e}")

    # Stage 1.5 & Stage 2: Evaluate each detected candidate independently so multi-object frames
    # emit each verified object as its own reported event for independent human review & reclassification.
    verified_candidates: List[Dict[str, Any]] = []
    last_filter_status: Optional[Dict[str, Any]] = None

    for cand in candidates:
        similar_examples = []
        fb_eval = None
        try:
            import feedback_learner
            fb_eval = feedback_learner.evaluate_candidate_with_feedback(cand, mode=mode)
            if fb_eval.get("veto_as_false_positive") and not cand.get("adsb_match") and not cand.get("orbital_match"):
                logger.info(f"[{mode}] {frame_filename}: Candidate suppressed by Active Learning memory ({fb_eval.get('veto_reason')}).")
                last_filter_status = {
                    "status": "feedback_filtered",
                    "mode": mode,
                    "reason": fb_eval.get("veto_reason"),
                    "timestamp": t0,
                }
                continue
            similar_examples = fb_eval.get("similar_examples") or []
            if not cand.get("adsb_match") and not cand.get("orbital_match") and fb_eval.get("suggested_category") and fb_eval.get("knn_confidence", 0.0) >= 0.72:
                sug_cat = fb_eval["suggested_category"]
                if sug_cat in CATEGORY_PRIORITY:
                    logger.info(
                        f"[{mode}] {frame_filename}: Active Learning k-NN refined Stage-1 '{cand['category']}' -> '{sug_cat}' "
                        f"(conf={fb_eval['knn_confidence']})"
                    )
                    cand["category"] = sug_cat
                    cand["subtype"] = fb_eval.get("suggested_subject") or cand["subtype"]
                    cand["subject"] = fb_eval.get("suggested_subject") or cand.get("subject")
        except Exception as e:
            logger.debug(f"Feedback learner evaluation skipped: {e}")

        vlm_res = None
        if cand.get("adsb_match"):
            # Transponder-verified aircraft — no need to risk VLM misclassifying as meteor or vetoing
            vlm_res = {
                "is_interesting": True,
                "category": cand["category"],
                "subject": cand["subject"],
                "description": cand.get("adsb_description"),
            }
        elif cand.get("orbital_match"):
            # SGP4-verified sunlit orbital satellite / rocket body / classified object
            vlm_res = {
                "is_interesting": True,
                "category": cand["category"],
                "subject": cand["subject"],
                "description": cand.get("orbital_description"),
            }
        else:
            vlm_res = query_local_vlm_sky(
                curr_img=img,
                prev_img=prev_img,
                cand=cand,
                mode=mode,
                meta=meta,
                cfg=cfg,
                similar_examples=similar_examples,
            )

        if vlm_res and not cand.get("adsb_match") and not cand.get("orbital_match"):
            if vlm_res.get("is_interesting") is False or vlm_res.get("category") == "false_positive":
                is_pristine_thin_streak = (
                    cand.get("linearity", 0.0) >= 0.985
                    and cand.get("width_px", 99.0) <= 5.5
                    and cand.get("local_cloud_shift", 0.0) < 4.0
                )
                if not is_pristine_thin_streak:
                    logger.info(
                        f"[{mode}] {frame_filename}: Stage-2 VLM vetoed candidate '{cand['subtype']}' "
                        f"({vlm_res.get('description', 'false positive')})."
                    )
                    last_filter_status = {
                        "status": "vlm_filtered",
                        "mode": mode,
                        "stage1": cand["category"],
                        "vlm": vlm_res,
                        "timestamp": t0,
                    }
                    continue

            vlm_cat = str(vlm_res.get("category", "")).lower()
            # Preserve multi-peak aircraft strobe classification or high-confidence human k-NN match
            if cand["category"] == "aircraft" and vlm_cat in ("meteor", "fireball"):
                pass
            elif fb_eval and fb_eval.get("suggested_category") and fb_eval.get("knn_confidence", 0.0) >= 0.85:
                pass
            elif vlm_cat in CATEGORY_PRIORITY:
                cand["category"] = vlm_cat
                if vlm_res.get("subject"):
                    cand["subject"] = vlm_res["subject"]

        cand["_vlm_res"] = vlm_res
        verified_candidates.append(cand)

    if not verified_candidates:
        if last_filter_status:
            history.setdefault("processed_frames", {})[hist_frame_key] = last_filter_status
            save_history(history)
        return None

    iso_dt, date_str, hour_int, time_display = parse_frame_timestamp(frame_filename, meta)
    exp_start_ts, exp_end_ts = get_frame_exposure_window(frame_filename, exp_sec, t0, overlay_meta=meta)
    frame_mid_ts = 0.5 * (exp_start_ts + exp_end_ts)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    raw_stem = os.path.splitext(frame_filename)[0]
    base_stem = raw_stem if cam_id == "cam1" else f"{cam_id}_{raw_stem}"
    total_objs = len(verified_candidates)
    emitted_events: List[Dict[str, Any]] = []

    for obj_idx, cand in enumerate(verified_candidates, start=1):
        ev_frame_filename = (
            (frame_filename if cam_id == "cam1" else f"{cam_id}_{frame_filename}")
            if obj_idx == 1
            else f"{base_stem}_obj{obj_idx}.jpg"
        )
        snap_name = f"{base_stem}_anomaly.jpg" if obj_idx == 1 else f"{base_stem}_obj{obj_idx}_anomaly.jpg"
        snap_path = os.path.join(SNAPSHOT_DIR, snap_name)

        stereo_info = match_multi_camera_stereo(cand, cfg, frame_mid_ts)
        if stereo_info:
            cand["stereo_match"] = stereo_info
            cand["conf"] = max(float(cand.get("conf", 0.85)), 0.94)

        ordered_for_render = [cand] + [other for other in verified_candidates if other is not cand]
        annotated = render_annotated_snapshot(
            curr_img=img,
            prev_img=prev_img,
            detections=ordered_for_render,
            mode=mode,
            frame_time_str=time_display,
            meta=meta,
            cfg=cfg,
            object_index=obj_idx,
            objects_in_frame=total_objs,
        )
        cv2.imwrite(snap_path, annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        elapsed = round(time.time() - t0, 2)
        features_payload: Dict[str, Any] = {
            "camera_id": cam_id,
            "camera_name": cam_name,
            "length_px": cand.get("length_px", 0.0),
            "width_px": cand.get("width_px", 2.0),
            "aspect": cand.get("aspect", 1.0),
            "linearity": cand.get("linearity", 0.0),
            "fill_ratio": cand.get("fill_ratio", 1.0),
            "strobe_peaks": cand.get("strobe_peaks", 0),
            "deep_gaps": cand.get("deep_gaps", 0),
            "chroma_spread": cand.get("chroma_spread", 0.0),
            "snr": cand.get("snr", 0.0),
            "alt_deg": cand.get("alt_deg", 0.0),
            "az_deg": cand.get("az_deg", 0.0),
            "angle_deg": cand.get("angle_deg"),
            "p1": cand.get("p1"),
            "p2": cand.get("p2"),
            "bbox": cand.get("bbox", []),
            "object_index": obj_idx,
            "objects_in_frame": total_objs,
            "raw_frame_filename": frame_filename,
        }
        if stereo_info:
            features_payload["stereo_match"] = stereo_info
        if cand.get("adsb_match"):
            features_payload["adsb_flight"] = {
                "hex": cand["adsb_match"].get("hex"),
                "callsign": cand["adsb_match"].get("callsign"),
                "registration": cand["adsb_match"].get("registration"),
                "type_code": cand["adsb_match"].get("type_code"),
                "model_name": cand["adsb_match"].get("model_name"),
                "alt_ft": cand["adsb_match"].get("alt_ft"),
                "gs_kt": cand["adsb_match"].get("gs_kt"),
                "slant_miles": cand["adsb_match"].get("slant_miles"),
            }
        orb_ref = cand.get("orbital_match")
        if orb_ref:
            features_payload["orbital_object"] = {
                "norad_id": orb_ref.get("norad_id"),
                "name": orb_ref.get("name"),
                "obj_type": orb_ref.get("obj_type"),
                "type_desc": orb_ref.get("type_desc"),
                "orbit_alt_km": orb_ref.get("orbit_alt_km"),
                "orbit_alt_miles": orb_ref.get("orbit_alt_miles"),
                "speed_kms": orb_ref.get("speed_kms"),
                "speed_mph": orb_ref.get("speed_mph"),
                "slant_range_km": orb_ref.get("slant_range_km"),
                "match_mode": orb_ref.get("match_mode"),
                "match_dist_px": orb_ref.get("match_dist_px"),
                "match_ang_deg": orb_ref.get("match_ang_deg"),
            }
        vlm_res = cand.get("_vlm_res")
        event_subject = cand.get("subject") or cand["subtype"]
        event_desc = cand.get("adsb_description") or cand.get("orbital_description") or (vlm_res or {}).get("description") or ""
        if stereo_info:
            if stereo_info.get("triangulated_alt_km") is not None:
                stereo_note = (
                    f" [Stereo 3D Coincidence with {stereo_info.get('partner_camera_name')}: "
                    f"Δθ={stereo_info.get('parallax_deg')}°, Triangulated Alt ≈ {stereo_info.get('triangulated_alt_km')} km]"
                )
            else:
                stereo_note = (
                    f" [Multi-Camera Coincidence with {stereo_info.get('partner_camera_name')}: "
                    f"Δθ={stereo_info.get('parallax_deg')}°]"
                )
            event_desc = (event_desc + stereo_note).strip()

        event = {
            "frame_filename": ev_frame_filename,
            "camera_id": cam_id,
            "camera_name": cam_name,
            "snapshot_filename": snap_name,
            "snapshot_path": snap_path,
            "mode": mode,
            "frame_datetime": iso_dt,
            "frame_date": date_str,
            "frame_hour": hour_int,
            "frame_time_display": time_display,
            "category": cand["category"],
            "subtype": cand["subtype"],
            "subject": event_subject,
            "vlm_description": event_desc,
            "confidence": cand["conf"],
            "snr": cand["snr"],
            "length_px": cand.get("length_px", 0.0),
            "aspect": cand.get("aspect", 1.0),
            "linearity": cand.get("linearity", 0.0),
            "alt_deg": cand["alt_deg"],
            "az_deg": cand["az_deg"],
            "cardinal": cand["cardinal"],
            "bbox": cand["bbox"],
            "exposure_str": exp_str,
            "gain_str": gain_str,
            "processing_sec": elapsed,
            "detected_at": time.time(),
            "features_json": json.dumps(features_payload),
            "discord_message_id": "",
        }

        obj_log_tag = f" [Obj {obj_idx}/{total_objs}]" if total_objs > 1 else ""
        logger.info(
            f"[{mode}:{cam_id}] DETECTED {event['category'].upper()}{obj_log_tag} in {ev_frame_filename} ({elapsed}s): "
            f"{event['subject']} @ Alt {event['alt_deg']}° {event['cardinal']} (SNR={event['snr']}σ) | "
            f"VLM: {event.get('vlm_description') or 'N/A'}"
        )

        if send_discord:
            event["discord_sent"] = send_discord_sky_alert(event, cfg)
        else:
            event["discord_sent"] = False

        history.setdefault("recent_events", []).append(event)
        try:
            import web_portal
            web_portal.upsert_detection_event(event)
        except Exception as e:
            logger.warning(f"Failed upserting Allsky event to SQLite DB: {e}")

        emitted_events.append(event)

    first_ev = emitted_events[0]
    history.setdefault("processed_frames", {})[hist_frame_key] = {
        "status": "detected",
        "camera_id": cam_id,
        "mode": mode,
        "category": first_ev["category"],
        "subject": first_ev["subject"],
        "objects_in_frame": total_objs,
        "timestamp": time.time(),
    }
    save_history(history)

    return first_ev


# ---------------------------------------------------------------------------
# Realistic Synthetic Night Event Generator (for instant testing & verification)
# ---------------------------------------------------------------------------
def run_synthetic_night_test(
    event_type: str = "meteor",
    send_discord: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Fetches two real consecutive night frames from the Allsky Pi (20261014) or local cache,
    injects a physically realistic synthetic night event (shooting star, fireball, or satellite)
    into the second frame, and runs the full Stage-1 + Stage-2 VLM + PiP Snapshot + Discord pipeline!
    """
    cfg = load_config()
    session = requests.Session()
    base_url = cfg.get("allsky_base_url", "http://192.168.1.100")

    # Fetch 2 real night frames from 20261014 on the Pi
    url1 = f"{base_url}/images/20261014/image-20261015010004.jpg"
    url2 = f"{base_url}/images/20261014/image-20261015010040.jpg"

    def fetch_img(u: str) -> Optional[np.ndarray]:
        try:
            r = session.get(u, timeout=8)
            if r.status_code == 200:
                arr = np.frombuffer(r.content, dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            pass
        return None

    img1 = fetch_img(url1)
    img2 = fetch_img(url2)
    if img1 is None or img2 is None:
        logger.error("Could not fetch baseline night frames from Allsky Pi for synthetic test.")
        return None

    buf = RollingSkyBuffer(maxlen=5)
    meta = {
        "DAY_OR_NIGHT": "NIGHT",
        "AS_sEXPOSURE": "3.5 sec",
        "AS_EXPOSURE_US": "3500000",
        "AS_GAIN": "300",
        "AS_TEMPERATURE_F": "75.0",
        "AS_MOON_ELEVATION": "-12.4°",
        "AS_MOON_ILLUMINATION": "26.3",
    }

    # Prime buffer with baseline frame 1
    process_frame_array(img1, "image-20261015010004.jpg", meta, buf, cfg, send_discord=False, force=True)

    # Render realistic synthetic night sky event onto img2 inside the clear sky dome
    synth = img2.copy()
    if event_type == "fireball":
        # Bright green-white bolide streak with terminal flare in the NW sky
        x_start, y_start = 1320, 1420
        x_end, y_end = 1580, 1620
        num_steps = 300
        for i in range(num_steps):
            t = i / float(num_steps - 1)
            # Tapered ionization curve + bright terminal burst at t=0.78
            env = math.sin(math.pi * t) ** 0.7
            burst = 1.65 * math.exp(-((t - 0.78) / 0.08) ** 2)
            intensity = min(1.0, 0.55 * env + burst)
            px = int(round(x_start + (x_end - x_start) * t))
            py = int(round(y_start + (y_end - y_start) * t))
            rad = 3 if burst > 0.6 else 2
            color = (int(210 * intensity), int(255 * intensity), int(230 * intensity))
            cv2.circle(synth, (px, py), rad, color, -1, cv2.LINE_AA)
        synth = cv2.GaussianBlur(synth, (3, 3), 0)
        synth = np.maximum(img2, synth)
    else:
        # Classic crisp shooting star (meteor) streak in the upper-right sky dome
        x_start, y_start = 2050, 1340
        x_end, y_end = 2235, 1465
        num_steps = 220
        layer = np.zeros_like(synth, dtype=np.float32)
        for i in range(num_steps):
            t = i / float(num_steps - 1)
            # Smooth rise, peak at 65% along trajectory, smooth fade
            env = (math.sin(math.pi * (t ** 0.85))) ** 1.2
            px = int(round(x_start + (x_end - x_start) * t))
            py = int(round(y_start + (y_end - y_start) * t))
            val = 145.0 * env
            cv2.circle(layer, (px, py), 1, (val * 0.92, val * 1.0, val * 0.95), -1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (3, 3), 0.8)
        synth = np.clip(synth.astype(np.float32) + layer, 0, 255).astype(np.uint8)

    ts_tag = datetime.now().strftime("%Y%m%d%H%M%S")
    synth_fname = f"image-{ts_tag}-sim-{event_type}.jpg"
    return process_frame_array(synth, synth_fname, meta, buf, cfg, send_discord=send_discord, force=True)


# ---------------------------------------------------------------------------
# Archive Date Scanner & Live Continuous Poller Daemon (1..N Cameras)
# ---------------------------------------------------------------------------
def scan_archive_date(
    date_folder: str,
    stride: int = 1,
    limit: int = 150,
    send_discord: bool = False,
    force: bool = False,
    camera_id: Optional[str] = None,
) -> List[dict]:
    """Scans archived frames from 1..N configured camera nodes for <date_folder>."""
    cfg = load_config()
    cameras = get_configured_cameras(cfg, enabled_only=True)
    if camera_id:
        cameras = [c for c in cameras if str(c.get("id")) == str(camera_id)]
    session = requests.Session()
    events: List[dict] = []

    for cam_entry in cameras:
        cam_cfg = get_effective_camera_cfg(cfg, cam_entry)
        cam_id = str(cam_cfg.get("camera_id") or "cam1")
        base_url = cam_cfg.get("allsky_base_url", "http://192.168.1.100").rstrip("/")
        dir_url = f"{base_url}/images/{date_folder}/"

        try:
            r = session.get(dir_url, timeout=8)
            if r.status_code != 200:
                logger.error(f"[{cam_id}] Archive directory {dir_url} returned HTTP {r.status_code}")
                continue
            files = re.findall(r'href="(image-\d+\.jpg)"', r.text)
        except Exception as e:
            logger.error(f"[{cam_id}] Failed listing archive {dir_url}: {e}")
            continue

        if stride > 1:
            files = files[::stride]
        if limit > 0:
            files = files[-limit:]

        logger.info(f"[{cam_id}] Scanning {len(files)} frame(s) in {date_folder} (stride={stride})...")
        buf = RollingSkyBuffer(maxlen=5)

        for fname in files:
            img_url = f"{dir_url}{fname}"
            try:
                resp = session.get(img_url, timeout=8)
                if resp.status_code != 200:
                    continue
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                ev = process_frame_array(img, fname, {}, buf, cam_cfg, send_discord=send_discord, force=force)
                if ev:
                    events.append(ev)
            except Exception as e:
                logger.warning(f"[{cam_id}] Error scanning {fname}: {e}")

    logger.info(f"Finished archive scan of {date_folder}: {len(events)} anomaly event(s) detected across {len(cameras)} camera(s).")
    return events


def catch_up_missed_archive_frames(session: requests.Session, buf: RollingSkyBuffer, cfg: dict) -> str:
    """
    On daemon startup (e.g., after a PC reboot), scans the active Allsky Pi archive folder(s)
    for any frames captured since the last processed frame (up to 90 frames / ~45 mins),
    primes the rolling buffer with the preceding frame, and processes all missed frames.
    """
    cam_id = str(cfg.get("camera_id") or "cam1")
    base_url = cfg.get("allsky_base_url", "http://192.168.1.100").rstrip("/")
    now_dt = datetime.now()
    folders = list(dict.fromkeys([
        (now_dt - timedelta(hours=12)).strftime("%Y%m%d"),
        now_dt.strftime("%Y%m%d"),
    ]))
    processed_frames = load_history().get("processed_frames", {})
    meta = fetch_overlay_metadata(session, cfg)
    last_caught_up = ""

    def _hist_key(fn: str) -> str:
        return fn if cam_id == "cam1" else f"{cam_id}:{fn}"

    for df in folders:
        dir_url = f"{base_url}/images/{df}/"
        try:
            r = session.get(dir_url, timeout=6)
            if r.status_code != 200:
                continue
            files = re.findall(r'href="(image-\d+\.jpg)"', r.text)
            if not files:
                continue
        except Exception as e:
            logger.debug(f"[{cam_id}] Startup catch-up skipped {dir_url}: {e}")
            continue

        proc_indices = [i for i, fn in enumerate(files) if _hist_key(fn) in processed_frames]
        if proc_indices:
            last_idx = proc_indices[-1]
            prime_files = [files[last_idx]]
            missed_files = [fn for fn in files[last_idx + 1 :] if _hist_key(fn) not in processed_frames][-90:]
        else:
            prime_files = files[-31:-30]
            missed_files = [fn for fn in files[-30:] if _hist_key(fn) not in processed_frames]

        if prime_files and (missed_files or len(buf.frames) == 0):
            p_fname = prime_files[0]
            try:
                resp = session.get(f"{dir_url}{p_fname}", timeout=8)
                if resp.status_code == 200 and len(resp.content) > 50000:
                    arr = np.frombuffer(resp.content, dtype=np.uint8)
                    p_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if p_img is not None:
                        h, w = p_img.shape[:2]
                        sky_mask = ensure_sky_mask(cfg, (h, w))
                        gray = cv2.cvtColor(p_img, cv2.COLOR_BGR2GRAY)
                        hp, _ = compute_highpass(gray, bg_kernel=int(cfg.get("night_bg_kernel", 35)))
                        mode = infer_day_or_night(p_img, sky_mask, meta)
                        buf.push(p_fname, mode, p_img, gray, hp, ts=time.time())
                        last_caught_up = p_fname
            except Exception as e:
                logger.debug(f"[{cam_id}] Startup buffer prime error on {p_fname}: {e}")

        if missed_files:
            logger.info(
                f"[{cam_id}] Startup catch-up: scanning {len(missed_files)} missed frame(s) in {df} "
                f"({missed_files[0]} .. {missed_files[-1]})..."
            )
            caught_events = 0
            for fname in missed_files:
                try:
                    resp = session.get(f"{dir_url}{fname}", timeout=8)
                    if resp.status_code != 200 or len(resp.content) <= 50000:
                        continue
                    arr = np.frombuffer(resp.content, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    ev = process_frame_array(
                        img=img,
                        frame_filename=fname,
                        meta=dict(meta),
                        buffer=buf,
                        cfg=cfg,
                        send_discord=True,
                        force=False,
                    )
                    if ev:
                        caught_events += 1
                    last_caught_up = fname
                except Exception as e:
                    logger.warning(f"[{cam_id}] Startup catch-up error on {fname}: {e}")
            logger.info(f"[{cam_id}] Startup catch-up finished for {df}: {caught_events} anomaly event(s) detected.")

    return last_caught_up


def run_daemon() -> None:
    """
    Continuous Real-Time Multi-Camera (1..N) Allsky Frame Watcher & Anomaly Detection Daemon.
    Polls each enabled camera node's live_image_url via lightweight HTTP HEAD every poll_interval_seconds,
    maintains independent per-camera rolling sky buffers, correlates simultaneous multi-camera stereo events,
    and runs the full detection pipeline the instant a new frame is captured!
    """
    import web_portal

    cfg = load_config()
    ensure_sky_mask(cfg)

    if cfg.get("adsb_enabled", True):
        try:
            import adsb_tracker
            adsb_tracker.start_background_adsb_thread()
        except Exception as e:
            logger.warning(f"Failed starting ADS-B tracker thread: {e}")

    if cfg.get("orbital_tracking_enabled", True):
        try:
            import orbital_tracker
            orbital_tracker.ensure_catalog_loaded()
        except Exception as e:
            logger.warning(f"Failed loading orbital TLE catalog on startup: {e}")

    portal_port = int(cfg.get("web_portal_port", 8094))
    try:
        requests.get(f"http://localhost:{portal_port}/api/version", timeout=2)
    except Exception:
        py_exe = sys.executable
        portal_script = os.path.join(BASE_DIR, "web_portal.py")
        subprocess.Popen(
            [py_exe, portal_script, "--port", str(portal_port)],
            cwd=BASE_DIR,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) | getattr(subprocess, "DETACHED_PROCESS", 0x00000008),
            close_fds=True,
        )

    session = requests.Session()
    cam_states: Dict[str, Dict[str, Any]] = {}
    last_dawn_survey_date = ""

    init_cameras = get_configured_cameras(cfg, enabled_only=True)
    cam_desc_list = [f"{c['id']} ({c['live_image_url']})" for c in init_cameras]
    logger.info(
        f"Allsky Real-Time Anomaly Detector Daemon started for {len(init_cameras)} camera(s): "
        f"{', '.join(cam_desc_list)} -> Discord channel '{cfg.get('discord_channel', 'allsky')}'."
    )

    for cam_entry in init_cameras:
        cam_id = str(cam_entry["id"])
        cam_cfg = get_effective_camera_cfg(cfg, cam_entry)
        ensure_sky_mask(cam_cfg)
        state = {
            "buf": RollingSkyBuffer(maxlen=5),
            "last_etag": "",
            "last_frame_name": "",
            "last_raw_survey_save_ts": 0.0,
        }
        cam_states[cam_id] = state
        try:
            caught_up_last = catch_up_missed_archive_frames(session, state["buf"], cam_cfg)
            if caught_up_last:
                state["last_frame_name"] = caught_up_last
        except Exception as e:
            logger.warning(f"[{cam_id}] Startup archive catch-up warning: {e}")

    while True:
        cfg = load_config()
        poll_sec = max(1.5, float(cfg.get("poll_interval_seconds", 3.5)))
        enabled_cameras = get_configured_cameras(cfg, enabled_only=True)

        for cam_entry in enabled_cameras:
            cam_id = str(cam_entry["id"])
            cam_cfg = get_effective_camera_cfg(cfg, cam_entry)
            if cam_id not in cam_states:
                ensure_sky_mask(cam_cfg)
                cam_states[cam_id] = {
                    "buf": RollingSkyBuffer(maxlen=5),
                    "last_etag": "",
                    "last_frame_name": "",
                    "last_raw_survey_save_ts": 0.0,
                }
            state = cam_states[cam_id]
            live_url = cam_cfg.get("live_image_url", "http://192.168.1.100/current/tmp/image.jpg")

            try:
                head_resp = session.head(live_url, timeout=4)
                if head_resp.status_code == 200:
                    etag = head_resp.headers.get("ETag") or head_resp.headers.get("Last-Modified") or ""
                    if etag and etag != state["last_etag"]:
                        meta = fetch_overlay_metadata(session, cam_cfg)
                        cur_img_path = meta.get("CURRENT_IMAGE", "")
                        frame_name = os.path.basename(cur_img_path) if cur_img_path else f"image-{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"

                        if frame_name != state["last_frame_name"]:
                            img_resp = session.get(live_url, timeout=22)
                            if img_resp.status_code == 200 and len(img_resp.content) > 50000:
                                arr = np.frombuffer(img_resp.content, dtype=np.uint8)
                                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                if img is not None:
                                    state["last_etag"] = etag
                                    state["last_frame_name"] = frame_name
                                    try:
                                        live_path = get_camera_live_frame_path(cam_id)
                                        tmp_live = live_path + ".tmp"
                                        with open(tmp_live, "wb") as lf:
                                            lf.write(img_resp.content)
                                        os.replace(tmp_live, live_path)
                                    except Exception:
                                        pass
                                    now_dt = datetime.now()
                                    # Periodically archive untouched raw night frames every 20 min for multi-year deep survey
                                    if (now_dt.hour >= 19 or now_dt.hour < 6) and (time.time() - state["last_raw_survey_save_ts"] >= 1200.0):
                                        try:
                                            night_dir_tag = (now_dt - timedelta(hours=12)).strftime("%Y%m%d")
                                            raw_dir = os.path.join(BASE_DIR, "survey_data", "raw_archive", night_dir_tag)
                                            os.makedirs(raw_dir, exist_ok=True)
                                            raw_1920 = cv2.resize(img, (1920, 1920), interpolation=cv2.INTER_AREA)
                                            raw_fname = frame_name if cam_id == "cam1" else f"{cam_id}_{frame_name}"
                                            cv2.imwrite(os.path.join(raw_dir, raw_fname), raw_1920, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
                                            state["last_raw_survey_save_ts"] = time.time()
                                        except Exception as raw_e:
                                            logger.debug(f"[{cam_id}] Raw survey archive save skipped: {raw_e}")
                                    # Trigger automatic multi-epoch J2000 co-addition & tracklet linker each morning at 06:15
                                    today_tag = now_dt.strftime("%Y%m%d")
                                    if now_dt.hour == 6 and now_dt.minute >= 15 and last_dawn_survey_date != today_tag:
                                        last_dawn_survey_date = today_tag
                                        try:
                                            import threading as _th
                                            import deep_sky_survey
                                            _th.Thread(target=deep_sky_survey.run_multi_epoch_deep_survey, kwargs={"force_rebuild": False}, daemon=True).start()
                                        except Exception as surv_e:
                                            logger.warning(f"Dawn deep survey trigger error: {surv_e}")
                                    # Evaluate user-triggered Camera Realignment ONLY when armed by the user
                                    if cam_id == "cam1":
                                        try:
                                            import camera_realign
                                            if camera_realign.load_realign_state().get("armed"):
                                                def _reset_mask(c, sh):
                                                    _SKY_MASK_CACHE["mask"] = None
                                                    _SKY_MASK_CACHE["by_cam"].clear()
                                                    ensure_sky_mask(c, sh)
                                                camera_realign.process_frame_for_realignment(
                                                    bgr=img,
                                                    frame_filename=frame_name,
                                                    dt_utc=now_dt.astimezone(timezone.utc),
                                                    cfg=cfg,
                                                    save_config_fn=save_config,
                                                    reset_sky_mask_fn=_reset_mask,
                                                )
                                        except Exception as realign_e:
                                            logger.warning(f"Camera realignment check error: {realign_e}")
                                    process_frame_array(
                                        img=img,
                                        frame_filename=frame_name,
                                        meta=meta,
                                        buffer=state["buf"],
                                        cfg=cam_cfg,
                                        send_discord=True,
                                        force=False,
                                    )
            except Exception as e:
                logger.warning(f"[{cam_id}] Allsky live poll error: {e}")

        time.sleep(poll_sec)


def run_health_check() -> dict:
    """Checks and self-heals all Allsky workflow components via allsky_workflow_guard."""
    try:
        import allsky_workflow_guard
        report = allsky_workflow_guard.run_guard_cycle(trigger_discord_alerts=True)
        print(json.dumps(report, indent=2))
        return report
    except Exception as e:
        logger.error(f"Error running allsky_workflow_guard: {e}")
        return {"error": str(e), "healthy": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BluffSky Allsky Anomaly & Sky Event Detector (1..N Cameras)")
    parser.add_argument("--daemon", action="store_true", help="Run continuous real-time Allsky frame watcher daemon")
    parser.add_argument("--scan-date", help="Scan archived date folder on Allsky Pi (e.g. 20261014 or 20261015)")
    parser.add_argument("--camera-id", help="Optional camera_id filter for --scan-date (e.g. cam1, cam2)")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride when scanning archive folder")
    parser.add_argument("--limit", type=int, default=150, help="Max frames to scan in archive folder")
    parser.add_argument("--simulate-test", nargs="?", const="meteor", choices=["meteor", "fireball"], help="Run a realistic synthetic night event test")
    parser.add_argument("--send-discord", "-d", action="store_true", help="Send Discord alert for detections")
    parser.add_argument("--force", action="store_true", help="Re-analyze even if already in detections_history.json")
    parser.add_argument("--health-check", action="store_true", help="Check and self-heal Allsky detector components")
    args = parser.parse_args()

    if args.health_check:
        run_health_check()
    elif args.daemon:
        run_daemon()
    elif args.simulate_test:
        run_synthetic_night_test(event_type=args.simulate_test, send_discord=args.send_discord)
    elif args.scan_date:
        scan_archive_date(args.scan_date, stride=args.stride, limit=args.limit, send_discord=args.send_discord, force=args.force, camera_id=args.camera_id)
    else:
        parser.print_help()
