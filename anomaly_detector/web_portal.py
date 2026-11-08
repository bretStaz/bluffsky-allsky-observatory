import argparse
from datetime import datetime
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import socket
import sqlite3
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "allsky_detections.db")
HISTORY_FILE = os.path.join(BASE_DIR, "detections_history.json")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
LIVE_STATUS_FILE = os.path.join(BASE_DIR, "live_status.json")
LATEST_LIVE_FRAME_FILE = os.path.join(BASE_DIR, "latest_live_frame.jpg")
REALIGN_STATE_FILE = os.path.join(BASE_DIR, "realignment_state.json")
MASK_PREVIEW_FILE = os.path.join(BASE_DIR, "sky_mask_preview.jpg")
PORTAL_LOG_FILE = os.path.join(BASE_DIR, "web_portal.log")
DISCORD_CONFIG_FILE = r"E:\discord_messages\config.json"

_CACHE_LOCK = threading.Lock()
_EVENTS_MUTATION_SEQ = 0
_DB_EVENTS_CACHE: Dict[str, Any] = {"key": None, "bundle": None}
_HISTORY_COUNT_CACHE: Dict[str, Any] = {"mtime": -1.0, "count": 0}
_STATS_HTTP_CACHE: Dict[str, Any] = {"etag": "", "raw_bytes": b"", "gzip_bytes": b"", "sig": None}
_LIVE_IMG_RENDER_CACHE: Dict[tuple, Tuple[float, bytes]] = {}

logger = logging.getLogger("AllskyWebPortal")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = RotatingFileHandler(PORTAL_LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

DB_LOCK = threading.Lock()
_DB_INITIALIZED = False

CATEGORY_BASE_INTEREST = {
    "fireball": 52.0,
    "meteor": 45.0,
    "sky_phenomenon": 34.0,
    "space_object": 30.0,
    "transient_flash": 24.0,
    "day_anomaly": 20.0,
    "day_bird": 17.0,
    "day_aircraft": 14.0,
    "aircraft": 12.0,
}


def compute_interest_score(ev: Dict[str, Any]) -> float:
    """
    Calculates an astronomical / visual Interest Score (0.0 - 99.9) used to rank the
    Top 50 Most Interesting sky captures.
    Prioritizes fireballs, shooting stars, and high-SNR space flares over routine aircraft,
    while rewarding long, crisp streaks and high sky altitude away from horizon obstacles.
    """
    cat = str(ev.get("category") or "meteor").lower()
    base = CATEGORY_BASE_INTEREST.get(cat, 18.0)

    snr = max(0.0, float(ev.get("snr") or 0.0))
    length_px = max(0.0, float(ev.get("length_px") or 0.0))
    linearity = max(0.0, min(1.0, float(ev.get("linearity") or 0.0)))
    alt_deg = max(0.0, min(90.0, float(ev.get("alt_deg") or 0.0)))
    conf = max(0.0, min(1.0, float(ev.get("confidence") or 0.0)))

    snr_bonus = min(22.0, snr * 0.25)
    len_bonus = min(14.0, length_px * 0.11)
    lin_bonus = linearity * 5.0 if length_px >= 15.0 else 1.5
    alt_bonus = min(7.0, alt_deg / 12.8)
    conf_bonus = conf * 3.0

    # Down-weight synthetic dusk auto-exposure zenith flash if length_px == 0 and category == sky_phenomenon
    if cat == "sky_phenomenon" and length_px == 0.0:
        base = 20.0

    score = base + snr_bonus + len_bonus + lin_bonus + alt_bonus + conf_bonus
    return round(min(99.9, max(1.0, score)), 1)


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE, timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    global _DB_INITIALIZED
    if _DB_INITIALIZED and os.path.exists(DB_FILE):
        return
    with DB_LOCK:
        conn = get_db_connection()
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sky_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_filename TEXT UNIQUE NOT NULL,
                    snapshot_filename TEXT,
                    snapshot_path TEXT,
                    mode TEXT NOT NULL,
                    frame_datetime TEXT NOT NULL,
                    frame_date TEXT NOT NULL,
                    frame_hour INTEGER NOT NULL,
                    frame_time_display TEXT,
                    category TEXT NOT NULL,
                    subtype TEXT,
                    subject TEXT,
                    vlm_description TEXT,
                    confidence REAL DEFAULT 0.0,
                    snr REAL DEFAULT 0.0,
                    length_px REAL DEFAULT 0.0,
                    aspect REAL DEFAULT 1.0,
                    linearity REAL DEFAULT 0.0,
                    alt_deg REAL DEFAULT 0.0,
                    az_deg REAL DEFAULT 0.0,
                    cardinal TEXT,
                    exposure_str TEXT,
                    gain_str TEXT,
                    processing_sec REAL DEFAULT 0.0,
                    discord_sent INTEGER DEFAULT 0,
                    detected_at REAL NOT NULL,
                    interest_score REAL DEFAULT 0.0,
                    is_starred INTEGER DEFAULT 0,
                    is_hidden INTEGER DEFAULT 0,
                    camera_id TEXT DEFAULT 'cam1',
                    camera_name TEXT DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_sky_events_dt ON sky_events(frame_datetime DESC);
                CREATE INDEX IF NOT EXISTS idx_sky_events_cat ON sky_events(category);
                CREATE INDEX IF NOT EXISTS idx_sky_events_mode ON sky_events(mode);
                """
            )
            # Ensure new columns exist on existing databases
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(sky_events)").fetchall()}
            if "interest_score" not in cols:
                conn.execute("ALTER TABLE sky_events ADD COLUMN interest_score REAL DEFAULT 0.0;")
            if "is_starred" not in cols:
                conn.execute("ALTER TABLE sky_events ADD COLUMN is_starred INTEGER DEFAULT 0;")
            if "is_hidden" not in cols:
                conn.execute("ALTER TABLE sky_events ADD COLUMN is_hidden INTEGER DEFAULT 0;")
            if "discord_message_id" not in cols:
                conn.execute("ALTER TABLE sky_events ADD COLUMN discord_message_id TEXT DEFAULT '';")
            if "user_corrected_category" not in cols:
                conn.execute("ALTER TABLE sky_events ADD COLUMN user_corrected_category TEXT DEFAULT '';")
            if "user_feedback_note" not in cols:
                conn.execute("ALTER TABLE sky_events ADD COLUMN user_feedback_note TEXT DEFAULT '';")
            if "features_json" not in cols:
                conn.execute("ALTER TABLE sky_events ADD COLUMN features_json TEXT DEFAULT '';")
            if "camera_id" not in cols:
                conn.execute("ALTER TABLE sky_events ADD COLUMN camera_id TEXT DEFAULT 'cam1';")
            if "camera_name" not in cols:
                conn.execute("ALTER TABLE sky_events ADD COLUMN camera_name TEXT DEFAULT '';")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sky_events_cam ON sky_events(camera_id);")

            # Backfill interest_score where 0
            rows = conn.execute("SELECT * FROM sky_events WHERE interest_score IS NULL OR interest_score <= 0.1").fetchall()
            for r in rows:
                sc = compute_interest_score(dict(r))
                conn.execute("UPDATE sky_events SET interest_score = ? WHERE id = ?", (sc, r["id"]))

            conn.commit()
            _DB_INITIALIZED = True
        finally:
            conn.close()
    try:
        import feedback_learner
        feedback_learner.ensure_feedback_schema()
    except Exception as e:
        logger.warning(f"feedback_learner init warning: {e}")


def upsert_detection_event(ev: Dict[str, Any]) -> None:
    init_db()
    score = compute_interest_score(ev)
    cam_id = str(ev.get("camera_id") or "cam1")
    cam_name = str(ev.get("camera_name") or "")
    with DB_LOCK:
        conn = get_db_connection()
        try:
            conn.execute(
                """
                INSERT INTO sky_events (
                    frame_filename, snapshot_filename, snapshot_path, mode,
                    frame_datetime, frame_date, frame_hour, frame_time_display,
                    category, subtype, subject, vlm_description,
                    confidence, snr, length_px, aspect, linearity,
                    alt_deg, az_deg, cardinal, exposure_str, gain_str,
                    processing_sec, discord_sent, detected_at, interest_score,
                    discord_message_id, features_json, camera_id, camera_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(frame_filename) DO UPDATE SET
                    category=CASE WHEN sky_events.user_corrected_category != '' THEN sky_events.category ELSE excluded.category END,
                    subject=CASE WHEN sky_events.user_corrected_category != '' THEN sky_events.subject ELSE excluded.subject END,
                    vlm_description=excluded.vlm_description,
                    discord_sent=excluded.discord_sent,
                    detected_at=excluded.detected_at,
                    interest_score=CASE WHEN sky_events.user_corrected_category != '' THEN sky_events.interest_score ELSE excluded.interest_score END,
                    discord_message_id=CASE WHEN excluded.discord_message_id != '' THEN excluded.discord_message_id ELSE sky_events.discord_message_id END,
                    features_json=CASE WHEN excluded.features_json != '' THEN excluded.features_json ELSE sky_events.features_json END,
                    camera_id=excluded.camera_id,
                    camera_name=CASE WHEN excluded.camera_name != '' THEN excluded.camera_name ELSE sky_events.camera_name END
                """,
                (
                    ev.get("frame_filename"),
                    ev.get("snapshot_filename"),
                    ev.get("snapshot_path"),
                    ev.get("mode", "NIGHT"),
                    ev.get("frame_datetime", ""),
                    ev.get("frame_date", ""),
                    int(ev.get("frame_hour", 0)),
                    ev.get("frame_time_display", ""),
                    ev.get("category", "meteor"),
                    ev.get("subtype", ""),
                    ev.get("subject", ""),
                    ev.get("vlm_description", ""),
                    float(ev.get("confidence", 0.0)),
                    float(ev.get("snr", 0.0)),
                    float(ev.get("length_px", 0.0)),
                    float(ev.get("aspect", 1.0)),
                    float(ev.get("linearity", 0.0)),
                    float(ev.get("alt_deg", 0.0)),
                    float(ev.get("az_deg", 0.0)),
                    ev.get("cardinal", ""),
                    ev.get("exposure_str", ""),
                    ev.get("gain_str", ""),
                    float(ev.get("processing_sec", 0.0)),
                    1 if ev.get("discord_sent") else 0,
                    float(ev.get("detected_at", time.time())),
                    score,
                    str(ev.get("discord_message_id") or ""),
                    str(ev.get("features_json") or ""),
                    cam_id,
                    cam_name,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    bump_events_cache()


def sync_from_history_json() -> None:
    init_db()
    if not os.path.exists(HISTORY_FILE):
        return
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            hist = json.load(f)
        with DB_LOCK:
            conn = get_db_connection()
            try:
                existing = {r["frame_filename"] for r in conn.execute("SELECT frame_filename FROM sky_events").fetchall()}
            finally:
                conn.close()
        for ev in hist.get("recent_events", []):
            fname = ev.get("frame_filename")
            if fname and fname not in existing:
                upsert_detection_event(ev)
    except Exception as e:
        logger.warning(f"History sync warning: {e}")


def _parse_dt_seconds(ev: Dict[str, Any]) -> float:
    dt_str = ev.get("frame_datetime") or ""
    if dt_str:
        try:
            return datetime.fromisoformat(dt_str).timestamp()
        except Exception:
            pass
    return float(ev.get("detected_at") or 0.0)


def _base_raw_frame_stem(fname: Any) -> str:
    return re.sub(r"_obj\d+(?=\.jpg$|$)", "", str(fname or ""))


def assign_multi_frame_passes(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identifies consecutive frames (within <= 95 seconds of each other) that capture the same
    aircraft or satellite transit across the sky dome.
    Never bundles two distinct objects captured in the SAME frame (`dt_gap < 5.0` or same base frame stem)
    so each object in a multi-object frame remains independently visible and reclassifiable.
    Assigns `pass_id`, `pass_count`, `is_pass_best`, and `pass_frames` (chronological list of frame filenames/snapshots)
    so the UI can either bundle multi-frame transits into a single expandable card OR show all frames.
    """
    if not events:
        return []

    # Sort chronologically (oldest to newest) to chain consecutive frames
    chron = sorted(events, key=_parse_dt_seconds)
    groups: List[List[Dict[str, Any]]] = []
    transit_cats = {"aircraft", "space_object", "transient_flash", "day_aircraft"}
    orbital_cats = {"space_object", "transient_flash"}

    for ev in chron:
        ev["camera_id"] = str(ev.get("camera_id") or "cam1")
        ev["camera_name"] = str(ev.get("camera_name") or "")
        ev["is_test"] = "-sim-" in str(ev.get("frame_filename") or "")
        if not ev.get("interest_score"):
            ev["interest_score"] = compute_interest_score(ev)
        else:
            ev["interest_score"] = round(float(ev["interest_score"]), 1)

        if ev["is_test"]:
            groups.append([ev])
            continue

        ev_cat = ev.get("category")
        ev_ts = _parse_dt_seconds(ev)
        ev_stem = _base_raw_frame_stem(ev.get("frame_filename"))
        ev_alt = float(ev.get("alt_deg") or 0.0)
        ev_az = float(ev.get("az_deg") or 0.0)

        placed = False
        if ev_cat in transit_cats and groups:
            # Check recent open groups (newest first) within the 95-second transit window
            for cand_grp in reversed(groups[-8:]):
                last_ev = cand_grp[-1]
                if last_ev.get("is_test") or last_ev.get("mode") != ev.get("mode"):
                    continue
                if (last_ev.get("camera_id") or "cam1") != (ev.get("camera_id") or "cam1"):
                    continue
                last_cat = last_ev.get("category")
                if last_cat not in transit_cats:
                    continue
                dt_gap = abs(ev_ts - _parse_dt_seconds(last_ev))
                if dt_gap > 95.0:
                    continue
                # Never merge two objects captured in the SAME exposure frame!
                last_stem = _base_raw_frame_stem(last_ev.get("frame_filename"))
                if dt_gap < 5.0 or (ev_stem and ev_stem == last_stem):
                    continue
                # Require compatible category (aircraft with aircraft, orbital/flash with orbital/flash)
                cat_compatible = (ev_cat == last_cat) or ({ev_cat, last_cat} <= orbital_cats)
                if not cat_compatible:
                    continue
                # Require plausible sky trajectory continuity across consecutive frames
                last_alt = float(last_ev.get("alt_deg") or 0.0)
                last_az = float(last_ev.get("az_deg") or 0.0)
                az_diff = abs((ev_az - last_az + 180.0) % 360.0 - 180.0)
                alt_diff = abs(ev_alt - last_alt)
                if alt_diff <= 48.0 and (az_diff <= 80.0 or ev_alt >= 68.0 or last_alt >= 68.0):
                    cand_grp.append(ev)
                    placed = True
                    break

        if not placed:
            groups.append([ev])

    for g_idx, grp in enumerate(groups, start=1):
        pass_id = f"pass-{g_idx}"
        # Pick the best frame in the pass (starred first, then highest interest_score, then highest SNR)
        best_ev = max(
            grp,
            key=lambda x: (int(x.get("is_starred") or 0), float(x.get("interest_score") or 0.0), float(x.get("snr") or 0.0)),
        )
        pass_frames_meta = [
            {
                "id": item.get("id"),
                "frame_filename": item.get("frame_filename"),
                "snapshot_filename": item.get("snapshot_filename"),
                "frame_time_display": item.get("frame_time_display"),
                "category": item.get("category"),
                "subject": item.get("subject"),
                "alt_deg": item.get("alt_deg"),
                "az_deg": item.get("az_deg"),
                "cardinal": item.get("cardinal"),
                "snr": item.get("snr"),
                "length_px": item.get("length_px"),
                "interest_score": item.get("interest_score"),
            }
            for item in grp
        ]
        for item in grp:
            is_best = item["frame_filename"] == best_ev["frame_filename"]
            item["pass_id"] = pass_id
            item["pass_count"] = len(grp)
            item["is_pass_best"] = is_best
            # Attach pass_frames list only once per pass (on is_pass_best) to avoid O(K^2) JSON bloat
            item["pass_frames"] = pass_frames_meta if (is_best and len(grp) > 1) else []
            if len(grp) > 1:
                item["pass_time_span"] = f"{grp[0].get('frame_time_display', '')[-11:]} → {grp[-1].get('frame_time_display', '')[-11:]}"
            else:
                item["pass_time_span"] = ""

    # Return newest-first by capture timestamp
    return sorted(chron, key=lambda x: (str(x.get("frame_datetime") or ""), float(x.get("detected_at") or 0.0)), reverse=True)


def _safe_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path) if os.path.exists(path) else 0.0
    except Exception:
        return 0.0


def bump_events_cache() -> None:
    global _EVENTS_MUTATION_SEQ
    with _CACHE_LOCK:
        _EVENTS_MUTATION_SEQ += 1


def _get_cached_db_events_bundle() -> Dict[str, Any]:
    init_db()
    db_key = (_safe_mtime(DB_FILE), _safe_mtime(DB_FILE + "-wal"), _EVENTS_MUTATION_SEQ)
    with _CACHE_LOCK:
        if _DB_EVENTS_CACHE["key"] == db_key and _DB_EVENTS_CACHE["bundle"] is not None:
            return _DB_EVENTS_CACHE["bundle"]

    with DB_LOCK:
        conn = get_db_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM sky_events WHERE is_hidden = 0 ORDER BY detected_at DESC LIMIT 3000"
            ).fetchall()
            raw_events = [dict(r) for r in rows]
        finally:
            conn.close()

    events = assign_multi_frame_passes(raw_events)

    # Compute Top 50 Most Interesting Space & Astronomical Captures (excluding routine aircraft & synthetic tests)
    real_pass_bests = [
        e for e in events
        if e.get("is_pass_best", True)
        and (not e.get("is_test") or e.get("is_starred"))
        and e.get("mode") == "NIGHT"
        and e.get("category") in ("fireball", "meteor", "space_object", "transient_flash", "sky_phenomenon")
    ]
    top_50 = sorted(
        real_pass_bests,
        key=lambda e: (
            int(e.get("is_starred") or 0),
            float(e.get("interest_score") or 0.0),
            float(e.get("snr") or 0.0),
        ),
        reverse=True,
    )[:50]
    for rank_idx, item in enumerate(top_50, start=1):
        item["top_rank"] = rank_idx

    rank_map = {item["frame_filename"]: item["top_rank"] for item in top_50}
    for e in events:
        e["top_rank"] = rank_map.get(e["frame_filename"], 0)

    by_category: Dict[str, int] = {}
    by_date: Dict[str, Dict[str, Any]] = {}
    night_count = 0
    day_count = 0
    real_events_count = 0
    unique_passes_count = 0

    for e in events:
        if e.get("is_test"):
            continue
        real_events_count += 1
        if e.get("is_pass_best"):
            unique_passes_count += 1
        cat = e["category"]
        by_category[cat] = by_category.get(cat, 0) + 1
        if e.get("mode") == "NIGHT":
            night_count += 1
        else:
            day_count += 1

        fdate = e.get("frame_date") or "Unknown"
        if fdate not in by_date:
            by_date[fdate] = {"date": fdate, "total": 0, "passes": 0, "meteors": 0, "space": 0, "aircraft": 0}
        by_date[fdate]["total"] += 1
        if e.get("is_pass_best"):
            by_date[fdate]["passes"] += 1
        if cat in ("meteor", "fireball"):
            by_date[fdate]["meteors"] += 1
        elif cat in ("space_object", "transient_flash", "sky_phenomenon"):
            by_date[fdate]["space"] += 1
        elif cat in ("aircraft", "day_aircraft"):
            by_date[fdate]["aircraft"] += 1

    date_folders = sorted(by_date.values(), key=lambda d: d["date"], reverse=True)
    starred_count = sum(1 for e in events if e.get("is_starred"))

    bundle = {
        "events": events,
        "top_50": top_50,
        "date_folders": date_folders,
        "real_events_count": real_events_count,
        "unique_passes_count": unique_passes_count,
        "starred_count": starred_count,
        "night_count": night_count,
        "day_count": day_count,
        "by_category": by_category,
    }
    with _CACHE_LOCK:
        _DB_EVENTS_CACHE["key"] = db_key
        _DB_EVENTS_CACHE["bundle"] = bundle
    return bundle


def get_events_and_stats() -> Dict[str, Any]:
    b = _get_cached_db_events_bundle()

    live_status = {}
    if os.path.exists(LIVE_STATUS_FILE):
        try:
            with open(LIVE_STATUS_FILE, "r", encoding="utf-8") as f:
                live_status = json.load(f)
        except Exception:
            pass

    hist_mtime = _safe_mtime(HISTORY_FILE)
    if hist_mtime != _HISTORY_COUNT_CACHE["mtime"]:
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                h_data = json.load(f)
                _HISTORY_COUNT_CACHE["count"] = len(h_data.get("processed_frames", {}))
                _HISTORY_COUNT_CACHE["mtime"] = hist_mtime
        except Exception:
            pass
    processed_total = _HISTORY_COUNT_CACHE["count"]

    cfg = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            pass

    discord_allsky_webhook = str(cfg.get("discord_webhook_url") or cfg.get("allsky_webhook_url") or "")
    discord_default_webhook = ""
    if os.path.exists(DISCORD_CONFIG_FILE):
        try:
            with open(DISCORD_CONFIG_FILE, "r", encoding="utf-8") as f:
                d_cfg = json.load(f)
                whs = d_cfg.get("webhooks") or {}
                discord_allsky_webhook = whs.get("allsky") or whs.get("all sky") or discord_allsky_webhook
                discord_default_webhook = d_cfg.get("default_webhook_url") or whs.get("default") or ""
        except Exception:
            pass

    feedback_summary = {}
    try:
        import feedback_learner
        feedback_summary = feedback_learner.get_feedback_summary()
    except Exception as e:
        logger.warning(f"Failed loading feedback summary: {e}")

    adsb_snap = {}
    try:
        import adsb_tracker
        adsb_snap = adsb_tracker.get_live_adsb_snapshot()
    except Exception as e:
        logger.debug(f"Failed loading ADS-B snapshot: {e}")

    realign_state = {}
    try:
        import camera_realign
        realign_state = camera_realign.load_realign_state()
    except Exception as e:
        logger.debug(f"Failed loading realignment state: {e}")

    cameras_list = []
    try:
        import detector
        cameras_list = detector.get_configured_cameras(cfg, enabled_only=False)
    except Exception as e:
        logger.debug(f"Failed loading configured cameras: {e}")

    return {
        "events": b["events"],
        "top_50": b["top_50"],
        "date_folders": b["date_folders"],
        "feedback": feedback_summary,
        "adsb": adsb_snap,
        "realignment": realign_state,
        "cameras": cameras_list,
        "stats": {
            "total_events": b["real_events_count"],
            "unique_passes": b["unique_passes_count"],
            "starred_count": b["starred_count"],
            "night_events": b["night_count"],
            "day_events": b["day_count"],
            "processed_frames": processed_total,
            "by_category": b["by_category"],
            "live": live_status,
            "allsky_webhook_configured": bool(discord_allsky_webhook.startswith("https://discord")),
            "default_webhook_configured": bool(discord_default_webhook.startswith("https://discord")),
            "allsky_webhook_url": discord_allsky_webhook,
        },
        "config": cfg,
    }


def get_serialized_stats_response() -> Tuple[str, bytes, bytes]:
    """
    Returns (etag, raw_utf8_bytes, gzip_bytes) for `/api/stats` and `/api/events`,
    reusing cached serialized + gzipped bytes whenever underlying state has not changed.
    """
    adsb_key: Tuple[Any, ...] = ()
    try:
        import adsb_tracker
        snap = adsb_tracker.get_live_adsb_snapshot()
        adsb_key = (
            bool(snap.get("local_receiver_ok")),
            int(snap.get("aircraft_in_dome_count") or 0),
            int(snap.get("aircraft_in_range_count") or 0),
            tuple(str(a.get("callsign") or "") for a in (snap.get("aircraft_in_dome") or [])[:2]),
        )
    except Exception:
        pass

    sig = (
        _safe_mtime(DB_FILE),
        _safe_mtime(DB_FILE + "-wal"),
        _EVENTS_MUTATION_SEQ,
        _safe_mtime(LIVE_STATUS_FILE),
        _safe_mtime(REALIGN_STATE_FILE),
        _safe_mtime(CONFIG_FILE),
        _safe_mtime(DISCORD_CONFIG_FILE),
        adsb_key,
    )
    with _CACHE_LOCK:
        if _STATS_HTTP_CACHE["sig"] == sig and _STATS_HTTP_CACHE["raw_bytes"]:
            return _STATS_HTTP_CACHE["etag"], _STATS_HTTP_CACHE["raw_bytes"], _STATS_HTTP_CACHE["gzip_bytes"]

    payload = get_events_and_stats()
    raw_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    gzip_bytes = gzip.compress(raw_bytes, compresslevel=5)
    etag = f'W/"{hash(sig) & 0xFFFFFFFFFFFFFFFF:016x}"'
    with _CACHE_LOCK:
        _STATS_HTTP_CACHE["sig"] = sig
        _STATS_HTTP_CACHE["etag"] = etag
        _STATS_HTTP_CACHE["raw_bytes"] = raw_bytes
        _STATS_HTTP_CACHE["gzip_bytes"] = gzip_bytes
    return etag, raw_bytes, gzip_bytes


PORTAL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BluffSky Allsky AI • Sky Anomaly & Meteor Observatory</title>
<style>
  :root {
    --bg: #070b14;
    --panel: #0f172a;
    --panel2: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --green: #34d399;
    --amber: #fbbf24;
    --red: #f87171;
    --purple: #a78bfa;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: system-ui, -apple-system, sans-serif;
    background: radial-gradient(circle at top, #0f172a 0%, #070b14 70%);
    color: var(--text); min-height: 100vh;
  }
  header {
    background: rgba(15, 23, 42, 0.92); backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border); padding: 14px 24px;
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;
    position: sticky; top: 0; z-index: 50;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand h1 { margin: 0; font-size: 1.25rem; font-weight: 700; color: #e0f2fe; }
  .brand span { font-size: 0.82rem; color: var(--muted); }
  .telemetry-bar { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  .pill {
    background: var(--panel2); border: 1px solid var(--border);
    padding: 5px 12px; border-radius: 99px; font-size: 0.8rem; font-weight: 600;
    display: inline-flex; align-items: center; gap: 6px;
  }
  .pill.night { background: #1e1b4b; border-color: #6366f1; color: #c7d2fe; }
  .pill.day { background: #0c4a6e; border-color: #0284c7; color: #bae6fd; }
  .pill.ok { background: rgba(16, 185, 129, 0.15); border-color: #059669; color: #6ee7b7; }
  .pill.warn { background: rgba(245, 158, 11, 0.15); border-color: #d97706; color: #fde68a; }
  .container { max-width: 1480px; margin: 22px auto; padding: 0 20px 50px 20px; }
  .top-grid { display: grid; grid-template-columns: 390px 1fr; gap: 20px; margin-bottom: 24px; }
  @media (max-width: 1024px) { .top-grid { grid-template-columns: 1fr; } }
  .card {
    background: rgba(15, 23, 42, 0.86); border: 1px solid var(--border);
    border-radius: 14px; padding: 18px 20px;
  }
  .card h2 { margin: 0 0 12px 0; font-size: 1.05rem; color: #e2e8f0; display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
  .live-box { position: relative; width: 100%; aspect-ratio: 1 / 1; background: #000; border-radius: 10px; overflow: hidden; border: 1px solid #1e293b; }
  .live-box img { width: 100%; height: 100%; object-fit: contain; display: block; cursor: pointer; }
  .kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 14px; }
  .kpi { background: var(--panel2); border: 1px solid var(--border); border-radius: 10px; padding: 11px 13px; cursor: pointer; transition: border-color 0.15s; }
  .kpi:hover { border-color: var(--accent); }
  .kpi .lbl { font-size: 0.74rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .kpi .val { font-size: 1.4rem; font-weight: 700; margin-top: 4px; color: var(--text); }
  .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); gap: 12px; margin-top: 12px; }
  label { display: block; font-size: 0.79rem; color: var(--muted); margin-bottom: 4px; }
  input[type="text"], input[type="password"], input[type="number"], select {
    width: 100%; padding: 7px 10px; background: #070b14; border: 1px solid var(--border);
    border-radius: 6px; color: var(--text); font-size: 0.86rem;
  }
  .btn {
    background: #2563eb; color: #fff; border: none; padding: 8px 14px;
    border-radius: 7px; font-weight: 600; font-size: 0.84rem; cursor: pointer;
    display: inline-flex; align-items: center; gap: 6px; text-decoration: none;
  }
  .btn:hover { background: #1d4ed8; }
  .btn.secondary { background: #334155; }
  .btn.secondary:hover { background: #475569; }
  .btn.emerald { background: #059669; }
  .btn.emerald:hover { background: #047857; }
  .btn.purple { background: #7c3aed; }
  .btn.purple:hover { background: #6d28d9; }
  .btn.amber { background: #d97706; }
  .btn.amber:hover { background: #b45309; }

  /* Primary View Tabs */
  .view-tabs {
    display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
    background: rgba(15, 23, 42, 0.9); border: 1px solid var(--border);
    padding: 10px 14px; border-radius: 12px; margin-bottom: 18px;
  }
  .view-tab {
    background: transparent; border: 1px solid transparent; color: var(--muted);
    padding: 9px 16px; border-radius: 9px; font-size: 0.92rem; font-weight: 700;
    cursor: pointer; display: inline-flex; align-items: center; gap: 8px; transition: all 0.15s;
  }
  .view-tab:hover { color: var(--text); background: rgba(51, 65, 85, 0.45); }
  .view-tab.active {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
    color: #fff; border-color: #38bdf8; box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3);
  }
  .view-tab.top50-tab.active {
    background: linear-gradient(135deg, #d97706 0%, #7c3aed 100%);
    border-color: #fbbf24; box-shadow: 0 4px 14px rgba(217, 119, 6, 0.35);
  }
  .tab-count {
    background: rgba(0, 0, 0, 0.35); padding: 2px 8px; border-radius: 99px; font-size: 0.76rem;
  }

  /* Sub-toolbar for Filters, Date Folders, Sort & Grouping */
  .org-toolbar {
    background: rgba(15, 23, 42, 0.75); border: 1px solid var(--border);
    border-radius: 12px; padding: 14px 16px; margin-bottom: 18px;
    display: flex; flex-direction: column; gap: 12px;
  }
  .toolbar-row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; justify-content: space-between; }
  .filters { display: flex; gap: 7px; flex-wrap: wrap; align-items: center; }
  .filter-btn {
    background: var(--panel2); border: 1px solid var(--border); color: var(--muted);
    padding: 6px 13px; border-radius: 99px; font-size: 0.81rem; font-weight: 600; cursor: pointer;
  }
  .filter-btn:hover { color: var(--text); border-color: #64748b; }
  .filter-btn.active { background: #0284c7; border-color: #38bdf8; color: #fff; }

  .date-folders-strip {
    display: flex; gap: 8px; overflow-x: auto; padding-bottom: 4px;
  }
  .date-folder-card {
    background: var(--panel2); border: 1px solid var(--border); border-radius: 9px;
    padding: 8px 13px; min-width: 165px; cursor: pointer; transition: all 0.15s;
  }
  .date-folder-card:hover { border-color: var(--accent); }
  .date-folder-card.active { background: rgba(2, 132, 199, 0.22); border-color: var(--accent); }
  .date-folder-card .df-title { font-weight: 700; font-size: 0.86rem; color: #e0f2fe; display: flex; justify-content: space-between; }
  .date-folder-card .df-sub { font-size: 0.74rem; color: var(--muted); margin-top: 3px; }

  /* Spotlight Strip on Latest Tab */
  .spotlight-box {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid #4f46e5; border-radius: 14px; padding: 16px 18px; margin-bottom: 22px;
  }
  .spotlight-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; margin-top: 12px;
  }

  /* Events Grid & Cards */
  .events-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 18px; }
  .event-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    overflow: hidden; display: flex; flex-direction: column; transition: transform 0.15s, border-color 0.15s;
    position: relative;
  }
  .event-card:hover { transform: translateY(-2px); border-color: var(--accent); }
  .event-card.rank-top3 { border-color: #fbbf24; box-shadow: 0 0 18px rgba(251, 191, 36, 0.12); }
  .event-thumb { position: relative; width: 100%; aspect-ratio: 1 / 1; background: #000; cursor: pointer; overflow: hidden; }
  .event-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform 0.25s; }
  .event-thumb:hover img { transform: scale(1.02); }
  .badge-cat {
    position: absolute; top: 10px; left: 10px; padding: 4px 10px; border-radius: 6px;
    font-size: 0.75rem; font-weight: 700; background: rgba(15, 23, 42, 0.9);
    border: 1px solid var(--accent); color: #e0f2fe; backdrop-filter: blur(4px);
  }
  .badge-rank {
    position: absolute; top: 10px; right: 10px; padding: 4px 10px; border-radius: 6px;
    font-size: 0.78rem; font-weight: 800; background: linear-gradient(135deg, #f59e0b, #d97706);
    color: #070b14; box-shadow: 0 2px 8px rgba(0,0,0,0.5);
  }
  .badge-score {
    position: absolute; bottom: 10px; left: 10px; padding: 3px 9px; border-radius: 6px;
    font-size: 0.74rem; font-weight: 700; background: rgba(15, 23, 42, 0.9);
    color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.45);
  }
  .badge-pass {
    position: absolute; top: 42px; left: 10px; padding: 3px 9px; border-radius: 6px;
    font-size: 0.72rem; font-weight: 700; background: rgba(124, 58, 237, 0.92);
    color: #fff; border: 1px solid #c4b5fd;
  }
  .badge-coords {
    position: absolute; bottom: 10px; right: 10px; padding: 3px 9px; border-radius: 6px;
    font-size: 0.74rem; font-weight: 600; background: rgba(15, 23, 42, 0.9);
    color: var(--green); border: 1px solid rgba(52, 211, 153, 0.4);
  }
  .event-body { padding: 13px 15px; flex: 1; display: flex; flex-direction: column; gap: 6px; }
  .event-header-row { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; }
  .event-title { font-weight: 700; font-size: 0.96rem; color: #f1f5f9; }
  .card-actions { display: flex; gap: 4px; }
  .icon-btn {
    background: var(--panel2); border: 1px solid var(--border); color: var(--muted);
    border-radius: 6px; padding: 3px 7px; font-size: 0.8rem; cursor: pointer;
  }
  .icon-btn:hover { color: #fff; border-color: #64748b; }
  .icon-btn.starred { color: #fbbf24; border-color: #fbbf24; background: rgba(251, 191, 36, 0.15); }
  .event-desc { font-size: 0.82rem; color: #cbd5e1; line-height: 1.38; }
  .event-meta { margin-top: auto; padding-top: 8px; border-top: 1px solid #1e293b; font-size: 0.75rem; color: var(--muted); display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px; }

  .pagination-bar {
    margin-top: 22px; display: flex; justify-content: center; align-items: center; gap: 12px;
  }

  /* Interactive Lightbox Modal with Unlimited Zoom/Pan, Prev/Next & Multi-Frame Pass Scrubber */
  #modal {
    display: none; position: fixed; inset: 0; background: rgba(4, 7, 15, 0.95);
    z-index: 200; flex-direction: column; align-items: center; justify-content: center; padding: 12px 16px;
    user-select: none;
  }
  #modal.open { display: flex; }
  .modal-top-bar {
    width: 100%; max-width: 1440px; display: flex; justify-content: space-between; align-items: center;
    padding: 6px 10px; color: var(--text); gap: 10px; flex-wrap: wrap; z-index: 210;
  }
  .modal-stage {
    position: relative; display: flex; align-items: center; justify-content: center;
    width: 94vw; max-width: 1440px; height: 78vh;
    overflow: hidden; border-radius: 12px; border: 1px solid var(--border);
    background: #02040a; cursor: grab; touch-action: none;
  }
  .modal-stage.dragging { cursor: grabbing; }
  .modal-stage img {
    max-width: 90vw; max-height: 76vh; border-radius: 6px; background: #000;
    transform-origin: center center; will-change: transform;
    user-select: none; -webkit-user-drag: none; pointer-events: none;
  }
  .modal-zoom-bar {
    display: inline-flex; align-items: center; gap: 4px;
    background: #0f172a; padding: 4px 8px; border-radius: 8px; border: 1px solid #334155;
  }
  .modal-stage-hint {
    position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
    background: rgba(15, 23, 42, 0.86); backdrop-filter: blur(8px);
    border: 1px solid rgba(148, 163, 184, 0.25); color: #cbd5e1;
    padding: 4px 12px; border-radius: 999px; font-size: 0.74rem; pointer-events: none;
    white-space: nowrap; z-index: 205;
  }
  .nav-arrow {
    position: absolute; top: 50%; transform: translateY(-50%);
    background: rgba(15, 23, 42, 0.88); border: 1px solid var(--border); color: #fff;
    width: 46px; height: 46px; border-radius: 99px; font-size: 1.3rem; cursor: pointer;
    display: flex; align-items: center; justify-content: center; z-index: 206;
  }
  .nav-arrow:hover { background: #2563eb; border-color: #38bdf8; }
  .nav-arrow.prev { left: 12px; }
  .nav-arrow.next { right: 12px; }
  .pass-scrubber {
    margin-top: 8px; display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; align-items: center;
    max-width: 1280px; z-index: 210;
  }
  .pass-step-btn {
    background: var(--panel2); border: 1px solid var(--border); color: var(--muted);
    padding: 5px 11px; border-radius: 7px; font-size: 0.78rem; font-weight: 600; cursor: pointer;
  }
  .pass-step-btn.active { background: #7c3aed; border-color: #c4b5fd; color: #fff; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div style="font-size:1.8rem;">🌌</div>
    <div>
      <h1>BluffSky Allsky AI • Sky Anomaly & Meteor Observatory</h1>
      <span>Real-Time Night & Day Sky Anomaly Detection • ZWO ASI676MC (3552×3552) + RTX 8000 VLM</span>
    </div>
  </div>
  <div class="telemetry-bar">
    <span id="modePill" class="pill night">🌙 Mode: --</span>
    <span id="expPill" class="pill">⏱️ Exp: --</span>
    <span id="moonPill" class="pill">🌕 Moon: --</span>
    <span id="adsbPill" class="pill ok">✈️ ADS-B: --</span>
    <span id="discordPill" class="pill ok">🔔 Discord #all-sky: Ready</span>
    <a href="/live" class="btn emerald" style="text-decoration:none;">🔭 Full-Screen Live Sky (3552×3552)</a>
    <a href="/discoveries" class="btn purple" style="text-decoration:none;">🌌 Deep-Space & Meteor Discovery Patrol</a>
    <a href="/astrophoto" class="btn amber" style="text-decoration:none;">✨ Milky Way & Deep-Sky Portraits</a>
    <a href="http://192.168.1.50/tar1090/" target="_blank" class="btn secondary">📡 ADBSTracker Map</a>
    <a href="http://192.168.1.100/" target="_blank" class="btn secondary">📷 Allsky Pi WebUI</a>
    <a href="http://localhost:8095/" target="_blank" class="btn secondary">🏠 Home IoT Portal</a>
  </div>
</header>

<div class="container">
  <!-- LIVE CAMERA REALIGNMENT STATUS BANNER (Shown when user arms realignment after moving camera) -->
  <div id="realignmentBanner" style="display:none;margin-bottom:16px;padding:14px 18px;border-radius:12px;border:1px solid #f59e0b;background:linear-gradient(135deg,rgba(120,53,15,0.45),rgba(15,23,42,0.96));box-shadow:0 8px 24px rgba(0,0,0,0.45);">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
      <div>
        <div id="realignBannerTitle" style="font-weight:800;font-size:1.0rem;color:#fde68a;display:flex;align-items:center;gap:8px;">
          🧭 Camera Realignment Armed — Waiting for Clear Starry Sky...
        </div>
        <div id="realignBannerMsg" style="font-size:0.84rem;color:#e2e8f0;margin-top:4px;line-height:1.4;">
          Watching incoming camera frames until >= 55 sharp stars & >= 10 catalog navigation stars match across 2 clear frames.
        </div>
        <div id="realignBannerSub" style="font-size:0.76rem;color:#94a3b8;margin-top:4px;font-family:monospace;"></div>
      </div>
      <div style="display:flex;gap:8px;align-items:center;">
        <button id="btnCancelRealign" class="btn secondary" style="padding:7px 13px;font-size:0.8rem;color:#fca5a5;border-color:rgba(248,113,113,0.45);" onclick="cancelCameraRealignment()">
          ✕ Cancel Realignment
        </button>
      </div>
    </div>
  </div>

  <!-- PRIMARY HERO FOCUS: Multi-Epoch Deep-Sky & Solar System Discovery Center -->
  <div class="card" style="margin-bottom:20px;border:1px solid rgba(56,189,248,0.5);background:linear-gradient(180deg,rgba(15,23,42,0.98),rgba(8,14,28,0.98));box-shadow:0 14px 40px rgba(0,0,0,0.65);">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:14px;border-bottom:1px solid rgba(51,65,85,0.6);padding-bottom:12px;">
      <div>
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
          <span style="font-size:1.3rem;font-weight:800;color:#f8fafc;letter-spacing:-0.01em;">
            🌌 Multi-Epoch Deep-Sky & Solar System Discovery Center
          </span>
          <span class="pill" style="background:rgba(16,185,129,0.18);color:#6ee7b7;border:1px solid rgba(16,185,129,0.4);font-size:0.76rem;">
            ✨ PRIMARY OBSERVATORY FOCUS
          </span>
        </div>
        <div id="heroSurveySub" style="font-size:0.84rem;color:#94a3b8;margin-top:4px;">
          J2000.0 Sidereal De-Rotation • Multi-Night Median Co-Addition • Reference Image Subtraction (ΔI = Night − Master) • 3-Night Collinear Tracklet Linker
        </div>
      </div>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        <button id="btnLiveBlinkToggle" class="btn amber" style="padding:7px 14px;font-size:0.82rem;" onclick="toggleHeroBlinkMode()">
          ⚡ Enable Live Blink Comparator (Ref ⇄ Night)
        </button>
        <a href="/discoveries" class="btn purple" style="padding:7px 14px;font-size:0.82rem;text-decoration:none;">
          🗺️ J2000 Sky Patrol & Nova Watchlist →
        </a>
        <a href="/astrophoto" class="btn" style="padding:7px 14px;font-size:0.82rem;text-decoration:none;background:#0284c7;">
          ✨ 4K Milky Way & Deep-Sky Portraits →
        </a>
        <button id="btnArmRealignTop" class="btn secondary" style="padding:7px 12px;font-size:0.8rem;border-color:rgba(56,189,248,0.45);color:#bae6fd;" onclick="armCameraRealignment()" title="Click ONLY after physically moving or rotating the camera to wait for a clear starry night and solve the new 3D alignment">
          🧭 Camera Moved — Realign
        </button>
        <button class="btn secondary" style="padding:7px 12px;font-size:0.8rem;" onclick="toggleSecondaryDrawer()">
          🎛️ Live Dome & Settings
        </button>
      </div>
    </div>

    <!-- 6 Hero Discovery KPIs -->
    <div id="heroDiscoveryKpis" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:10px;margin-bottom:16px;"></div>

    <!-- Master J2000.0 Co-Added Reference Sky Atlas Banner -->
    <div style="background:#050914;border:1px solid rgba(56,189,248,0.35);border-radius:12px;padding:12px;margin-bottom:18px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
        <div style="font-size:0.92rem;font-weight:700;color:#e0f2fe;">
          🗺️ Master J2000.0 Multi-Night Co-Added Star & Deep-Sky Atlas (Click to Open Full-Res 4000% Zoom & Pan Lightbox)
        </div>
        <span style="font-size:0.78rem;color:#6ee7b7;font-family:monospace;">
          🟢 Green: Recovered Planets & Ice Giant Uranus • 🟡 Gold: Messier Nebulae/Galaxies • 🔵 Cyan: 3-Night Moving Tracklets
        </span>
      </div>
      <img id="heroMasterAtlasImg" src="/survey_data/master_j2000_atlas.jpg" style="width:100%;border-radius:8px;cursor:zoom-in;display:block;border:1px solid #1e293b;" onclick="openCustomImageModal('Master J2000.0 Multi-Epoch Co-Added Reference Sky Atlas', '13 Clear Nights Co-Added (1,150 Sidereally De-Rotated Exposures • +3.83 Mag Depth Gain • Airplanes & Satellites Erased)', this.src)" />
    </div>

    <!-- Discovery Category Filter Pills & Blink Comparator Showcase Grid -->
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:12px;">
      <div style="display:flex;gap:8px;flex-wrap:wrap;" id="heroDiscFilterBar">
        <button class="filter-btn active" onclick="setHeroDiscFilter('all', this)">🌟 All Multi-Epoch Discoveries</button>
        <button class="filter-btn" onclick="setHeroDiscFilter('solsys', this)">🪐 Recovered Solar System Worlds (Mars, Jupiter, Uranus)</button>
        <button class="filter-btn" onclick="setHeroDiscFilter('tracklet', this)">🛸 Candidate Undiscovered Moving Tracklets (3-Night Link)</button>
        <button class="filter-btn" onclick="setHeroDiscFilter('transient', this)">✨ Multi-Frame Deep-Sky Optical Transients</button>
      </div>
      <div style="font-size:0.78rem;color:#fde68a;font-weight:600;" id="heroBlinkStatusText">
        👁️ Showing 3-Panel Triplet Cutouts (Click "⚡ Enable Live Blink Comparator" above to animate Reference ⇄ Discovery Night!)
      </div>
    </div>

    <div id="heroDiscoveriesGrid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(390px,1fr));gap:14px;"></div>
  </div>

  <!-- SECONDARY COLLAPSIBLE DRAWER: Live Allsky Dome & Detector/Discord Settings -->
  <div id="secondaryControlsDrawer" class="top-grid" style="display:none;margin-bottom:20px;">
    <!-- Left Column: Live Allsky Dome & Calibrated Sky Mask -->
    <div class="card">
      <h2>
        <span style="display:inline-flex;align-items:center;gap:8px;flex-wrap:wrap;">
          <span>🔭 Live Allsky Dome</span>
          <select id="liveCameraSelect" style="width:auto;padding:3px 8px;font-size:0.76rem;background:#0f172a;color:#e0f2fe;border:1px solid #334155;border-radius:6px;" onchange="refreshLiveImage(true)">
            <option value="cam1">cam1 (192.168.1.100)</option>
          </select>
        </span>
        <span style="display:inline-flex;gap:10px;align-items:center;flex-wrap:wrap;">
          <label style="margin:0;display:inline-flex;align-items:center;gap:5px;cursor:pointer;font-size:0.78rem;color:#fbbf24;">
            <input type="checkbox" id="showAdsbToggle" checked onchange="refreshLiveImage(true)"> ✈️ ADS-B
          </label>
          <label style="margin:0;display:inline-flex;align-items:center;gap:5px;cursor:pointer;font-size:0.78rem;color:#38bdf8;">
            <input type="checkbox" id="showSatsToggle" checked onchange="refreshLiveImage(true)"> 🛰️ Satellites
          </label>
          <label style="margin:0;display:inline-flex;align-items:center;gap:5px;cursor:pointer;font-size:0.78rem;color:#6ee7b7;">
            <input type="checkbox" id="showMaskToggle" checked onchange="refreshLiveImage(true)"> Sky Mask
          </label>
        </span>
      </h2>
      <div class="live-box">
        <img id="liveSkyImg" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" data-lazy="1" onclick="window.location.href='/live'" title="Click to open Full-Resolution 3552×3552 Live Sky Radar" />
      </div>
      <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;font-size:0.78rem;color:var(--muted);">
        <span id="lastFrameText">Frame: --</span>
        <div style="display:flex;gap:6px;">
          <button class="btn secondary" style="padding:5px 10px;font-size:0.76rem;" onclick="refreshLiveImage(true)">🔄 Refresh</button>
          <a href="/live" class="btn emerald" style="padding:5px 11px;font-size:0.76rem;text-decoration:none;">⛶ Full-Res Live Page (3552×3552)</a>
        </div>
      </div>
    </div>

    <!-- Right Column: KPIs, Discord Channel Setup & Sensitivity Controls -->
    <div class="card">
      <h2>
        <span>⚡ Sky Event Telemetry & <code>#all-sky</code> Controls</span>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <button class="btn amber" onclick="switchView('top50')">🏆 Browse Top 50 Most Interesting</button>
          <button class="btn purple" onclick="runSimTest('meteor')">🌠 Test Meteor</button>
          <button class="btn emerald" onclick="runSimTest('fireball')">🔥 Test Fireball</button>
        </div>
      </h2>

      <div class="kpi-row">
        <div class="kpi" onclick="switchView('top50')" title="Click to browse Top 50 Most Interesting">
          <div class="lbl">🏆 Top Interesting</div>
          <div class="val" id="kpiTop50" style="color:#fbbf24;">0</div>
        </div>
        <div class="kpi" onclick="quickCategoryFilter('meteor')">
          <div class="lbl">🌠 Meteors / Bolides</div>
          <div class="val" id="kpiMeteors" style="color:#34d399;">0</div>
        </div>
        <div class="kpi" onclick="quickCategoryFilter('space_object')">
          <div class="lbl">🛰️ Space / Flashes</div>
          <div class="val" id="kpiSpace" style="color:#38bdf8;">0</div>
        </div>
        <div class="kpi" onclick="quickCategoryFilter('aircraft')">
          <div class="lbl">✈️ Aircraft Transits</div>
          <div class="val" id="kpiAircraft" style="color:#a78bfa;">0</div>
        </div>
        <div class="kpi" onclick="switchView('archive')">
          <div class="lbl">Unique Sky Passes</div>
          <div class="val" id="kpiPasses">0</div>
        </div>
        <div class="kpi">
          <div class="lbl">Frames Analyzed</div>
          <div class="val" id="kpiFrames">0</div>
        </div>
      </div>

      <!-- Discord #all-sky Webhook & Active Learning Bot Box -->
      <div style="background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:12px 14px;margin-bottom:12px;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:6px;">
          <div style="font-weight:600;font-size:0.86rem;color:#e0f2fe;">
            🔔 Discord <code>#all-sky</code> Channel Webhook & 🤖 Active Learning Bot
          </div>
          <span id="botStatusPill" class="pill warn" style="font-size:0.74rem;padding:3px 9px;">🤖 Bot: Setup Token</span>
        </div>
        <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;">
          <input type="password" id="allskyWebhookUrl" placeholder="Paste https://discord.com/api/webhooks/... for your #all-sky channel" />
          <button class="btn secondary" onclick="const i=document.getElementById('allskyWebhookUrl');i.type=i.type==='password'?'text':'password';">Show</button>
        </div>
        <div style="display:grid;grid-template-columns:1fr 220px auto;gap:8px;align-items:center;">
          <input type="password" id="discordBotToken" placeholder="🤖 Discord Bot Token (enables replying/reacting in #all-sky to train AI)" />
          <input type="text" id="discordClientId" placeholder="Bot Application/Client ID (optional)" />
          <div style="display:flex;gap:6px;">
            <button class="btn secondary" onclick="const i=document.getElementById('discordBotToken');i.type=i.type==='password'?'text':'password';">Show</button>
            <button class="btn" onclick="saveSettings()">Save Settings</button>
          </div>
        </div>
        <div style="margin-top:7px;font-size:0.76rem;color:#cbd5e1;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;">
          <span id="botHelperText">💡 Reply to any Discord image with <code>plane</code>, <code>meteor</code>, <code>satellite</code>, <code>fireball</code>, <code>flash</code>, or <code>false</code> (or react ☄️ 🔥 🛰️ ✈️ ✨ ❌) to train the classifier!</span>
          <a id="botInviteLink" href="https://discord.com/developers/applications" target="_blank" style="color:var(--accent);font-weight:700;text-decoration:none;">🔗 Discord Developer Portal →</a>
        </div>
      </div>

      <!-- Detector Sensitivity & Mode Toggles -->
      <div class="form-grid">
        <div>
          <label>Night Anomaly Sigma (<code>night_k_sigma</code>)</label>
          <input type="number" step="0.2" min="2.5" max="10.0" id="cfgNightSigma" value="4.8" />
        </div>
        <div>
          <label>Min Meteor Length (px on 3552×3552)</label>
          <input type="number" step="2" min="15" max="300" id="cfgMeteorLen" value="36" />
        </div>
        <div>
          <label>Night Aircraft Strobe Alerts</label>
          <select id="cfgAlertAircraft">
            <option value="true">Enabled (Alert on Night Planes)</option>
            <option value="false">Muted (Meteors & Space Objects Only)</option>
          </select>
        </div>
        <div>
          <label>Daytime Sky Anomaly Detection</label>
          <select id="cfgDayEnabled">
            <option value="true">Enabled</option>
            <option value="false">Disabled (Night Only)</option>
          </select>
        </div>
      </div>
      <div id="saveStatus" style="margin-top:8px;font-size:0.82rem;font-weight:600;"></div>
    </div>
  </div>

  <!-- Primary Navigation View Tabs (Secondary Single-Frame Captures) -->
  <div class="view-tabs">
    <button id="tab-top50" class="view-tab top50-tab active" onclick="switchView('top50')">
      🏆 Top 50 Space & Meteor Highlights <span class="tab-count" id="cntTop50">0</span>
    </button>
    <button id="tab-latest" class="view-tab" onclick="switchView('latest')">
      ⚡ Last 50 All-Sky Logs <span class="tab-count" id="cntLatest">50</span>
    </button>
    <button id="tab-archive" class="view-tab" onclick="switchView('archive')">
      🗂️ Organized Archive (By Night & Category) <span class="tab-count" id="cntArchive">0</span>
    </button>
    <button id="tab-starred" class="view-tab" onclick="switchView('starred')">
      ⭐ Starred Favorites <span class="tab-count" id="cntStarred">0</span>
    </button>
    <button id="tab-feedback" class="view-tab" onclick="switchView('feedback')">
      🧠 AI Learning & Corrections <span class="tab-count" id="cntFeedback">0</span>
    </button>
    <div style="margin-left:auto;display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
      <button class="btn amber" style="padding:7px 13px;font-size:0.8rem;" onclick="openSlideshowTop50()">
        🎬 Launch Top 50 Fullscreen
      </button>
      <button class="btn secondary" style="padding:7px 13px;font-size:0.8rem;" onclick="openSlideshowLatest50()">
        🖼️ Launch Last 50 Fullscreen
      </button>
    </div>
  </div>

  <!-- Spotlight Top Highlights Box (Shown only on Latest View) -->
  <div id="spotlightSection" class="spotlight-box" style="display:none;">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
      <div>
        <span style="font-weight:800;font-size:1.02rem;color:#fef08a;">🏆 Highest-Ranked Sky Highlights</span>
        <span style="font-size:0.82rem;color:#cbd5e1;margin-left:8px;">Top-scoring captures ranked by astronomical rarity, streak length, SNR & sky altitude</span>
      </div>
      <button class="btn amber" style="padding:6px 12px;font-size:0.78rem;" onclick="switchView('top50')">
        Browse All Top 50 Most Interesting →
      </button>
    </div>
    <div id="spotlightGrid" class="spotlight-grid"></div>
  </div>

  <!-- Organization & Filtering Toolbar -->
  <div class="org-toolbar">
    <!-- Date Folders Strip (Shown in Archive View) -->
    <div id="dateFoldersRow" style="display:none;">
      <div style="font-size:0.78rem;color:var(--muted);margin-bottom:6px;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;">
        📅 Select Observing Date / Night Folder:
      </div>
      <div id="dateFoldersStrip" class="date-folders-strip"></div>
    </div>

    <!-- Category Pills + Sort / Grouping Controls -->
    <div class="toolbar-row">
      <div class="filters">
        <button class="filter-btn active" data-cat="all" onclick="setFilter('all', this)">All Categories</button>
        <button class="filter-btn" data-cat="meteor_fireball" onclick="setFilter('meteor_fireball', this)">🌠 Meteors & Fireballs</button>
        <button class="filter-btn" data-cat="space_object" onclick="setFilter('space_object', this)">🛰️ Satellites & Space</button>
        <button class="filter-btn" data-cat="transient_flash" onclick="setFilter('transient_flash', this)">✨ Flashes & Phenomena</button>
        <button class="filter-btn" data-cat="no_aircraft" onclick="setFilter('no_aircraft', this)">🌌 Non-Aircraft Only</button>
        <button class="filter-btn" data-cat="aircraft" onclick="setFilter('aircraft', this)">✈️ Aircraft</button>
        <button class="filter-btn" data-cat="DAY" onclick="setFilter('DAY', this)">☀️ Daytime</button>
      </div>

      <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
        <label style="margin:0;display:inline-flex;align-items:center;gap:5px;cursor:pointer;font-size:0.8rem;color:#cbd5e1;" title="Bundles consecutive frames of the same airplane or satellite transit into 1 clean card">
          <input type="checkbox" id="groupPassesToggle" checked onchange="resetPageAndRender()"> 🎞️ Group Multi-Frame Passes
        </label>
        <label style="margin:0;display:inline-flex;align-items:center;gap:5px;cursor:pointer;font-size:0.8rem;color:#94a3b8;">
          <input type="checkbox" id="showTestsToggle" onchange="resetPageAndRender()"> 🧪 Include Test Alerts
        </label>
        <select id="cameraFilterSelect" style="width:auto;padding:6px 10px;font-size:0.82rem;" onchange="resetPageAndRender()">
          <option value="all">🎥 All Cameras (1..N)</option>
        </select>
        <select id="sortSelect" style="width:auto;padding:6px 10px;font-size:0.82rem;" onchange="resetPageAndRender()">
          <option value="default">Sort: View Default</option>
          <option value="interest">🔥 Most Interesting Score</option>
          <option value="newest">🕒 Newest First</option>
          <option value="snr">⚡ Highest SNR (σ)</option>
          <option value="length">📏 Longest Streak (px)</option>
        </select>
        <input type="text" id="searchBox" placeholder="🔍 Search subject, direction (NNE, SE)..." style="width:210px;padding:6px 10px;font-size:0.82rem;" oninput="resetPageAndRender()" />
      </div>
    </div>
  </div>

  <!-- Section Title Banner -->
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:8px;">
    <div id="sectionBannerTitle" style="font-size:1.05rem;font-weight:700;color:#e0f2fe;">⚡ Latest Sky Captures</div>
    <div id="sectionBannerSub" style="font-size:0.82rem;color:var(--muted);"></div>
  </div>

  <!-- Events Gallery -->
  <div id="eventsGrid" class="events-grid"></div>

  <!-- Pagination Controls -->
  <div id="paginationBar" class="pagination-bar"></div>
</div>

<!-- Lightbox Modal with Unlimited Zoom/Pan, Prev/Next & Multi-Frame Pass Scrubber -->
<div id="modal" onclick="if(event.target.id==='modal') closeModal()">
  <div class="modal-top-bar">
    <div>
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
        <div id="modalTitle" style="font-weight:700;font-size:1.05rem;color:#f8fafc;"></div>
        <span id="modalBlinkPhasePill" style="display:none;padding:3px 10px;border-radius:99px;font-size:0.74rem;font-weight:800;font-family:monospace;"></span>
      </div>
      <div id="modalSubtitle" style="font-size:0.82rem;color:var(--muted);margin-top:2px;"></div>
    </div>
    <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
      <div id="modalBlinkBar" style="display:none;gap:5px;align-items:center;background:#0f172a;padding:4px 8px;border-radius:8px;border:1px solid #334155;">
        <button class="btn" id="pBtnBlinkAuto" style="padding:4px 10px;font-size:0.75rem;" onclick="setPortalModalBlinkMode('auto')">⚡ Live Blink (Ref ⇄ Night)</button>
        <button class="btn secondary" id="pBtnBlinkRef" style="padding:4px 9px;font-size:0.75rem;" onclick="setPortalModalBlinkMode('ref')">1️⃣ Reference</button>
        <button class="btn secondary" id="pBtnBlinkSci" style="padding:4px 9px;font-size:0.75rem;" onclick="setPortalModalBlinkMode('sci')">2️⃣ Discovery Night</button>
        <button class="btn secondary" id="pBtnBlinkTriplet" style="padding:4px 9px;font-size:0.75rem;" onclick="setPortalModalBlinkMode('triplet')">🖼️ 3-Panel Triplet</button>
      </div>
      <div class="modal-zoom-bar">
        <button class="icon-btn" onclick="zoomModalBy(0.74)" title="Zoom Out (Key: - or Scroll Down)">➖</button>
        <button id="modalZoomLabel" class="icon-btn" style="min-width:58px;color:#38bdf8;font-weight:700;" onclick="resetModalZoom()" title="Current Zoom Level — Click to Reset to 100% Fit (Key: 0)">100%</button>
        <button class="icon-btn" onclick="zoomModalBy(1.35)" title="Zoom In (Key: + or Scroll Up)">➕</button>
        <button id="modalFocusBoxBtn" class="icon-btn" style="color:#34d399;border-color:rgba(52,211,153,0.4);" onclick="focusModalTargetBox()" title="Zoom directly into the detected event box (Key: F)">🎯 Focus Target</button>
        <button class="icon-btn" onclick="resetModalZoom()" title="Reset Zoom & Pan to Fit (Key: 0)">🔄 Fit</button>
      </div>
      <div id="modalTeachBar" style="display:inline-flex;gap:4px;align-items:center;background:#0f172a;padding:4px 8px;border-radius:8px;border:1px solid #334155;">
        <span style="font-size:0.74rem;color:#94a3b8;margin-right:3px;">🧠 Teach AI:</span>
        <button class="icon-btn" onclick="reclassifyModal('meteor')" title="Teach AI: Shooting Star / Meteor">☄️ Meteor</button>
        <button class="icon-btn" onclick="reclassifyModal('fireball')" title="Teach AI: Fireball / Bolide">🔥 Fireball</button>
        <button class="icon-btn" onclick="reclassifyModal('space_object')" title="Teach AI: Satellite / Space">🛰️ Sat</button>
        <button class="icon-btn" onclick="reclassifyModal('aircraft')" title="Teach AI: Aircraft">✈️ Plane</button>
        <button class="icon-btn" onclick="reclassifyModal('transient_flash')" title="Teach AI: Transient Flash">✨ Flash</button>
        <button class="icon-btn" style="color:#fca5a5;" onclick="reclassifyModal('false_positive')" title="Teach AI: False Positive (Cloud/Glare/Hotspot)">❌ False</button>
      </div>
      <button id="modalStarBtn" class="btn secondary" style="padding:6px 12px;" onclick="toggleModalStar()">⭐ Star</button>
      <button id="modalHideBtn" class="btn secondary" style="padding:6px 12px;color:#fca5a5;" onclick="hideModalEvent()">🗑️ Hide</button>
      <button class="btn secondary" style="padding:6px 12px;" onclick="closeModal()">✕ Close (Esc)</button>
    </div>
  </div>
  <div class="modal-stage" id="modalStage">
    <button id="modalPrevBtn" class="nav-arrow prev" onclick="event.stopPropagation(); stepModal(-1)" title="Previous Image (Left Arrow)">❮</button>
    <img id="modalImg" src="" draggable="false" />
    <button id="modalNextBtn" class="nav-arrow next" onclick="event.stopPropagation(); stepModal(1)" title="Next Image (Right Arrow)">❯</button>
    <div class="modal-stage-hint">🖱️ Scroll Wheel: Zoom In/Out As Much As You Want (10% – 4000%) • Click & Drag: Pan • Double-Click: Zoom 3× • Keys: + / - / 0 / F</div>
  </div>
  <div id="modalPassScrubber" class="pass-scrubber"></div>
</div>

<script>
let ALL_EVENTS = [];
let TOP_50 = [];
let DATE_FOLDERS = [];
let FEEDBACK_SUMMARY = {};
let CONFIGURED_CAMERAS = [];
let CURRENT_VIEW = 'top50'; // 'latest' | 'top50' | 'archive' | 'starred' | 'feedback'
let CURRENT_FILTER = 'all';
let CURRENT_DATE = 'all';
let CURRENT_PAGE = 1;
const PAGE_SIZE_ARCHIVE = 24;
const LATEST_MAX_CARDS = 50;
let INITIAL_LOADED = false;

// Currently displayed list in the grid for modal Prev/Next navigation
let ACTIVE_DISPLAY_LIST = [];
let MODAL_INDEX = -1;
let MODAL_PASS_FRAME_INDEX = 0;

const CAT_ICONS = {
  fireball: '🔥 Fireball / Bolide',
  meteor: '🌠 Shooting Star',
  space_object: '🛰️ Space Object',
  transient_flash: '✨ Space Flash',
  sky_phenomenon: '🌌 Sky Phenomenon',
  aircraft: '✈️ Night Aircraft',
  day_aircraft: '🛩️ Day Aircraft',
  day_bird: '🦅 Soaring Bird',
  day_anomaly: '☀️ Day Anomaly'
};

function switchView(view) {
  CURRENT_VIEW = view;
  CURRENT_PAGE = 1;
  document.querySelectorAll('.view-tab').forEach(b => b.classList.remove('active'));
  const tabBtn = document.getElementById(`tab-${view}`);
  if (tabBtn) tabBtn.classList.add('active');

  document.getElementById('dateFoldersRow').style.display = (view === 'archive') ? 'block' : 'none';
  renderAll();
}

function quickCategoryFilter(cat) {
  CURRENT_VIEW = 'archive';
  CURRENT_PAGE = 1;
  document.querySelectorAll('.view-tab').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-archive').classList.add('active');
  document.getElementById('dateFoldersRow').style.display = 'block';

  const mapped = (cat === 'meteor') ? 'meteor_fireball' : cat;
  CURRENT_FILTER = mapped;
  document.querySelectorAll('.filter-btn').forEach(b => {
    b.classList.toggle('active', b.getAttribute('data-cat') === mapped);
  });
  renderAll();
}

function setFilter(f, btn) {
  CURRENT_FILTER = f;
  CURRENT_PAGE = 1;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  renderAll();
}

function selectDateFolder(d) {
  CURRENT_DATE = d;
  CURRENT_PAGE = 1;
  renderDateFolders();
  renderAll();
}

function resetPageAndRender() {
  CURRENT_PAGE = 1;
  renderAll();
}

function matchesCategoryFilter(ev, filter) {
  if (filter === 'all') return true;
  if (filter === 'DAY') return ev.mode === 'DAY';
  if (filter === 'meteor_fireball') return ev.category === 'meteor' || ev.category === 'fireball';
  if (filter === 'transient_flash') return ev.category === 'transient_flash' || ev.category === 'sky_phenomenon';
  if (filter === 'no_aircraft') return ev.category !== 'aircraft' && ev.category !== 'day_aircraft';
  if (filter === 'aircraft') return ev.category === 'aircraft' || ev.category === 'day_aircraft';
  return ev.category === filter;
}

function renderDateFolders() {
  const strip = document.getElementById('dateFoldersStrip');
  if (!strip) return;
  const totalReal = DATE_FOLDERS.reduce((acc, d) => acc + d.total, 0);
  const totalPasses = DATE_FOLDERS.reduce((acc, d) => acc + d.passes, 0);

  let html = `
    <div class="date-folder-card ${CURRENT_DATE === 'all' ? 'active' : ''}" onclick="selectDateFolder('all')">
      <div class="df-title"><span>🗂️ All Observing Dates</span><span>${totalPasses}</span></div>
      <div class="df-sub">${totalReal} total frames captured</div>
    </div>
  `;
  html += DATE_FOLDERS.map(d => `
    <div class="date-folder-card ${CURRENT_DATE === d.date ? 'active' : ''}" onclick="selectDateFolder('${d.date}')">
      <div class="df-title"><span>📅 ${d.date}</span><span>${d.passes} passes</span></div>
      <div class="df-sub">🌠 ${d.meteors} • 🛰️ ${d.space} • ✈️ ${d.aircraft} (${d.total} imgs)</div>
    </div>
  `).join('');
  strip.innerHTML = html;
}

function getBaseRawFrameName(fname) {
  return String(fname || '').replace(/_obj\d+(?=\.jpg$|$)/i, '');
}

function getSameFrameSiblings(fname) {
  const base = getBaseRawFrameName(fname);
  if (!base) return [];
  return (ALL_EVENTS || []).filter(x => getBaseRawFrameName(x.frame_filename) === base);
}

function openEventByFilename(frameFilename, domEvent) {
  if (domEvent) domEvent.stopPropagation();
  let idx = ACTIVE_DISPLAY_LIST.findIndex(x => x.frame_filename === frameFilename);
  if (idx >= 0) {
    openEventModal(idx);
    return;
  }
  const fullEv = (ALL_EVENTS || []).find(x => x.frame_filename === frameFilename);
  if (fullEv) {
    ACTIVE_DISPLAY_LIST = [fullEv, ...ACTIVE_DISPLAY_LIST];
    openEventModal(0);
  }
}

function buildEventCardHtml(ev, idxInList, showRankRibbon) {
  const snapUrl = `/snapshots/${encodeURIComponent(ev.snapshot_filename || '')}`;
  const catBadge = CAT_ICONS[ev.category] || ev.category;
  const desc = ev.vlm_description || `${ev.subtype || ev.subject} detected at ${ev.snr || 0}σ SNR (${ev.length_px || 0}px streak).`;
  const score = Number(ev.interest_score || 0).toFixed(1);
  const rank = ev.display_rank || ev.top_rank || 0;
  const isTop3 = showRankRibbon && rank >= 1 && rank <= 3;
  const rankMedal = rank === 1 ? '🥇 #1' : (rank === 2 ? '🥈 #2' : (rank === 3 ? '🥉 #3' : `#${rank}`));
  const passCount = Number(ev.pass_count || 1);
  const groupPasses = document.getElementById('groupPassesToggle').checked;
  const userCorrected = ev.user_corrected_category ? `<span style="color:#6ee7b7;font-weight:700;font-size:0.73rem;">✅ Human Verified</span>` : '';
  const camBadge = ((CONFIGURED_CAMERAS && CONFIGURED_CAMERAS.length > 1) || (ev.camera_id && ev.camera_id !== 'cam1'))
    ? `<span style="color:#38bdf8;font-weight:700;font-size:0.72rem;background:rgba(14,116,144,0.22);padding:1px 6px;border-radius:4px;border:1px solid rgba(56,189,248,0.35);">🎥 ${ev.camera_name || ev.camera_id || 'cam1'}</span>`
    : '';

  let tagBannerHtml = '';
  let multiObjBannerHtml = '';
  let stereoBannerHtml = '';
  try {
    const fj = ev.features_json ? (typeof ev.features_json === 'string' ? JSON.parse(ev.features_json) : ev.features_json) : {};
    if (fj && fj.stereo_match) {
      const sm = fj.stereo_match;
      const altTxt = sm.triangulated_alt_km != null ? ` • 3D Alt ≈ ${sm.triangulated_alt_km} km (${sm.triangulated_alt_miles} mi)` : '';
      stereoBannerHtml = `<div style="margin:4px 0;padding:4px 8px;border-radius:7px;background:rgba(6,78,59,0.28);border:1px solid rgba(52,211,153,0.48);color:#6ee7b7;font-size:0.72rem;font-weight:700;">
        🤝 STEREO COINCIDENCE (${sm.partner_camera_name || sm.partner_camera_id}): Δθ=${sm.parallax_deg}° • Δt=${sm.dt_sec}s${altTxt}
      </div>`;
    }
    const siblings = getSameFrameSiblings(ev.frame_filename);
    const totalInFrame = Math.max(Number(fj.objects_in_frame || 1), siblings.length);
    let objIdx = Number(fj.object_index || 0);
    if (!objIdx && totalInFrame > 1) {
      const m = String(ev.frame_filename || '').match(/_obj(\d+)/i);
      objIdx = m ? Number(m[1]) : 1;
    }
    if (totalInFrame > 1) {
      const otherChips = siblings
        .filter(s => s.frame_filename !== ev.frame_filename)
        .map(s => {
          const sIcon = (CAT_ICONS[s.category] || '🎯').split(' ')[0];
          const sTitle = String(s.subject || s.subtype || s.category).slice(0, 26);
          return `<button class="icon-btn" style="padding:1px 6px;font-size:0.69rem;border-color:rgba(167,139,250,0.45);color:#ddd6fe;" onclick="openEventByFilename('${s.frame_filename}', event)" title="Inspect sibling object in this same exposure">↗ ${sIcon} ${sTitle}</button>`;
        })
        .join(' ');
      multiObjBannerHtml = `<div style="margin:4px 0;padding:4px 8px;border-radius:7px;background:rgba(109,40,217,0.18);border:1px solid rgba(167,139,250,0.42);color:#e9d5ff;font-size:0.72rem;font-weight:700;display:flex;align-items:center;justify-content:space-between;gap:6px;flex-wrap:wrap;">
        <span>🎯 Object ${objIdx || 1} of ${totalInFrame} in Frame (Independent Review)</span>
        <span style="display:flex;gap:4px;flex-wrap:wrap;">${otherChips}</span>
      </div>`;
    }
    if (fj && fj.orbital_object && fj.orbital_object.name) {
      const o = fj.orbital_object;
      const altKm = Number(o.orbit_alt_km || 0);
      const reg = o.orbit_short || (altKm >= 33000 ? 'GEO Deep-Space' : (altKm >= 2000 ? 'MEO' : 'LEO'));
      const spd = o.speed_mph ? `${Number(o.speed_mph).toLocaleString()} mph` : `${o.speed_kms || ''} km/s`;
      tagBannerHtml = `<div style="margin:5px 0;padding:5px 9px;border-radius:7px;background:rgba(14,116,144,0.22);border:1px solid rgba(56,189,248,0.45);color:#7dd3fc;font-size:0.74rem;font-weight:700;">
        🛰️ SAT TAG: ${o.name} (NORAD #${o.norad_id}) • ${reg} ${altKm.toLocaleString()} km (${Number(o.orbit_alt_miles || 0).toLocaleString()} mi) • ${spd}
      </div>`;
    } else if (fj && fj.adsb_flight && fj.adsb_flight.callsign) {
      const a = fj.adsb_flight;
      tagBannerHtml = `<div style="margin:5px 0;padding:5px 9px;border-radius:7px;background:rgba(180,83,9,0.22);border:1px solid rgba(251,191,36,0.45);color:#fcd34d;font-size:0.74rem;font-weight:700;">
        ✈️ FLIGHT TAG: ${a.callsign} (${a.model_name || a.type_code || 'ADS-B'}) • ${Number(a.alt_ft || 0).toLocaleString()} ft • ${Math.round(a.gs_kt || 0)} kt
      </div>`;
    }
  } catch (e) {}

  return `
    <div class="event-card ${isTop3 ? 'rank-top3' : ''}">
      <div class="event-thumb" onclick="openEventModal(${idxInList})">
        <img src="${snapUrl}" loading="lazy" />
        <span class="badge-cat">${catBadge}${ev.is_test ? ' (TEST)' : ''}</span>
        ${(showRankRibbon && rank > 0) ? `<span class="badge-rank">${rankMedal} Most Interesting</span>` : (rank > 0 && rank <= 10 ? `<span class="badge-rank">🏆 Top #${rank}</span>` : '')}
        ${(groupPasses && passCount > 1) ? `<span class="badge-pass">🎞️ ${passCount}-Frame Pass (${ev.pass_time_span || 'click to scrub'})</span>` : ''}
        <span class="badge-score">🔥 Score ${score}</span>
        <span class="badge-coords">Alt ${Number(ev.alt_deg || 0).toFixed(0)}° • ${ev.cardinal || ''} (${Number(ev.az_deg || 0).toFixed(0)}°)</span>
      </div>
      <div class="event-body">
        <div class="event-header-row">
          <div class="event-title">${ev.subject || ev.subtype} ${camBadge} ${userCorrected}</div>
          <div class="card-actions">
            <button class="icon-btn ${ev.is_starred ? 'starred' : ''}" onclick="toggleStar('${ev.frame_filename}', event)" title="Pin/Star to Favorites & Top 50">
              ${ev.is_starred ? '⭐' : '☆'}
            </button>
            <button class="icon-btn" onclick="hideEvent('${ev.frame_filename}', event)" title="Hide/Dismiss this image">
              🗑️
            </button>
          </div>
        </div>
        ${stereoBannerHtml}
        ${multiObjBannerHtml}
        ${tagBannerHtml}
        <div class="event-desc">${desc}</div>
        <div style="display:flex;gap:4px;flex-wrap:wrap;align-items:center;margin-top:4px;padding-top:6px;border-top:1px dashed #1e293b;">
          <span style="font-size:0.71rem;color:#94a3b8;margin-right:2px;">🧠 Teach:</span>
          <button class="icon-btn" style="padding:2px 6px;font-size:0.72rem;" onclick="reclassifyEvent('${ev.frame_filename}', 'meteor', event)" title="Reclassify as Shooting Star / Meteor">☄️ Meteor</button>
          <button class="icon-btn" style="padding:2px 6px;font-size:0.72rem;" onclick="reclassifyEvent('${ev.frame_filename}', 'fireball', event)" title="Reclassify as Fireball / Bolide">🔥 Fireball</button>
          <button class="icon-btn" style="padding:2px 6px;font-size:0.72rem;" onclick="reclassifyEvent('${ev.frame_filename}', 'space_object', event)" title="Reclassify as Satellite / Space">🛰️ Sat</button>
          <button class="icon-btn" style="padding:2px 6px;font-size:0.72rem;" onclick="reclassifyEvent('${ev.frame_filename}', 'aircraft', event)" title="Reclassify as Aircraft">✈️ Plane</button>
          <button class="icon-btn" style="padding:2px 6px;font-size:0.72rem;" onclick="reclassifyEvent('${ev.frame_filename}', 'transient_flash', event)" title="Reclassify as Flash">✨ Flash</button>
          <button class="icon-btn" style="padding:2px 6px;font-size:0.72rem;color:#fca5a5;" onclick="reclassifyEvent('${ev.frame_filename}', 'false_positive', event)" title="Mark as False Positive (Cloud/Glare/Hotspot)">❌ False</button>
        </div>
        <div class="event-meta">
          <span>🕒 ${ev.frame_time_display || ev.frame_filename}</span>
          <span>SNR ${Number(ev.snr || 0).toFixed(1)}σ • ${Number(ev.length_px || 0).toFixed(0)}px</span>
        </div>
      </div>
    </div>
  `;
}

function renderSpotlight() {
  const spotSec = document.getElementById('spotlightSection');
  const spotGrid = document.getElementById('spotlightGrid');
  if (CURRENT_VIEW !== 'latest' || TOP_50.length === 0) {
    spotSec.style.display = 'none';
    return;
  }
  spotSec.style.display = 'block';
  const top4 = TOP_50.slice(0, 4);
  spotGrid.innerHTML = top4.map((ev, i) => {
    const snapUrl = `/snapshots/${encodeURIComponent(ev.snapshot_filename || '')}`;
    const catBadge = CAT_ICONS[ev.category] || ev.category;
    const medal = i === 0 ? '🥇 #1' : (i === 1 ? '🥈 #2' : (i === 2 ? '🥉 #3' : `#${i + 1}`));
    return `
      <div class="event-card rank-top3" style="cursor:pointer;" onclick="openTop50At(${i})">
        <div class="event-thumb" style="aspect-ratio: 16 / 10;">
          <img src="${snapUrl}" loading="lazy" />
          <span class="badge-cat">${catBadge}</span>
          <span class="badge-rank">${medal} • 🔥 ${Number(ev.interest_score || 0).toFixed(1)}</span>
          <span class="badge-coords">Alt ${Number(ev.alt_deg || 0).toFixed(0)}° ${ev.cardinal || ''}</span>
        </div>
        <div style="padding:10px 12px;">
          <div style="font-weight:700;font-size:0.88rem;color:#f8fafc;">${ev.subject || ev.subtype}</div>
          <div style="font-size:0.75rem;color:var(--muted);margin-top:3px;display:flex;justify-content:space-between;">
            <span>${ev.frame_time_display || ''}</span>
            <span>SNR ${Number(ev.snr || 0).toFixed(1)}σ • ${Number(ev.length_px || 0).toFixed(0)}px</span>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function renderAll() {
  renderSpotlight();

  const groupPasses = document.getElementById('groupPassesToggle').checked;
  const showTests = document.getElementById('showTestsToggle').checked;
  const camFilterEl = document.getElementById('cameraFilterSelect');
  const camFilter = camFilterEl ? camFilterEl.value : 'all';
  const sortMode = document.getElementById('sortSelect').value;
  const query = (document.getElementById('searchBox').value || '').trim().toLowerCase();

  let sourceList = [];
  if (CURRENT_VIEW === 'top50') {
    sourceList = ALL_EVENTS.filter(e => {
      if (!showTests && e.is_test && !e.is_starred) return false;
      if (groupPasses && !e.is_pass_best) return false;
      if (camFilter !== 'all' && (e.camera_id || 'cam1') !== camFilter) return false;
      if (!matchesCategoryFilter(e, CURRENT_FILTER)) return false;
      return true;
    });
    sourceList.sort((a, b) => {
      if ((b.is_starred || 0) !== (a.is_starred || 0)) return (b.is_starred || 0) - (a.is_starred || 0);
      if ((b.interest_score || 0) !== (a.interest_score || 0)) return (b.interest_score || 0) - (a.interest_score || 0);
      return (b.snr || 0) - (a.snr || 0);
    });
    sourceList = sourceList.slice(0, 50);
    sourceList.forEach((e, idx) => { e.display_rank = idx + 1; });
  } else if (CURRENT_VIEW === 'starred') {
    sourceList = ALL_EVENTS.filter(e => e.is_starred && (camFilter === 'all' || (e.camera_id || 'cam1') === camFilter) && matchesCategoryFilter(e, CURRENT_FILTER));
  } else if (CURRENT_VIEW === 'feedback') {
    sourceList = ALL_EVENTS.filter(e => Boolean(e.user_corrected_category) && (camFilter === 'all' || (e.camera_id || 'cam1') === camFilter) && matchesCategoryFilter(e, CURRENT_FILTER));
  } else {
    // 'latest' or 'archive'
    sourceList = ALL_EVENTS.filter(e => {
      if (!showTests && e.is_test && !e.is_starred) return false;
      if (groupPasses && !e.is_pass_best) return false;
      if (camFilter !== 'all' && (e.camera_id || 'cam1') !== camFilter) return false;
      if (CURRENT_VIEW === 'archive' && CURRENT_DATE !== 'all' && e.frame_date !== CURRENT_DATE) return false;
      if (!matchesCategoryFilter(e, CURRENT_FILTER)) return false;
      return true;
    });
  }

  if (query) {
    sourceList = sourceList.filter(e => {
      const hay = `${e.subject || ''} ${e.subtype || ''} ${e.category || ''} ${e.vlm_description || ''} ${e.cardinal || ''} ${e.frame_time_display || ''}`.toLowerCase();
      return hay.includes(query);
    });
  }

  if (sortMode === 'interest') {
    sourceList.sort((a, b) => (b.interest_score || 0) - (a.interest_score || 0));
  } else if (sortMode === 'newest') {
    sourceList.sort((a, b) => (b.detected_at || 0) - (a.detected_at || 0));
  } else if (sortMode === 'snr') {
    sourceList.sort((a, b) => (b.snr || 0) - (a.snr || 0));
  } else if (sortMode === 'length') {
    sourceList.sort((a, b) => (b.length_px || 0) - (a.length_px || 0));
  }

  const titleEl = document.getElementById('sectionBannerTitle');
  const subEl = document.getElementById('sectionBannerSub');
  const grid = document.getElementById('eventsGrid');
  const pagBar = document.getElementById('paginationBar');

  let displaySlice = [];
  if (CURRENT_VIEW === 'latest') {
    displaySlice = sourceList.slice(0, LATEST_MAX_CARDS);
    titleEl.innerHTML = `⚡ Last ${displaySlice.length} Sky Captures`;
    subEl.innerHTML = sourceList.length > LATEST_MAX_CARDS
      ? `Showing newest ${LATEST_MAX_CARDS} of ${sourceList.length} organized sky captures — <a href="javascript:void(0)" onclick="switchView('archive')" style="color:var(--accent);font-weight:700;">Browse Full Archive (${sourceList.length}) →</a>`
      : `Showing all ${displaySlice.length} recent sky captures`;
    pagBar.innerHTML = sourceList.length > LATEST_MAX_CARDS
      ? `<button class="btn" onclick="switchView('archive')">🗂️ Open Full Organized Archive (${sourceList.length} Events)</button>
         <button class="btn amber" onclick="switchView('top50')">🏆 Browse Top 50 Most Interesting</button>`
      : '';
  } else if (CURRENT_VIEW === 'top50') {
    displaySlice = sourceList;
    titleEl.innerHTML = `🏆 Top ${displaySlice.length} Most Interesting Sky Captures`;
    subEl.innerHTML = `Ranked by Astronomical Interest Score (Category Rarity + SNR + Streak Length + Sky Altitude + Starred Pin)`;
    pagBar.innerHTML = '';
  } else {
    const totalItems = sourceList.length;
    const totalPages = Math.max(1, Math.ceil(totalItems / PAGE_SIZE_ARCHIVE));
    if (CURRENT_PAGE > totalPages) CURRENT_PAGE = totalPages;
    const startIdx = (CURRENT_PAGE - 1) * PAGE_SIZE_ARCHIVE;
    displaySlice = sourceList.slice(startIdx, startIdx + PAGE_SIZE_ARCHIVE);

    if (CURRENT_VIEW === 'starred') {
      titleEl.innerHTML = `⭐ Starred Favorite Sky Captures (${totalItems})`;
      subEl.innerHTML = `Click the ⭐ button on any card to pin or unpin favorites`;
    } else if (CURRENT_VIEW === 'feedback') {
      const fbTotal = FEEDBACK_SUMMARY.total_samples || totalItems;
      titleEl.innerHTML = `🧠 Active Learning & Human Corrections (${fbTotal} total learned)`;
      subEl.innerHTML = `Every correction trains the 11-D k-NN classifier, spatial hotspot filter, and few-shot RTX 8000 VLM prompt`;
    } else {
      const dateLbl = CURRENT_DATE === 'all' ? 'All Dates' : CURRENT_DATE;
      titleEl.innerHTML = `🗂️ Organized Sky Archive • ${dateLbl} (${totalItems} matching)`;
      subEl.innerHTML = `Page ${CURRENT_PAGE} of ${totalPages}`;
    }

    if (totalPages > 1) {
      pagBar.innerHTML = `
        <button class="btn secondary" ${CURRENT_PAGE <= 1 ? 'disabled' : ''} onclick="CURRENT_PAGE--;renderAll();">← Previous Page</button>
        <span style="font-size:0.86rem;color:var(--muted);">Page <b>${CURRENT_PAGE}</b> of <b>${totalPages}</b> (${totalItems} events)</span>
        <button class="btn secondary" ${CURRENT_PAGE >= totalPages ? 'disabled' : ''} onclick="CURRENT_PAGE++;renderAll();">Next Page →</button>
      `;
    } else {
      pagBar.innerHTML = '';
    }
  }

  ACTIVE_DISPLAY_LIST = displaySlice;

  if (!displaySlice.length) {
    if (CURRENT_VIEW === 'feedback') {
      grid.innerHTML = `<div class="card" style="grid-column:1/-1;text-align:center;padding:36px;color:#cbd5e1;">
        <div style="font-size:1.1rem;font-weight:700;margin-bottom:8px;">🧠 Ready to Learn from Your Corrections!</div>
        <div style="font-size:0.88rem;color:var(--muted);max-width:740px;margin:0 auto;line-height:1.5;">
          Reply to any Allsky alert in Discord <code>#all-sky</code> with <code>plane</code>, <code>meteor</code>, <code>fireball</code>, <code>satellite</code>, <code>flash</code>, or <code>false</code>
          (or react with ☄️ 🔥 🛰️ ✈️ ✨ ❌), OR click any <b>🧠 Teach</b> button on an event card to train the classifier!
        </div>
      </div>`;
    } else {
      grid.innerHTML = `<div class="card" style="grid-column:1/-1;text-align:center;padding:40px;color:var(--muted);">
        No sky events match this view/filter yet. Try switching the category filter to <b>"All Categories"</b>!
      </div>`;
    }
    return;
  }

  const showRankRibbon = (CURRENT_VIEW === 'top50');
  grid.innerHTML = displaySlice.map((ev, idx) => buildEventCardHtml(ev, idx, showRankRibbon)).join('');
}

function openRawModal(src, title) {
  MODAL_INDEX = -1;
  document.getElementById('modalTitle').textContent = title || 'Allsky Dome Preview';
  document.getElementById('modalSubtitle').textContent = '';
  document.getElementById('modalImg').src = src;
  document.getElementById('modalPrevBtn').style.display = 'none';
  document.getElementById('modalNextBtn').style.display = 'none';
  document.getElementById('modalStarBtn').style.display = 'none';
  document.getElementById('modalHideBtn').style.display = 'none';
  document.getElementById('modalTeachBar').style.display = 'none';
  document.getElementById('modalPassScrubber').innerHTML = '';
  document.getElementById('modal').classList.add('open');
}

function openTop50At(idx) {
  ACTIVE_DISPLAY_LIST = TOP_50;
  openEventModal(idx);
}

function openSlideshowLatest50() {
  switchView('latest');
  if (ACTIVE_DISPLAY_LIST.length > 0) {
    openEventModal(0);
  }
}

function openSlideshowTop50() {
  switchView('top50');
  if (ACTIVE_DISPLAY_LIST.length > 0) {
    openEventModal(0);
  }
}

let MODAL_ZOOM = 1.0;
let MODAL_PAN_X = 0;
let MODAL_PAN_Y = 0;
let MODAL_DRAGGING = false;
let MODAL_DRAG_START_X = 0;
let MODAL_DRAG_START_Y = 0;

function applyModalTransform() {
  const imgEl = document.getElementById('modalImg');
  if (imgEl) {
    imgEl.style.transform = `translate(${MODAL_PAN_X}px, ${MODAL_PAN_Y}px) scale(${MODAL_ZOOM})`;
  }
  const lbl = document.getElementById('modalZoomLabel');
  if (lbl) {
    lbl.textContent = `${Math.round(MODAL_ZOOM * 100)}%`;
  }
}

function resetModalZoom() {
  MODAL_ZOOM = 1.0;
  MODAL_PAN_X = 0;
  MODAL_PAN_Y = 0;
  applyModalTransform();
}

function zoomModalBy(factor, clientX, clientY) {
  const stage = document.getElementById('modalStage');
  if (!stage) return;
  const rect = stage.getBoundingClientRect();
  const mx = (typeof clientX === 'number') ? (clientX - rect.left - rect.width / 2) : 0;
  const my = (typeof clientY === 'number') ? (clientY - rect.top - rect.height / 2) : 0;
  const nextZoom = Math.max(0.10, Math.min(40.0, MODAL_ZOOM * factor));
  MODAL_PAN_X = mx - (mx - MODAL_PAN_X) * (nextZoom / MODAL_ZOOM);
  MODAL_PAN_Y = my - (my - MODAL_PAN_Y) * (nextZoom / MODAL_ZOOM);
  MODAL_ZOOM = nextZoom;
  applyModalTransform();
}

function getActiveModalFrame() {
  if (MODAL_INDEX < 0 || MODAL_INDEX >= ACTIVE_DISPLAY_LIST.length) return null;
  const ev = ACTIVE_DISPLAY_LIST[MODAL_INDEX];
  const passFrames = ev.pass_frames || [];
  return (passFrames.length > 1 && passFrames[MODAL_PASS_FRAME_INDEX]) ? passFrames[MODAL_PASS_FRAME_INDEX] : ev;
}

function getActiveModalBbox() {
  const af = getActiveModalFrame();
  if (!af) return null;
  const ev = ACTIVE_DISPLAY_LIST[MODAL_INDEX] || {};
  let rawFj = af.features_json;
  if (!rawFj && af.frame_filename) {
    const matched = ALL_EVENTS.find(x => x.frame_filename === af.frame_filename);
    if (matched) rawFj = matched.features_json;
  }
  if (!rawFj) rawFj = ev.features_json;
  if (!rawFj) return null;
  try {
    const fj = (typeof rawFj === 'string') ? JSON.parse(rawFj) : rawFj;
    if (fj && Array.isArray(fj.bbox) && fj.bbox.length === 4) {
      return fj.bbox;
    }
  } catch (e) {}
  return null;
}

function focusModalTargetBox() {
  const bbox = getActiveModalBbox();
  const imgEl = document.getElementById('modalImg');
  if (!bbox || !imgEl) {
    zoomModalBy(2.0);
    return;
  }
  const w = imgEl.clientWidth || 720;
  const h = imgEl.clientHeight || 720;
  const cxNorm = ((bbox[0] + bbox[2]) * 0.5) / 3552.0 - 0.5;
  const cyNorm = ((bbox[1] + bbox[3]) * 0.5) / 3552.0 - 0.5;
  const ix = cxNorm * w;
  const iy = cyNorm * h;
  const targetZoom = (MODAL_ZOOM >= 3.2 && MODAL_ZOOM < 18.0) ? Math.min(40.0, MODAL_ZOOM * 1.8) : 4.0;
  MODAL_ZOOM = targetZoom;
  MODAL_PAN_X = -ix * targetZoom;
  MODAL_PAN_Y = -iy * targetZoom;
  applyModalTransform();
}

(function initModalZoomAndPan() {
  const stage = document.getElementById('modalStage');
  if (!stage) return;

  stage.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.18 : 0.85;
    zoomModalBy(factor, e.clientX, e.clientY);
  }, { passive: false });

  stage.addEventListener('mousedown', (e) => {
    if (e.button !== 0 || e.target.closest('button')) return;
    MODAL_DRAGGING = true;
    MODAL_DRAG_START_X = e.clientX - MODAL_PAN_X;
    MODAL_DRAG_START_Y = e.clientY - MODAL_PAN_Y;
    stage.classList.add('dragging');
  });

  window.addEventListener('mousemove', (e) => {
    if (!MODAL_DRAGGING) return;
    MODAL_PAN_X = e.clientX - MODAL_DRAG_START_X;
    MODAL_PAN_Y = e.clientY - MODAL_DRAG_START_Y;
    applyModalTransform();
  });

  window.addEventListener('mouseup', () => {
    if (!MODAL_DRAGGING) return;
    MODAL_DRAGGING = false;
    stage.classList.remove('dragging');
  });

  stage.addEventListener('dblclick', (e) => {
    if (e.target.closest('button')) return;
    zoomModalBy(2.5, e.clientX, e.clientY);
  });
})();

function openEventModal(idx) {
  if (idx < 0 || idx >= ACTIVE_DISPLAY_LIST.length) return;
  MODAL_INDEX = idx;
  MODAL_PASS_FRAME_INDEX = 0;
  const ev = ACTIVE_DISPLAY_LIST[MODAL_INDEX];
  const passFrames = ev.pass_frames || [];
  if (passFrames.length > 1) {
    const foundIdx = passFrames.findIndex(pf => pf.frame_filename === ev.frame_filename);
    MODAL_PASS_FRAME_INDEX = foundIdx >= 0 ? foundIdx : 0;
  }
  updateModalView();
  document.getElementById('modal').classList.add('open');
}

function updateModalView() {
  if (typeof stopPortalModalBlink === 'function') stopPortalModalBlink();
  PORTAL_BLINK_ITEM = null;
  const bb = document.getElementById('modalBlinkBar');
  const bp = document.getElementById('modalBlinkPhasePill');
  if (bb) bb.style.display = 'none';
  if (bp) bp.style.display = 'none';
  document.getElementById('modalImg').onerror = null;
  if (MODAL_INDEX < 0 || MODAL_INDEX >= ACTIVE_DISPLAY_LIST.length) return;
  const ev = ACTIVE_DISPLAY_LIST[MODAL_INDEX];
  const passFrames = ev.pass_frames || [];
  const activeFrame = (passFrames.length > 1 && passFrames[MODAL_PASS_FRAME_INDEX]) ? passFrames[MODAL_PASS_FRAME_INDEX] : ev;

  resetModalZoom();
  const snapUrl = `/snapshots/${encodeURIComponent(activeFrame.snapshot_filename || ev.snapshot_filename || '')}`;
  document.getElementById('modalImg').src = snapUrl;

  const focusBtn = document.getElementById('modalFocusBoxBtn');
  if (focusBtn) {
    focusBtn.style.display = getActiveModalBbox() ? 'inline-flex' : 'none';
  }

  const rankStr = ev.top_rank ? `🏆 Top #${ev.top_rank} Most Interesting • ` : '';
  document.getElementById('modalTitle').textContent = `${rankStr}${activeFrame.subject || ev.subject || ev.subtype} (🔥 Score ${Number(activeFrame.interest_score || ev.interest_score || 0).toFixed(1)})`;
  document.getElementById('modalSubtitle').textContent =
    `${activeFrame.frame_time_display || ev.frame_time_display} • Alt ${Number(activeFrame.alt_deg || 0).toFixed(1)}° Az ${Number(activeFrame.az_deg || 0).toFixed(0)}° (${activeFrame.cardinal || ''}) • SNR ${Number(activeFrame.snr || 0).toFixed(1)}σ • ${Number(activeFrame.length_px || 0).toFixed(0)}px [${MODAL_INDEX + 1} of ${ACTIVE_DISPLAY_LIST.length}]`;

  document.getElementById('modalPrevBtn').style.display = ACTIVE_DISPLAY_LIST.length > 1 ? 'flex' : 'none';
  document.getElementById('modalNextBtn').style.display = ACTIVE_DISPLAY_LIST.length > 1 ? 'flex' : 'none';

  const starBtn = document.getElementById('modalStarBtn');
  starBtn.style.display = 'inline-flex';
  starBtn.textContent = ev.is_starred ? '⭐ Starred' : '☆ Star Favorite';

  document.getElementById('modalHideBtn').style.display = 'inline-flex';
  document.getElementById('modalTeachBar').style.display = 'inline-flex';

  const scrub = document.getElementById('modalPassScrubber');
  const curFilename = activeFrame.frame_filename || ev.frame_filename;
  const sameFrameSiblings = getSameFrameSiblings(curFilename);
  let scrubHtml = '';
  if (sameFrameSiblings.length > 1) {
    scrubHtml += `<span style="font-size:0.8rem;color:#e9d5ff;font-weight:700;">🎯 Objects in This Frame (${sameFrameSiblings.length}):</span>` +
      sameFrameSiblings.map((sib, i) => `
        <button class="pass-step-btn ${sib.frame_filename === curFilename ? 'active' : ''}" onclick="openEventByFilename('${sib.frame_filename}', event)">
          Object ${i + 1}/${sameFrameSiblings.length}: ${String(sib.subject || sib.subtype || sib.category).slice(0, 28)} (${sib.category})
        </button>
      `).join('');
  }
  if (passFrames.length > 1) {
    scrubHtml += `<span style="font-size:0.8rem;color:#c4b5fd;font-weight:700;margin-left:${sameFrameSiblings.length > 1 ? '10px' : '0'};">🎞️ Multi-Frame Transit Pass (${passFrames.length} frames):</span>` +
      passFrames.map((pf, i) => `
        <button class="pass-step-btn ${i === MODAL_PASS_FRAME_INDEX ? 'active' : ''}" onclick="selectModalPassFrame(${i})">
          Frame ${i + 1}/${passFrames.length} (${(pf.frame_time_display || '').slice(-11)} • ${Number(pf.snr || 0).toFixed(0)}σ)
        </button>
      `).join('');
  }
  if (scrubHtml) {
    scrub.style.display = 'flex';
    scrub.innerHTML = scrubHtml;
  } else {
    scrub.style.display = 'none';
    scrub.innerHTML = '';
  }
}

function selectModalPassFrame(i) {
  MODAL_PASS_FRAME_INDEX = i;
  updateModalView();
}

function stepModal(delta) {
  if (!ACTIVE_DISPLAY_LIST.length) return;
  MODAL_INDEX = (MODAL_INDEX + delta + ACTIVE_DISPLAY_LIST.length) % ACTIVE_DISPLAY_LIST.length;
  MODAL_PASS_FRAME_INDEX = 0;
  updateModalView();
}

function closeModal() {
  if (typeof stopPortalModalBlink === 'function') stopPortalModalBlink();
  document.getElementById('modal').classList.remove('open');
}

document.addEventListener('keydown', (e) => {
  if (!document.getElementById('modal').classList.contains('open')) return;
  if (e.key === 'Escape') closeModal();
  else if (e.key === 'ArrowLeft') stepModal(-1);
  else if (e.key === 'ArrowRight') stepModal(1);
  else if (e.key === '+' || e.key === '=') zoomModalBy(1.35);
  else if (e.key === '-' || e.key === '_') zoomModalBy(0.74);
  else if (e.key === '0') resetModalZoom();
  else if (e.key === 'f' || e.key === 'F') focusModalTargetBox();
  else if (e.code === 'Space' && PORTAL_BLINK_ITEM) {
    e.preventDefault();
    setPortalModalBlinkMode(PORTAL_BLINK_MODE === 'ref' ? 'sci' : 'ref');
  }
});

function showTeachToast(msg, isOk) {
  let t = document.getElementById('teachToastBanner');
  if (!t) {
    t = document.createElement('div');
    t.id = 'teachToastBanner';
    t.style.cssText = 'position:fixed;bottom:22px;right:22px;z-index:99999;padding:11px 16px;border-radius:10px;font-size:0.84rem;font-weight:700;box-shadow:0 12px 32px rgba(0,0,0,0.75);transition:opacity 0.25s ease;max-width:520px;';
    document.body.appendChild(t);
  }
  t.style.background = isOk ? 'rgba(6,78,59,0.96)' : 'rgba(127,29,29,0.96)';
  t.style.border = isOk ? '1px solid #34d399' : '1px solid #f87171';
  t.style.color = isOk ? '#a7f3d0' : '#fecaca';
  t.textContent = msg;
  t.style.opacity = '1';
  t.style.display = 'block';
  clearTimeout(window._teachToastTimer);
  window._teachToastTimer = setTimeout(() => { t.style.opacity = '0'; setTimeout(() => { t.style.display = 'none'; }, 260); }, 4200);
}

async function reclassifyEvent(frameFilename, newCategory, domEvent) {
  if (domEvent) domEvent.stopPropagation();
  const statusEl = document.getElementById('saveStatus');
  statusEl.style.color = '#38bdf8';
  statusEl.textContent = `🧠 Teaching AI: Reclassifying ${frameFilename} → ${newCategory}...`;
  showTeachToast(`🧠 Teaching AI: Reclassifying ${frameFilename} → ${newCategory}...`, true);
  const res = await fetch('/api/reclassify', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frame_filename: frameFilename, new_category: newCategory, note: `Web Portal 1-click correction to ${newCategory}`})
  });
  const data = await res.json();
  statusEl.style.color = res.ok ? '#34d399' : '#f87171';
  statusEl.textContent = data.detail || 'Learned correction!';
  showTeachToast(data.detail || 'Learned correction!', res.ok);
  await loadData(true);
  if (MODAL_INDEX >= 0 && document.getElementById('modal').classList.contains('open')) {
    if (newCategory === 'false_positive') {
      if (ACTIVE_DISPLAY_LIST.length === 0) closeModal();
      else {
        if (MODAL_INDEX >= ACTIVE_DISPLAY_LIST.length) MODAL_INDEX = ACTIVE_DISPLAY_LIST.length - 1;
        updateModalView();
      }
    } else {
      updateModalView();
    }
  }
}

async function reclassifyModal(newCategory) {
  if (MODAL_INDEX < 0 || MODAL_INDEX >= ACTIVE_DISPLAY_LIST.length) return;
  const ev = ACTIVE_DISPLAY_LIST[MODAL_INDEX];
  const passFrames = ev.pass_frames || [];
  const activeFrame = (passFrames.length > 1 && passFrames[MODAL_PASS_FRAME_INDEX]) ? passFrames[MODAL_PASS_FRAME_INDEX] : ev;
  await reclassifyEvent(activeFrame.frame_filename || ev.frame_filename, newCategory, null);
}

async function toggleStar(frameFilename, domEvent) {
  if (domEvent) domEvent.stopPropagation();
  await fetch('/api/event-action', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action: 'star', frame_filename: frameFilename})
  });
  await loadData(true);
  if (MODAL_INDEX >= 0 && document.getElementById('modal').classList.contains('open')) {
    updateModalView();
  }
}

async function toggleModalStar() {
  if (MODAL_INDEX < 0 || MODAL_INDEX >= ACTIVE_DISPLAY_LIST.length) return;
  await toggleStar(ACTIVE_DISPLAY_LIST[MODAL_INDEX].frame_filename, null);
}

async function hideEvent(frameFilename, domEvent) {
  if (domEvent) domEvent.stopPropagation();
  await fetch('/api/event-action', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action: 'hide', frame_filename: frameFilename})
  });
  await loadData(true);
}

async function hideModalEvent() {
  if (MODAL_INDEX < 0 || MODAL_INDEX >= ACTIVE_DISPLAY_LIST.length) return;
  const fname = ACTIVE_DISPLAY_LIST[MODAL_INDEX].frame_filename;
  await hideEvent(fname, null);
  if (ACTIVE_DISPLAY_LIST.length === 0) closeModal();
  else {
    if (MODAL_INDEX >= ACTIVE_DISPLAY_LIST.length) MODAL_INDEX = ACTIVE_DISPLAY_LIST.length - 1;
    updateModalView();
  }
}

function toggleSecondaryDrawer() {
  const drawer = document.getElementById('secondaryControlsDrawer');
  if (!drawer) return;
  const opening = drawer.style.display === 'none';
  drawer.style.display = opening ? 'grid' : 'none';
  if (opening) refreshLiveImage(true);
}

function refreshLiveImage(force) {
  const drawer = document.getElementById('secondaryControlsDrawer');
  if (!force && (!drawer || drawer.style.display === 'none')) return;
  const showMask = document.getElementById('showMaskToggle').checked ? 1 : 0;
  const showAdsbEl = document.getElementById('showAdsbToggle');
  const showAdsb = (!showAdsbEl || showAdsbEl.checked) ? 1 : 0;
  const showSatsEl = document.getElementById('showSatsToggle');
  const showSats = (!showSatsEl || showSatsEl.checked) ? 1 : 0;
  const camSel = document.getElementById('liveCameraSelect');
  const camId = (camSel && camSel.value) ? camSel.value : 'cam1';
  document.getElementById('liveSkyImg').src = `/api/live-image?camera_id=${encodeURIComponent(camId)}&mask=${showMask}&adsb=${showAdsb}&sats=${showSats}&t=${Date.now()}`;
}

let LAST_STATS_ETAG = '';

async function loadData(forceFull) {
  try {
    const headers = (!forceFull && LAST_STATS_ETAG) ? {'If-None-Match': LAST_STATS_ETAG} : {};
    const res = await fetch('/api/stats', {headers});
    if (res.status === 304) return;
    LAST_STATS_ETAG = res.headers.get('ETag') || '';
    const data = await res.json();
    ALL_EVENTS = data.events || [];
    const passMap = {};
    ALL_EVENTS.forEach(e => { if (e.pass_frames && e.pass_frames.length > 1) passMap[e.pass_id] = e.pass_frames; });
    ALL_EVENTS.forEach(e => { if ((!e.pass_frames || !e.pass_frames.length) && passMap[e.pass_id]) e.pass_frames = passMap[e.pass_id]; });
    TOP_50 = data.top_50 || [];
    DATE_FOLDERS = data.date_folders || [];
    FEEDBACK_SUMMARY = data.feedback || {};
    CONFIGURED_CAMERAS = data.cameras || [];

    const camFilterSel = document.getElementById('cameraFilterSelect');
    if (camFilterSel && CONFIGURED_CAMERAS.length > 0) {
      const prevVal = camFilterSel.value || 'all';
      camFilterSel.innerHTML = `<option value="all">🎥 All Cameras (${CONFIGURED_CAMERAS.length})</option>` +
        CONFIGURED_CAMERAS.map(c => `<option value="${c.id}">🎥 ${c.name || c.id}${c.enabled === false ? ' (Off)' : ''}</option>`).join('');
      if ([...camFilterSel.options].some(o => o.value === prevVal)) camFilterSel.value = prevVal;
    }

    const liveCamSel = document.getElementById('liveCameraSelect');
    if (liveCamSel && CONFIGURED_CAMERAS.length > 0) {
      const prevCam = liveCamSel.value || 'cam1';
      liveCamSel.innerHTML = CONFIGURED_CAMERAS.map(c => `<option value="${c.id}">${c.name || c.id} (${(c.base_url || '').replace(/^https?:\/\//, '')})</option>`).join('');
      if ([...liveCamSel.options].some(o => o.value === prevCam)) liveCamSel.value = prevCam;
    }

    const adsb = data.adsb || {};
    const st = data.stats || {};
    const live = st.live || {};
    const byCat = st.by_category || {};

    document.getElementById('kpiTop50').textContent = TOP_50.length;
    document.getElementById('kpiMeteors').textContent = (byCat.meteor || 0) + (byCat.fireball || 0);
    document.getElementById('kpiSpace').textContent = (byCat.space_object || 0) + (byCat.transient_flash || 0) + (byCat.sky_phenomenon || 0);
    document.getElementById('kpiAircraft').textContent = (byCat.aircraft || 0) + (byCat.day_aircraft || 0);
    document.getElementById('kpiPasses').textContent = st.unique_passes || 0;
    document.getElementById('kpiFrames').textContent = st.processed_frames || 0;

    document.getElementById('cntLatest').textContent = Math.min(LATEST_MAX_CARDS, st.unique_passes || ALL_EVENTS.length);
    document.getElementById('cntTop50').textContent = TOP_50.length;
    document.getElementById('cntArchive').textContent = st.total_events || 0;
    document.getElementById('cntStarred').textContent = st.starred_count || 0;
    document.getElementById('cntFeedback').textContent = FEEDBACK_SUMMARY.total_samples || 0;

    const botSt = FEEDBACK_SUMMARY.bot_status || {};
    const botPill = document.getElementById('botStatusPill');
    const invLink = document.getElementById('botInviteLink');
    if (botSt.state === 'connected' || botSt.state === 'polling_rest_only') {
      botPill.textContent = `🤖 Bot Online: @${botSt.bot_username || 'AllskyBot'} (#all-sky)`;
      botPill.className = 'pill ok';
    } else if (botSt.state === 'error' || botSt.state === 'connecting') {
      botPill.textContent = `🤖 Bot: ${botSt.detail || botSt.state}`;
      botPill.className = 'pill warn';
    } else {
      botPill.textContent = `🤖 Bot: Paste Token to Listen in #all-sky (${FEEDBACK_SUMMARY.total_samples || 0} learned)`;
      botPill.className = 'pill warn';
    }
    if (botSt.invite_url) {
      invLink.href = botSt.invite_url;
      invLink.textContent = '➕ Invite Bot to Discord Server →';
    }

    const mode = live.mode || 'NIGHT';
    const mp = document.getElementById('modePill');
    mp.textContent = mode === 'NIGHT' ? '🌙 Mode: NIGHT' : '☀️ Mode: DAY';
    mp.className = 'pill ' + (mode === 'NIGHT' ? 'night' : 'day');

    document.getElementById('expPill').textContent = `⏱️ Exp: ${live.exposure || '--'} (Gain ${live.gain || '0'})`;
    document.getElementById('moonPill').textContent = `🌕 Moon: ${live.moon_elevation || '--'} (${live.moon_illumination || '--'}%)`;
    document.getElementById('lastFrameText').textContent = `Latest Frame: ${live.last_frame || '--'} (Sky Mean: ${live.mean_sky_brightness ?? '--'})`;

    const adsbPill = document.getElementById('adsbPill');
    if (adsbPill) {
      const domeCnt = adsb.aircraft_in_dome_count ?? 0;
      const rangeCnt = adsb.aircraft_in_range_count ?? 0;
      const domeFlights = (adsb.aircraft_in_dome || []).slice(0, 2).map(a => a.callsign).join(', ');
      if (adsb.local_receiver_ok) {
        adsbPill.textContent = domeCnt > 0
          ? `✈️ ADS-B: ${domeCnt} Overhead (${domeFlights}) • ${rangeCnt} in Range`
          : `✈️ ADS-B: ${rangeCnt} in Range (0 Overhead)`;
        adsbPill.className = 'pill ok';
      } else {
        adsbPill.textContent = `✈️ ADS-B: ${rangeCnt} Tracked`;
        adsbPill.className = 'pill warn';
      }
    }

    const dp = document.getElementById('discordPill');
    if (st.allsky_webhook_configured) {
      dp.textContent = '🔔 Discord #all-sky: Dedicated Webhook Active';
      dp.className = 'pill ok';
    } else if (st.default_webhook_configured) {
      dp.textContent = '🔔 Discord #all-sky: Using Default Webhook';
      dp.className = 'pill warn';
    }

    if (!INITIAL_LOADED && data.config) {
      INITIAL_LOADED = true;
      document.getElementById('cfgNightSigma').value = data.config.night_k_sigma ?? 4.8;
      document.getElementById('cfgMeteorLen').value = data.config.meteor_min_length_px ?? 36;
      document.getElementById('cfgAlertAircraft').value = String(data.config.alert_on_aircraft ?? true);
      document.getElementById('cfgDayEnabled').value = String(data.config.day_detection_enabled ?? true);
      if (st.allsky_webhook_url) {
        document.getElementById('allskyWebhookUrl').value = st.allsky_webhook_url;
      }
      if (botSt.client_id) {
        document.getElementById('discordClientId').value = botSt.client_id;
      }
    }

    if (data.realignment) {
      updateRealignmentUI(data.realignment, data.config || {});
    }

    renderDateFolders();
    renderAll();
  } catch (e) {
    console.error('Failed loading stats:', e);
  }
}

function updateRealignmentUI(realign, cfg) {
  const banner = document.getElementById('realignmentBanner');
  const titleEl = document.getElementById('realignBannerTitle');
  const msgEl = document.getElementById('realignBannerMsg');
  const subEl = document.getElementById('realignBannerSub');
  const topBtn = document.getElementById('btnArmRealignTop');
  const cancelBtn = document.getElementById('btnCancelRealign');
  if (!banner) return;

  const curAz = (cfg.bottom_azimuth_deg ?? 265.055).toFixed(2);
  const curTx = (cfg.tilt_x_deg ?? 6.292).toFixed(2);
  const curTy = (cfg.tilt_y_deg ?? -0.947).toFixed(2);

  if (realign.armed) {
    banner.style.display = 'block';
    banner.style.borderColor = '#f59e0b';
    banner.style.background = 'linear-gradient(135deg,rgba(120,53,15,0.48),rgba(15,23,42,0.96))';
    const cands = (realign.candidate_solutions || []).length;
    titleEl.innerHTML = `🧭 Camera Realignment Armed — Watching Sky (${cands}/2 Clear Frame Passes Verified)`;
    titleEl.style.color = '#fde68a';
    msgEl.textContent = realign.last_check_msg || 'Waiting for a clear starry night sky...';
    subEl.textContent = `Armed at: ${realign.armed_at_iso || '--'} • Frames Checked: ${realign.frames_evaluated || 0} • Daytime Mask Refreshed: ${realign.day_mask_updated ? 'Yes' : 'Waiting for Daylight'} • Current Locked Azimuth: ${curAz}°, Tilt: (${curTx}°, ${curTy}°)`;
    if (cancelBtn) cancelBtn.style.display = 'inline-flex';
    if (topBtn) {
      topBtn.textContent = '⏳ Realigning (Waiting for Clear Sky)...';
      topBtn.className = 'btn amber';
    }
  } else if (realign.status === 'completed' && realign.last_solved_result) {
    banner.style.display = 'block';
    banner.style.borderColor = '#10b981';
    banner.style.background = 'linear-gradient(135deg,rgba(6,78,59,0.48),rgba(15,23,42,0.96))';
    const r = realign.last_solved_result;
    titleEl.innerHTML = `✅ New Camera Alignment Solved & Locked! (${r.matched_stars} Stars Matched • RMS ${r.rms_deg}°)`;
    titleEl.style.color = '#6ee7b7';
    msgEl.textContent = realign.last_check_msg || 'Camera alignment locked.';
    subEl.textContent = `Matched Stars: ${(r.matched_names || []).slice(0, 10).join(', ')} • Completed: ${r.completed_at_iso || '--'}`;
    if (cancelBtn) {
      cancelBtn.textContent = '✕ Dismiss Banner';
      cancelBtn.style.display = 'inline-flex';
    }
    if (topBtn) {
      topBtn.textContent = '🧭 Camera Moved — Realign';
      topBtn.className = 'btn secondary';
    }
  } else {
    banner.style.display = 'none';
    if (topBtn) {
      topBtn.textContent = '🧭 Camera Moved — Realign';
      topBtn.className = 'btn secondary';
    }
  }
}

async function armCameraRealignment() {
  const ok = confirm(
    "Did you physically move or rotate the Allsky camera?\n\n" +
    "Click OK to arm Camera Realignment. The system will NOT guess on cloudy nights — " +
    "it will watch incoming frames until it gets a clear starry sky (>= 55 stars & >= 10 catalog matches across 2 frames) " +
    "to lock the new 3D alignment while preserving your historical nights' alignment."
  );
  if (!ok) return;
  try {
    const res = await fetch('/api/realign/arm', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({update_day_mask: true})
    });
    const data = await res.json();
    if (data.realignment) updateRealignmentUI(data.realignment, {});
    loadData();
  } catch (e) {
    console.error('Failed arming realignment:', e);
  }
}

async function cancelCameraRealignment() {
  try {
    const res = await fetch('/api/realign/cancel', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({})
    });
    const data = await res.json();
    if (data.realignment) updateRealignmentUI(data.realignment, {});
    loadData();
  } catch (e) {
    console.error('Failed cancelling realignment:', e);
  }
}

async function saveSettings() {
  const statusEl = document.getElementById('saveStatus');
  statusEl.style.color = '#94a3b8';
  statusEl.textContent = 'Saving configuration...';
  const payload = {
    night_k_sigma: parseFloat(document.getElementById('cfgNightSigma').value),
    meteor_min_length_px: parseInt(document.getElementById('cfgMeteorLen').value, 10),
    alert_on_aircraft: document.getElementById('cfgAlertAircraft').value === 'true',
    day_detection_enabled: document.getElementById('cfgDayEnabled').value === 'true',
    allsky_webhook_url: document.getElementById('allskyWebhookUrl').value.trim(),
    discord_bot_token: document.getElementById('discordBotToken').value.trim(),
    discord_client_id: document.getElementById('discordClientId').value.trim()
  };
  const res = await fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  statusEl.style.color = res.ok ? '#34d399' : '#f87171';
  statusEl.textContent = data.detail || 'Saved!';
  loadData();
}

async function runSimTest(type) {
  const statusEl = document.getElementById('saveStatus');
  statusEl.style.color = '#38bdf8';
  statusEl.textContent = `Running realistic ${type} detection + RTX 8000 VLM + Discord alert...`;
  const res = await fetch('/api/simulate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({type: type, send_discord: true})
  });
  const data = await res.json();
  statusEl.style.color = res.ok ? '#34d399' : '#f87171';
  statusEl.textContent = data.detail || 'Simulation complete!';
  document.getElementById('showTestsToggle').checked = true;
  loadData();
}

let HERO_SURVEY_DATA = null;
let HERO_DISC_FILTER = 'all';
let HERO_BLINK_ACTIVE = false;
let HERO_BLINK_PHASE = false;
let HERO_BLINK_TIMER = null;
let HERO_CARDS_LIST = [];
let PORTAL_BLINK_ITEM = null;
let PORTAL_BLINK_TIMER = null;
let PORTAL_BLINK_PHASE = true;
let PORTAL_BLINK_MODE = 'auto';

function stopPortalModalBlink() {
  if (PORTAL_BLINK_TIMER) {
    clearInterval(PORTAL_BLINK_TIMER);
    PORTAL_BLINK_TIMER = null;
  }
}

function updatePortalModalBlinkFrame() {
  if (!PORTAL_BLINK_ITEM) return;
  const img = document.getElementById('modalImg');
  const pill = document.getElementById('modalBlinkPhasePill');
  const bust = PORTAL_BLINK_ITEM.cacheBust || '';
  if (PORTAL_BLINK_MODE === 'triplet') {
    img.onerror = null;
    img.src = PORTAL_BLINK_ITEM.tripletUrl + bust;
    pill.style.display = 'inline-block';
    pill.style.background = 'rgba(56,189,248,0.18)';
    pill.style.color = '#7dd3fc';
    pill.style.border = '1px solid rgba(56,189,248,0.45)';
    pill.textContent = '🖼️ 3-PANEL TRIPLET (REF + DISCOVERY + ΔI)';
  } else {
    const showSci = (PORTAL_BLINK_MODE === 'sci') || (PORTAL_BLINK_MODE === 'auto' && PORTAL_BLINK_PHASE);
    img.onerror = () => {
      stopPortalModalBlink();
      PORTAL_BLINK_MODE = 'triplet';
      img.onerror = null;
      img.src = PORTAL_BLINK_ITEM.tripletUrl + bust;
    };
    img.src = (showSci ? PORTAL_BLINK_ITEM.sciUrl : PORTAL_BLINK_ITEM.refUrl) + bust;
    pill.style.display = 'inline-block';
    if (showSci) {
      pill.style.background = 'rgba(16,185,129,0.22)';
      pill.style.color = '#6ee7b7';
      pill.style.border = '1px solid rgba(16,185,129,0.55)';
      pill.textContent = '🟢 2. DISCOVERY NIGHT CO-ADD (SPACE OBJECT PRESENT)';
    } else {
      pill.style.background = 'rgba(245,158,11,0.20)';
      pill.style.color = '#fde68a';
      pill.style.border = '1px solid rgba(245,158,11,0.5)';
      pill.textContent = '🟠 1. MASTER J2000 REFERENCE ATLAS (STATIC STAR FIELD)';
    }
  }
  ['auto', 'ref', 'sci', 'triplet'].forEach(m => {
    const btn = document.getElementById('pBtnBlink' + m.charAt(0).toUpperCase() + m.slice(1));
    if (btn) btn.className = (PORTAL_BLINK_MODE === m) ? 'btn' : 'btn secondary';
  });
}

function setPortalModalBlinkMode(mode) {
  stopPortalModalBlink();
  PORTAL_BLINK_MODE = mode;
  if (mode === 'auto') {
    PORTAL_BLINK_PHASE = true;
    updatePortalModalBlinkFrame();
    PORTAL_BLINK_TIMER = setInterval(() => {
      PORTAL_BLINK_PHASE = !PORTAL_BLINK_PHASE;
      updatePortalModalBlinkFrame();
    }, 650);
  } else {
    updatePortalModalBlinkFrame();
  }
}

function openHeroBlinkModal(idx) {
  const c = HERO_CARDS_LIST[idx];
  if (!c) return;
  stopPortalModalBlink();
  MODAL_INDEX = -1;
  if (c.refUrl) { const p1 = new Image(); p1.src = c.refUrl; }
  if (c.sciUrl) { const p2 = new Image(); p2.src = c.sciUrl; }
  PORTAL_BLINK_ITEM = Object.assign({}, c, { cacheBust: '' });
  document.getElementById('modalTitle').textContent = c.title;
  document.getElementById('modalSubtitle').textContent = `${c.coords} • ${c.metrics}`;
  document.getElementById('modalBlinkBar').style.display = 'inline-flex';
  document.getElementById('modalTeachBar').style.display = 'none';
  document.getElementById('modalStarBtn').style.display = 'none';
  document.getElementById('modalHideBtn').style.display = 'none';
  document.getElementById('modalFocusBoxBtn').style.display = 'none';
  document.getElementById('modalPrevBtn').style.display = 'none';
  document.getElementById('modalNextBtn').style.display = 'none';
  document.getElementById('modalPassScrubber').style.display = 'none';
  resetModalZoom();
  document.getElementById('modal').classList.add('open');
  setPortalModalBlinkMode('auto');
}

function openCustomImageModal(title, subtitle, imgUrl) {
  stopPortalModalBlink();
  PORTAL_BLINK_ITEM = null;
  MODAL_INDEX = -1;
  document.getElementById('modalBlinkBar').style.display = 'none';
  document.getElementById('modalBlinkPhasePill').style.display = 'none';
  document.getElementById('modalTitle').textContent = title;
  document.getElementById('modalSubtitle').textContent = subtitle;
  document.getElementById('modalImg').onerror = null;
  document.getElementById('modalImg').src = imgUrl;
  document.getElementById('modalTeachBar').style.display = 'none';
  document.getElementById('modalStarBtn').style.display = 'none';
  document.getElementById('modalHideBtn').style.display = 'none';
  document.getElementById('modalFocusBoxBtn').style.display = 'none';
  document.getElementById('modalPrevBtn').style.display = 'none';
  document.getElementById('modalNextBtn').style.display = 'none';
  document.getElementById('modalPassScrubber').style.display = 'none';
  resetModalZoom();
  document.getElementById('modal').classList.add('open');
}

function setHeroDiscFilter(f, btn) {
  HERO_DISC_FILTER = f;
  document.querySelectorAll('#heroDiscFilterBar .filter-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  renderHeroDiscoveries();
}

function toggleHeroBlinkMode() {
  HERO_BLINK_ACTIVE = !HERO_BLINK_ACTIVE;
  const btn = document.getElementById('btnLiveBlinkToggle');
  const txt = document.getElementById('heroBlinkStatusText');
  if (HERO_BLINK_ACTIVE) {
    btn.textContent = '⏸️ Stop Live Blink Comparator (Show 3-Panel Triplets)';
    btn.className = 'btn emerald';
    txt.textContent = '⚡ LIVE BLINK COMPARATOR ACTIVE: Alternating every 650ms between MASTER J2000 REFERENCE ATLAS ⇄ DISCOVERY NIGHT STACK!';
    document.querySelectorAll('.hero-blink-img').forEach(img => {
      const refUrl = img.getAttribute('data-ref');
      const sciUrl = img.getAttribute('data-sci');
      if (refUrl) { const p1 = new Image(); p1.src = refUrl; }
      if (sciUrl) { const p2 = new Image(); p2.src = sciUrl; }
    });
    if (HERO_BLINK_TIMER) clearInterval(HERO_BLINK_TIMER);
    HERO_BLINK_TIMER = setInterval(() => {
      HERO_BLINK_PHASE = !HERO_BLINK_PHASE;
      document.querySelectorAll('.hero-blink-img').forEach(img => {
        const refUrl = img.getAttribute('data-ref');
        const sciUrl = img.getAttribute('data-sci');
        if (refUrl && sciUrl) {
          img.src = HERO_BLINK_PHASE ? sciUrl : refUrl;
        }
      });
    }, 650);
  } else {
    btn.textContent = '⚡ Enable Live Blink Comparator (Ref ⇄ Night)';
    btn.className = 'btn amber';
    txt.textContent = '👁️ Showing 3-Panel Triplet Cutouts (Click "⚡ Enable Live Blink Comparator" above to animate Reference ⇄ Discovery Night!)';
    if (HERO_BLINK_TIMER) {
      clearInterval(HERO_BLINK_TIMER);
      HERO_BLINK_TIMER = null;
    }
    renderHeroDiscoveries();
  }
}

function renderHeroDiscoveries() {
  if (!HERO_SURVEY_DATA) return;
  const ver = encodeURIComponent(HERO_SURVEY_DATA.generated_at || '1');
  const sol = HERO_SURVEY_DATA.solar_system_recoveries || [];
  const trk = HERO_SURVEY_DATA.candidate_moving_tracklets || [];
  const ots = HERO_SURVEY_DATA.multi_frame_optical_transients || [];

  const cards = [];
  if (HERO_DISC_FILTER === 'all' || HERO_DISC_FILTER === 'solsys') {
    sol.forEach(s => {
      const base = (s.cutout_url || '').replace(/\.jpg$/, '');
      cards.push({
        kind: 'solsys',
        badge: `🪐 RECOVERED ${s.body_type.toUpperCase()}`,
        badgeColor: '#10b981',
        title: `${s.name} (Mag ${s.mag >= 0 ? '+' : ''}${s.mag}) — ${s.nights_detected}-Night Sky Track`,
        coords: `J2000 ${s.latest_ra_hms} ${s.latest_dec_dms}`,
        metrics: `Moved ${s.total_arc_deg}° (${s.Angular_speed_deg_day}°/day = ${s.angular_speed_arcmin_day}'/day) • Peak SNR ${s.peak_snr}σ`,
        detail: `Tracked across ${s.nights_detected} clear nights (${s.first_night} → ${s.last_night}). Mean astrometric residual ±${s.mean_astrometric_error_deg}°.`,
        tripletUrl: `${s.cutout_url}?v=${ver}`,
        refUrl: `${base}_ref.jpg?v=${ver}`,
        sciUrl: `${base}_sci.jpg?v=${ver}`
      });
    });
  }
  if (HERO_DISC_FILTER === 'all' || HERO_DISC_FILTER === 'tracklet') {
    trk.forEach(t => {
      const base = (t.cutout_url || '').replace(/\.jpg$/, '');
      cards.push({
        kind: 'tracklet',
        badge: `🛸 3-NIGHT MOVING SPACE TRACKLET`,
        badgeColor: '#38bdf8',
        title: `${t.tracklet_id} — ${t.classification_hint}`,
        coords: `J2000 ${t.latest_ra_hms} ${t.latest_dec_dms}`,
        metrics: `Velocity ${t.speed_deg_per_day}°/day (Arc ${t.total_arc_deg}°) • Collinearity ±${t.collinearity_err_deg}° • Mean SNR ${t.mean_snr}σ`,
        detail: `Collinear multi-night detections across nights ${(t.dates || []).join(' → ')} with zero static star in Master Reference Atlas.`,
        tripletUrl: `${t.cutout_url}?v=${ver}`,
        refUrl: `${base}_ref.jpg?v=${ver}`,
        sciUrl: `${base}_sci.jpg?v=${ver}`
      });
    });
  }
  if (HERO_DISC_FILTER === 'all' || HERO_DISC_FILTER === 'transient') {
    ots.forEach(o => {
      const base = (o.cutout_url || '').replace(/\.jpg$/, '');
      cards.push({
        kind: 'transient',
        badge: `✨ MULTI-FRAME OPTICAL TRANSIENT`,
        badgeColor: '#f43f5e',
        title: `${o.transient_id} — Verified on ${o.frame_hits} Sidereal Exposures`,
        coords: `J2000 ${o.ra_hms} ${o.dec_dms}`,
        metrics: `Night ${o.date_str} • Difference SNR ${o.snr}σ (Flux +${o.diff_flux} vs Ref ${o.ref_flux})`,
        detail: `Stationary celestial point source detected across ${o.frame_hits} independent frames during Night ${o.date_str}, absent across other reference nights.`,
        tripletUrl: `${o.cutout_url}?v=${ver}`,
        refUrl: `${base}_ref.jpg?v=${ver}`,
        sciUrl: `${base}_sci.jpg?v=${ver}`
      });
    });
  }

  HERO_CARDS_LIST = cards;
  const grid = document.getElementById('heroDiscoveriesGrid');
  grid.innerHTML = cards.map((c, idx) => {
    const imgSrc = HERO_BLINK_ACTIVE ? c.sciUrl : c.tripletUrl;
    return `
      <div class="event-card" style="border-color:rgba(56,189,248,0.35);background:#0b1325;" onclick="openHeroBlinkModal(${idx})">
        <div style="background:#050914;border-bottom:1px solid #1e293b;display:flex;justify-content:center;align-items:center;height:185px;overflow:hidden;">
          <img class="hero-blink-img" src="${imgSrc}" data-ref="${c.refUrl}" data-sci="${c.sciUrl}" data-triplet="${c.tripletUrl}" style="width:100%;height:185px;object-fit:contain;" loading="lazy" />
        </div>
        <div class="event-body">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
            <span class="badge" style="background:${c.badgeColor}22;color:${c.badgeColor};border:1px solid ${c.badgeColor}66;">${c.badge}</span>
            <span style="font-size:0.74rem;color:#93c5fd;font-family:monospace;">${c.coords}</span>
          </div>
          <div style="font-weight:700;font-size:0.96rem;color:#f8fafc;margin-top:2px;">${c.title}</div>
          <div style="font-size:0.78rem;color:#6ee7b7;font-family:monospace;">${c.metrics}</div>
          <div style="font-size:0.77rem;color:#cbd5e1;line-height:1.38;">${c.detail}</div>
        </div>
      </div>
    `;
  }).join('');
}

async function loadHeroDeepSurvey() {
  try {
    const res = await fetch('/api/deep-survey');
    const data = await res.json();
    if (!data || !data.survey_overview) return;
    HERO_SURVEY_DATA = data;
    const ov = data.survey_overview;
    const sol = data.solar_system_recoveries || [];
    const trk = data.candidate_moving_tracklets || [];
    const ots = data.multi_frame_optical_transients || [];

    document.getElementById('heroDiscoveryKpis').innerHTML = `
      <div class="kpi" onclick="openCustomImageModal('Master J2000.0 Multi-Epoch Co-Added Reference Sky Atlas', '${ov.clear_nights_stacked} Clear Nights Stacked', '${ov.master_atlas_url}')">
        <div class="lbl">🌌 Clear Nights Co-Added</div>
        <div class="val" style="color:#38bdf8;">${ov.clear_nights_stacked} / ${ov.total_local_nights_scanned}</div>
        <div style="font-size:0.7rem;color:var(--muted);margin-top:2px;">${ov.total_sidereal_frames_coadded.toLocaleString()} sidereal frames (${ov.total_local_night_photos.toLocaleString()} local photos)</div>
      </div>
      <div class="kpi">
        <div class="lbl">🔭 Deep-Stack Depth Boost</div>
        <div class="val" style="color:#fbbf24;">${ov.limiting_magnitude_gain_mag}</div>
        <div style="font-size:0.7rem;color:var(--muted);margin-top:2px;">Reaches Est. Mag +${ov.estimated_deep_stack_limit_mag} (${ov.grid_resolution})</div>
      </div>
      <div class="kpi" onclick="setHeroDiscFilter('solsys', document.querySelectorAll('#heroDiscFilterBar .filter-btn')[1])">
        <div class="lbl">🪐 Solar System Worlds Tracked</div>
        <div class="val" style="color:#34d399;">${sol.length} Worlds</div>
        <div style="font-size:0.7rem;color:var(--muted);margin-top:2px;">Mars (10n), Jupiter (5n) & Ice Giant Uranus (Mag +5.61)</div>
      </div>
      <div class="kpi" onclick="setHeroDiscFilter('tracklet', document.querySelectorAll('#heroDiscFilterBar .filter-btn')[2])">
        <div class="lbl">🛸 Candidate Moving Tracklets</div>
        <div class="val" style="color:#e879f9;">${trk.length} Tracklets</div>
        <div style="font-size:0.7rem;color:var(--muted);margin-top:2px;">3-Night collinear sky motion (0.1°–3.0°/day)</div>
      </div>
      <div class="kpi" onclick="setHeroDiscFilter('transient', document.querySelectorAll('#heroDiscFilterBar .filter-btn')[3])">
        <div class="lbl">✨ Multi-Frame Optical Transients</div>
        <div class="val" style="color:#fb7185;">${ots.length} Candidates</div>
        <div style="font-size:0.7rem;color:var(--muted);margin-top:2px;">Verified across ≥4 exposures in a single night</div>
      </div>
      <div class="kpi" onclick="window.location.href='/discoveries'">
        <div class="lbl">☄️ Meteor Radiants & Watchlist</div>
        <div class="val" style="color:#a78bfa;">106 Meteors</div>
        <div style="font-size:0.7rem;color:var(--muted);margin-top:2px;">+ 10 Live Nova/Supernova Watch Targets →</div>
      </div>
    `;

    renderHeroDiscoveries();
  } catch (e) {
    console.error('Failed loading hero deep survey:', e);
  }
}

loadHeroDeepSurvey();
loadData();
setInterval(loadData, 8000);
setInterval(refreshLiveImage, 15000);
</script>
</body>
</html>
"""

LIVE_FULLSCREEN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>BluffSky Live Full-Frame Sky Radar (3552×3552)</title>
<style>
  :root {
    --bg: #030712;
    --panel: rgba(15, 23, 42, 0.88);
    --border: rgba(148, 163, 184, 0.22);
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #38bdf8;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; width: 100vw; height: 100vh;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    overflow: hidden; user-select: none;
  }
  .hud-top {
    position: fixed; top: 10px; left: 12px; right: 12px; z-index: 40;
    background: var(--panel); backdrop-filter: blur(14px);
    border: 1px solid var(--border); border-radius: 12px;
    padding: 9px 14px; display: flex; justify-content: space-between;
    align-items: center; flex-wrap: wrap; gap: 10px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.65);
    transition: opacity 0.25s ease, transform 0.25s ease;
  }
  .hud-top.collapsed {
    opacity: 0; pointer-events: none; transform: translateY(-20px);
  }
  .hud-brand {
    display: flex; align-items: center; gap: 10px;
  }
  .hud-brand h1 {
    margin: 0; font-size: 0.98rem; font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #a78bfa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hud-controls {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  }
  .chk-pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 10px; border-radius: 999px; font-size: 0.77rem; font-weight: 600;
    background: rgba(30, 41, 59, 0.9); border: 1px solid var(--border);
    cursor: pointer; color: #e2e8f0;
  }
  .chk-pill input { margin: 0; cursor: pointer; }
  .btn {
    background: linear-gradient(135deg, #0284c7, #2563eb);
    color: #fff; border: none; padding: 6px 12px; border-radius: 8px;
    font-size: 0.78rem; font-weight: 700; cursor: pointer; text-decoration: none;
    display: inline-flex; align-items: center; gap: 5px;
  }
  .btn.secondary {
    background: rgba(30, 41, 59, 0.92); border: 1px solid var(--border); color: #e2e8f0;
  }
  .btn.emerald { background: linear-gradient(135deg, #059669, #10b981); }
  .btn.amber { background: linear-gradient(135deg, #d97706, #f59e0b); }
  .viewport {
    position: fixed; inset: 0; width: 100vw; height: 100vh;
    display: flex; align-items: center; justify-content: center;
    cursor: grab; overflow: hidden; background: #02040a;
  }
  .viewport.dragging { cursor: grabbing; }
  #fullSkyImg {
    max-width: none; max-height: none;
    transform-origin: center center;
    transition: filter 0.15s ease;
    box-shadow: 0 0 60px rgba(0,0,0,0.9);
  }
  .drawer {
    position: fixed; top: 72px; right: 12px; bottom: 14px; width: 355px; z-index: 35;
    background: var(--panel); backdrop-filter: blur(14px);
    border: 1px solid var(--border); border-radius: 12px;
    display: flex; flex-direction: column; overflow: hidden;
    box-shadow: 0 12px 32px rgba(0,0,0,0.65);
    transition: transform 0.25s ease, opacity 0.25s ease;
  }
  .drawer.hidden {
    transform: translateX(380px); opacity: 0; pointer-events: none;
  }
  .drawer-hdr {
    padding: 11px 14px; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.85rem; font-weight: 700; color: #e2e8f0;
  }
  .drawer-body {
    flex: 1; overflow-y: auto; padding: 10px 12px; display: flex; flex-direction: column; gap: 8px;
  }
  .obj-card {
    background: rgba(15, 23, 42, 0.95); border: 1px solid var(--border);
    border-radius: 9px; padding: 8px 10px; font-size: 0.77rem; cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
  }
  .obj-card:hover {
    border-color: var(--accent); background: rgba(30, 41, 59, 0.95);
  }
  .obj-title {
    display: flex; justify-content: space-between; align-items: center;
    font-weight: 700; color: #f8fafc; margin-bottom: 3px;
  }
  .obj-sub { color: var(--muted); font-size: 0.72rem; line-height: 1.35; }
  .badge {
    font-size: 0.66rem; padding: 2px 6px; border-radius: 5px; font-weight: 700;
  }
  .badge.ac { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
  .badge.sat { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
  .badge.rb { background: rgba(249, 115, 22, 0.22); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.45); }
  .badge.cls { background: rgba(217, 70, 239, 0.22); color: #e879f9; border: 1px solid rgba(217, 70, 239, 0.45); }
  .hud-bottom {
    position: fixed; bottom: 12px; left: 12px; z-index: 35;
    background: var(--panel); backdrop-filter: blur(12px);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 7px 13px; font-size: 0.76rem; color: #cbd5e1;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  }
  .fab-toggle {
    position: fixed; top: 12px; right: 12px; z-index: 50;
    display: none;
  }
  .hud-top.collapsed ~ .fab-toggle { display: inline-flex; }
</style>
</head>
<body>

<div class="hud-top" id="hudTop">
  <div class="hud-brand">
    <a href="/" class="btn secondary" title="Return to BluffSky Portal">⬅ Portal</a>
    <div>
      <h1>🌌 BluffSky Full-Resolution Live Sky Radar (3552×3552 Native)</h1>
      <div id="liveTelemSub" style="font-size:0.72rem;color:var(--muted);">Loading live 3552×3552 sensor telemetry...</div>
    </div>
  </div>

  <div class="hud-controls">
    <label class="chk-pill" style="color:#fbbf24;">
      <input type="checkbox" id="chkAdsb" checked onchange="reloadFullFrame()"> ✈️ ADS-B Flights
    </label>
    <label class="chk-pill" style="color:#38bdf8;">
      <input type="checkbox" id="chkSats" checked onchange="reloadFullFrame()"> 🛰️ Sunlit Satellites & R/B
    </label>
    <label class="chk-pill" style="color:#a78bfa;">
      <input type="checkbox" id="chkStarlink" onchange="reloadFullFrame()"> 🌐 All Starlinks
    </label>
    <label class="chk-pill" style="color:#34d399;">
      <input type="checkbox" id="chkDetections" checked onchange="reloadFullFrame()"> ☄️ Latest Detections
    </label>
    <label class="chk-pill" style="color:#cbd5e1;">
      <input type="checkbox" id="chkCompass" checked onchange="reloadFullFrame()"> 🧭 Compass & El Grid
    </label>
    <label class="chk-pill" style="color:#6ee7b7;">
      <input type="checkbox" id="chkMask" onchange="reloadFullFrame()"> 🟢 Sky Mask
    </label>
    <label class="chk-pill" style="color:#e2e8f0;">
      <input type="checkbox" id="chkHud" checked onchange="reloadFullFrame()"> 📊 Image HUD Banner
    </label>

    <button class="btn secondary" onclick="zoomLiveBy(0.74)" title="Zoom Out (Key: -)">➖</button>
    <button class="btn secondary" onclick="fitToScreen()" title="Fit entire 3552×3552 dome in window (Key: 0)">📐 Fit</button>
    <button class="btn secondary" onclick="zoomNative()" title="1:1 Native 3552×3552 Pixel Resolution (Key: 1)">🔍 1:1 (3552px)</button>
    <button class="btn secondary" onclick="zoomLiveBy(1.35)" title="Zoom In (Key: +)">➕</button>
    <button class="btn secondary" onclick="reloadFullFrame(true)" title="Refresh live frame now (Key: R)">🔄 Refresh</button>
    <button class="btn secondary" onclick="toggleDrawer()" id="drawerBtn">📋 Radar List (<span id="radarTotalCnt">0</span>)</button>
    <a id="downloadLink" href="/api/live-image?full=1&adsb=1&sats=1&compass=1&detections=1&hud=1" download="allsky_live_3552x3552.jpg" class="btn secondary">💾 Save 3552px JPG</a>
    <button class="btn emerald" onclick="toggleBrowserFullscreen()" title="Toggle Browser Fullscreen (Key: F)">⛶ Fullscreen</button>
    <button class="btn secondary" onclick="toggleTopHud()" title="Hide/Show Top Control Bar (Key: H)">👁️ Hide Bar</button>
  </div>
</div>

<button class="btn secondary fab-toggle" onclick="toggleTopHud()">👁️ Show Controls (H)</button>

<div class="viewport" id="viewport">
  <img id="fullSkyImg" src="/api/live-image?full=1&adsb=1&sats=1&compass=1&detections=1&hud=1" alt="Live Full Resolution 3552x3552 Allsky Frame" draggable="false" />
</div>

<div class="drawer" id="radarDrawer">
  <div class="drawer-hdr">
    <span>📡 Live Objects in Camera Dome</span>
    <button class="btn secondary" style="padding:3px 8px;font-size:0.72rem;" onclick="toggleDrawer()">✕</button>
  </div>
  <div class="drawer-body" id="radarListBody">
    <div style="color:var(--muted);font-size:0.78rem;">Scanning overhead ADS-B & 13,212 orbital TLEs...</div>
  </div>
</div>

<div class="hud-bottom">
  <span id="zoomStatus">Zoom: Fit</span>
  <span>•</span>
  <span>🖱️ Scroll to Zoom In/Out As Much As You Want (8% – 4000%) • Drag to Pan • Double-Click to Zoom 2.5× • Click object in Radar List to center</span>
  <span>•</span>
  <label style="display:inline-flex;align-items:center;gap:5px;cursor:pointer;">
    <input type="checkbox" id="chkAutoRefresh" checked> Auto-Refresh (15s)
  </label>
</div>

<script>
const viewport = document.getElementById('viewport');
const img = document.getElementById('fullSkyImg');
let scale = 1.0;
let panX = 0;
let panY = 0;
let isDragging = false;
let startX = 0;
let startY = 0;
let initialFitDone = false;

function applyTransform() {
  img.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
  const pct = Math.round(scale * 100);
  document.getElementById('zoomStatus').textContent = `Zoom: ${pct}% (${img.naturalWidth || 3552}×${img.naturalHeight || 3552} native)`;
}

function fitToScreen() {
  const nw = img.naturalWidth || 3552;
  const nh = img.naturalHeight || 3552;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  scale = Math.min(vw / nw, vh / nh) * 0.96;
  panX = 0;
  panY = 0;
  applyTransform();
}

function zoomNative() {
  scale = 1.0;
  panX = 0;
  panY = 0;
  applyTransform();
}

function zoomLiveBy(factor, clientX, clientY) {
  const rect = viewport.getBoundingClientRect();
  const mx = (typeof clientX === 'number') ? (clientX - rect.left - rect.width / 2) : 0;
  const my = (typeof clientY === 'number') ? (clientY - rect.top - rect.height / 2) : 0;
  const nextScale = Math.max(0.08, Math.min(40.0, scale * factor));
  panX = mx - (mx - panX) * (nextScale / scale);
  panY = my - (my - panY) * (nextScale / scale);
  scale = nextScale;
  applyTransform();
}

function focusPixelOnDome(px, py) {
  const nw = img.naturalWidth || 3552;
  const nh = img.naturalHeight || 3552;
  scale = Math.max(scale, 0.85);
  const dxFromCenter = (nw / 2) - px;
  const dyFromCenter = (nh / 2) - py;
  panX = dxFromCenter * scale;
  panY = dyFromCenter * scale;
  applyTransform();
}

img.addEventListener('load', () => {
  if (!initialFitDone) {
    initialFitDone = true;
    fitToScreen();
  } else {
    applyTransform();
  }
});

window.addEventListener('resize', () => {
  if (initialFitDone) applyTransform();
});

viewport.addEventListener('wheel', (e) => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.18 : 0.85;
  zoomLiveBy(factor, e.clientX, e.clientY);
}, { passive: false });

viewport.addEventListener('dblclick', (e) => {
  zoomLiveBy(2.5, e.clientX, e.clientY);
});

viewport.addEventListener('mousedown', (e) => {
  if (e.button !== 0) return;
  isDragging = true;
  startX = e.clientX - panX;
  startY = e.clientY - panY;
  viewport.classList.add('dragging');
});

window.addEventListener('mousemove', (e) => {
  if (!isDragging) return;
  panX = e.clientX - startX;
  panY = e.clientY - startY;
  applyTransform();
});

window.addEventListener('mouseup', () => {
  isDragging = false;
  viewport.classList.remove('dragging');
});

function buildLiveImageUrl() {
  const adsb = document.getElementById('chkAdsb').checked ? 1 : 0;
  const sats = document.getElementById('chkSats').checked ? 1 : 0;
  const starlink = document.getElementById('chkStarlink').checked ? 1 : 0;
  const detections = document.getElementById('chkDetections').checked ? 1 : 0;
  const compass = document.getElementById('chkCompass').checked ? 1 : 0;
  const mask = document.getElementById('chkMask').checked ? 1 : 0;
  const hud = document.getElementById('chkHud').checked ? 1 : 0;
  return `/api/live-image?full=1&adsb=${adsb}&sats=${sats}&starlink=${starlink}&detections=${detections}&compass=${compass}&mask=${mask}&hud=${hud}`;
}

function reloadFullFrame() {
  const baseUrl = buildLiveImageUrl();
  document.getElementById('downloadLink').href = baseUrl;
  img.src = `${baseUrl}&t=${Date.now()}`;
  loadLiveRadarData();
}

function toggleDrawer() {
  document.getElementById('radarDrawer').classList.toggle('hidden');
}

function toggleTopHud() {
  document.getElementById('hudTop').classList.toggle('collapsed');
}

function toggleBrowserFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen().catch(() => {});
  }
}

window.addEventListener('keydown', (e) => {
  if (e.key === '0') fitToScreen();
  else if (e.key === '1') zoomNative();
  else if (e.key === '+' || e.key === '=') zoomLiveBy(1.35);
  else if (e.key === '-' || e.key === '_') zoomLiveBy(0.74);
  else if (e.key === 'f' || e.key === 'F') toggleBrowserFullscreen();
  else if (e.key === 'h' || e.key === 'H') toggleTopHud();
  else if (e.key === 'r' || e.key === 'R') reloadFullFrame();
});

async function loadLiveRadarData() {
  try {
    const starlink = document.getElementById('chkStarlink').checked ? 1 : 0;
    const res = await fetch(`/api/live-radar?starlink=${starlink}`);
    const data = await res.json();
    const live = data.live || {};
    const aircraft = data.aircraft_in_dome || [];
    const sats = data.satellites_in_dome || [];

    document.getElementById('radarTotalCnt').textContent = aircraft.length + sats.length;
    document.getElementById('liveTelemSub').textContent =
      `Frame: ${live.last_frame || '--'} • Mode: ${live.mode || 'NIGHT'} • Exp: ${live.exposure || '--'} (Gain ${live.gain || '--'}) • ✈️ ${aircraft.length} Flights • 🛰️ ${sats.length} Sunlit Orbital Objects`;

    const body = document.getElementById('radarListBody');
    let html = '';

    for (const ac of aircraft) {
      const px = ac.px || 1776;
      const py = ac.py || 1776;
      html += `
        <div class="obj-card" onclick="focusPixelOnDome(${px}, ${py})">
          <div class="obj-title">
            <span>✈️ ${ac.callsign} (${ac.type_code || 'ADS-B'})</span>
            <span class="badge ac">${Number(ac.alt_ft || 0).toLocaleString()} ft</span>
          </div>
          <div class="obj-sub">
            ${ac.model || 'Aircraft'} • ${Math.round(ac.gs_kt || 0)} kt • Dist ${ac.ground_dist_miles ?? '--'} mi<br/>
            Dome El ${ac.alt_deg}° • Az ${ac.az_deg}° • Pixel (${px}, ${py})
          </div>
        </div>
      `;
    }

    for (const s of sats) {
      let badgeCls = 'sat';
      let badgeTxt = 'PAYLOAD';
      if (s.is_rocket_body) { badgeCls = 'rb'; badgeTxt = 'ROCKET BODY'; }
      else if (s.is_classified) { badgeCls = 'cls'; badgeTxt = 'CLASSIFIED'; }
      else if (s.source_key === 'stations') { badgeCls = 'ac'; badgeTxt = 'STATION'; }

      html += `
        <div class="obj-card" onclick="focusPixelOnDome(${s.px_end}, ${s.py_end})">
          <div class="obj-title">
            <span>🛰️ ${s.name} (#${s.norad_id})</span>
            <span class="badge ${badgeCls}">${badgeTxt}</span>
          </div>
          <div class="obj-sub">
            Alt ${s.orbit_alt_km} km (${s.orbit_alt_miles} mi) • ${s.speed_kms} km/s (${Number(s.speed_mph).toLocaleString()} mph)<br/>
            Dome El ${s.alt_deg}° • Az ${s.az_deg}° • Pixel (${s.px_end}, ${s.py_end})
          </div>
        </div>
      `;
    }

    if (!html) {
      html = '<div style="color:var(--muted);font-size:0.78rem;padding:8px;">No sunlit satellites or aircraft above horizon right now.</div>';
    }
    body.innerHTML = html;
  } catch (e) {
    console.error('Failed loading live radar data:', e);
  }
}

loadLiveRadarData();
setInterval(() => {
  if (document.getElementById('chkAutoRefresh').checked) {
    reloadFullFrame();
  }
}, 15000);
</script>
</body>
</html>
"""


def _parse_exposure_seconds(exp_str: str) -> float:
    if not exp_str:
        return 30.0
    s = str(exp_str).strip().lower()
    try:
        if s.endswith("ms"):
            return max(0.05, float(s[:-2].strip()) / 1000.0)
        if s.endswith("us"):
            return max(0.01, float(s[:-2].strip()) / 1_000_000.0)
        if s.endswith("s"):
            return max(0.05, float(s[:-1].strip()))
        return max(0.05, float(s))
    except Exception:
        return 30.0


def render_live_full_frame_overlay(
    img: np.ndarray,
    cfg: dict,
    show_mask: bool = False,
    show_compass: bool = True,
    show_sats: bool = True,
    show_starlink: bool = False,
    show_adsb: bool = True,
    show_detections: bool = True,
    show_hud: bool = True,
) -> np.ndarray:
    """
    Renders all live calibrated sky overlays directly onto the native 3552x3552
    (or resized) Allsky camera frame.
    """
    import detector
    import adsb_tracker

    out = img.copy()
    h, w = out.shape[:2]
    scale = max(0.45, w / 2200.0)

    # Read live status for exposure duration and metadata
    live_meta = {}
    if os.path.exists(LIVE_STATUS_FILE):
        try:
            with open(LIVE_STATUS_FILE, "r", encoding="utf-8") as f:
                live_meta = json.load(f)
        except Exception:
            pass
    exp_sec = _parse_exposure_seconds(live_meta.get("exposure", "30s"))
    now_ts = time.time()

    # 1. Calibrated Sky Mask Contour
    if show_mask:
        try:
            mask = detector.ensure_sky_mask(cfg, (h, w))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, (0, 255, 160), max(2, int(round(4 * scale))))
        except Exception:
            pass

    # 2. Calibrated Compass Rose & Elevation Grid (30 deg, 60 deg, Zenith +)
    if show_compass:
        try:
            grid_col = (110, 95, 75)
            thick_g = max(1, int(round(1.4 * scale)))
            for el_ring in (30.0, 60.0):
                pts = []
                for az_step in range(0, 361, 4):
                    px, py = adsb_tracker.altaz_to_pixel(el_ring, float(az_step), cfg, (h, w))
                    pts.append([px, py])
                pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts_arr], False, grid_col, thick_g, cv2.LINE_AA)
                lx, ly = adsb_tracker.altaz_to_pixel(el_ring, 0.0, cfg, (h, w))
                cv2.putText(
                    out,
                    f"{int(el_ring)} deg El",
                    (lx + int(8 * scale), ly - int(6 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48 * scale,
                    (170, 160, 140),
                    max(1, int(round(1.4 * scale))),
                    cv2.LINE_AA,
                )

            # Zenith crosshair
            zx, zy = adsb_tracker.altaz_to_pixel(90.0, 0.0, cfg, (h, w))
            cr = int(14 * scale)
            cv2.line(out, (zx - cr, zy), (zx + cr, zy), (160, 220, 255), max(1, int(round(1.8 * scale))), cv2.LINE_AA)
            cv2.line(out, (zx, zy - cr), (zx, zy + cr), (160, 220, 255), max(1, int(round(1.8 * scale))), cv2.LINE_AA)
            cv2.putText(
                out,
                "ZENITH",
                (zx + int(10 * scale), zy - int(10 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48 * scale,
                (160, 220, 255),
                max(1, int(round(1.5 * scale))),
                cv2.LINE_AA,
            )

            # Compass cardinal & intercardinal markers at 12 deg elevation
            for label_c, az_c, is_card in (
                ("N", 0.0, True),
                ("NE", 45.0, False),
                ("E", 90.0, True),
                ("SE", 135.0, False),
                ("S", 180.0, True),
                ("SW", 225.0, False),
                ("W", 270.0, True),
                ("NW", 315.0, False),
            ):
                cx, cy = adsb_tracker.altaz_to_pixel(12.0, az_c, cfg, (h, w))
                if 25 <= cx < w - 40 and 25 <= cy < h - 25:
                    f_sc = (0.85 if is_card else 0.58) * scale
                    f_th = max(2, int(round((2.6 if is_card else 1.8) * scale)))
                    col_c = (90, 230, 255) if label_c == "N" else ((235, 235, 235) if is_card else (175, 175, 175))
                    cv2.circle(out, (cx, cy), max(3, int(5 * scale)), col_c, -1, cv2.LINE_AA)
                    cv2.putText(
                        out,
                        label_c,
                        (cx + int(10 * scale), cy + int(8 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        f_sc,
                        col_c,
                        f_th,
                        cv2.LINE_AA,
                    )
        except Exception as e_comp:
            logger.debug(f"Compass overlay skipped: {e_comp}")

    # 3. Sunlit Orbital Satellites, Space Stations, Rocket Bodies & Classified Objects
    sats_count = 0
    if show_sats and cfg.get("orbital_tracking_enabled", True):
        try:
            import orbital_tracker
            sats_list = orbital_tracker.get_live_orbital_snapshot(
                cfg=cfg,
                exp_end_ts=now_ts,
                exposure_sec=exp_sec,
                include_starlink=show_starlink,
                img_shape=(h, w),
            )
            sats_count = len(sats_list)
            out = orbital_tracker.draw_orbital_overlay_on_image(
                out,
                cfg=cfg,
                exp_end_ts=now_ts,
                exposure_sec=exp_sec,
                include_starlink=show_starlink,
            )
        except Exception as e_sat:
            logger.debug(f"Orbital overlay skipped: {e_sat}")

    # 4. Live ADS-B Aircraft
    adsb_count = 0
    if show_adsb and cfg.get("adsb_enabled", True):
        try:
            snap_adsb = adsb_tracker.get_live_adsb_snapshot()
            adsb_count = len(snap_adsb.get("aircraft_in_dome", []))
            out = adsb_tracker.draw_adsb_overlay_on_image(out, cfg=cfg, exposure_sec=exp_sec)
        except Exception as e_adsb:
            logger.debug(f"ADS-B overlay skipped: {e_adsb}")

    # 5. Latest Frame Anomaly Detections (if detected on current or recent frame within 10 min)
    if show_detections and os.path.exists(DB_FILE):
        try:
            conn = sqlite3.connect(DB_FILE)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT frame_filename, detected_at, category, subtype, subject, confidence,
                       alt_deg, az_deg, cardinal, features_json
                FROM sky_events
                WHERE COALESCE(is_hidden, 0) = 0
                ORDER BY id DESC LIMIT 4
                """
            )
            rows = cur.fetchall()
            conn.close()
            latest_frame_name = live_meta.get("last_frame", "")
            sx = w / 3552.0
            sy = h / 3552.0
            for r in rows:
                is_current = (r["frame_filename"] == latest_frame_name)
                age_s = now_ts - float(r["detected_at"] or 0.0)
                if not is_current and age_s > 600.0:
                    continue
                fj = json.loads(r["features_json"] or "{}")
                bbox = fj.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1 = int(round(float(bbox[0]) * sx))
                y1 = int(round(float(bbox[1]) * sy))
                x2 = int(round(float(bbox[2]) * sx))
                y2 = int(round(float(bbox[3]) * sy))
                cx_b, cy_b = (x1 + x2) // 2, (y1 + y2) // 2
                pad = int(round(28 * scale))
                min_h = int(round(42 * scale))
                bx1 = max(12, min(x1 - pad, cx_b - min_h))
                by1 = max(92, min(y1 - pad, cy_b - min_h))
                bx2 = min(w - 12, max(x2 + pad, cx_b + min_h))
                by2 = min(h - 12, max(y2 + pad, cy_b + min_h))

                orb = fj.get("orbital_object")
                ac = fj.get("adsb_flight")
                det_col = (255, 225, 0) if orb else ((0, 215, 255) if ac else (80, 255, 130))
                thick_b = max(2, int(round(3.0 * scale)))

                cv2.rectangle(out, (bx1 - 2, by1 - 2), (bx2 + 2, by2 + 2), (12, 16, 28), thick_b + 2, cv2.LINE_AA)
                cv2.rectangle(out, (bx1, by1), (bx2, by2), det_col, thick_b, cv2.LINE_AA)

                raw_subj = re.sub(r"[^\x20-\x7E]+", "", str(r["subject"] or r["subtype"] or "EVENT")).strip()
                prefix = "LIVE EVENT: " if is_current else f"RECENT ({int(max(1, age_s // 60))}m ago): "
                top_lbl = f"{prefix}{raw_subj} | El {float(r['alt_deg'] or 0):.0f}deg {r['cardinal'] or ''}"
                f_sc = 0.62 * scale
                f_th = max(2, int(round(2.0 * scale)))
                (tw, th), _ = cv2.getTextSize(top_lbl, cv2.FONT_HERSHEY_SIMPLEX, f_sc, f_th)
                tx = max(12, min(w - tw - 20, bx1))
                ty = max(int(96 * scale) + th, by1 - int(12 * scale))
                cv2.rectangle(out, (tx - 6, ty - th - 8), (tx + tw + 8, ty + 6), det_col, -1)
                cv2.putText(out, top_lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, f_sc, (10, 14, 24), f_th, cv2.LINE_AA)

                if orb and orb.get("name"):
                    bot_lbl = (
                        f"SAT TAG: {orb.get('name')} (NORAD #{orb.get('norad_id')}) | "
                        f"Alt {float(orb.get('orbit_alt_km') or 0):,.0f} km ({float(orb.get('orbit_alt_miles') or 0):,.0f} mi) | "
                        f"{int(orb.get('speed_mph') or 0):,} mph"
                    )
                elif ac and ac.get("callsign"):
                    bot_lbl = (
                        f"FLIGHT TAG: {ac.get('callsign')} ({ac.get('model_name') or ac.get('type_code') or 'ADS-B'}) | "
                        f"{int(ac.get('alt_ft') or 0):,} ft | {int(ac.get('gs_kt') or 0)} kt"
                    )
                else:
                    bot_lbl = ""

                if bot_lbl:
                    (btw, bth), _ = cv2.getTextSize(bot_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.54 * scale, max(1, f_th - 1))
                    btx = max(12, min(w - btw - 20, bx1))
                    bty = min(h - 16, by2 + bth + int(14 * scale))
                    cv2.rectangle(out, (btx - 6, bty - bth - 8), (btx + btw + 8, bty + 6), (12, 18, 32), -1)
                    cv2.rectangle(out, (btx - 6, bty - bth - 8), (btx + btw + 8, bty + 6), det_col, max(1, f_th - 1))
                    cv2.putText(out, bot_lbl, (btx, bty), cv2.FONT_HERSHEY_SIMPLEX, 0.54 * scale, det_col, max(1, f_th - 1), cv2.LINE_AA)
        except Exception as e_det:
            logger.debug(f"Live detection overlay skipped: {e_det}")

    # 6. Top Telemetry HUD Banner
    if show_hud:
        try:
            bar_h = int(round(86 * scale))
            overlay_bar = out[:bar_h, :].copy()
            cv2.rectangle(overlay_bar, (0, 0), (w, bar_h), (8, 12, 24), -1)
            cv2.addWeighted(overlay_bar, 0.78, out[:bar_h, :], 0.22, 0, out[:bar_h, :])
            cv2.line(out, (0, bar_h), (w, bar_h), (200, 140, 40), max(2, int(round(2 * scale))))

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            mode_s = live_meta.get("mode", "NIGHT")
            exp_s = live_meta.get("exposure", "30s")
            gain_s = live_meta.get("gain", "185")
            temp_s = live_meta.get("temperature_f") or live_meta.get("temp_c") or "--"
            moon_s = live_meta.get("moon_illumination", "--")

            line1 = f"BLUFFSKY ALLSKY OBSERVATORY ({w}x{h})  |  {now_str}  |  Mode: {mode_s}  |  Exp: {exp_s}  |  Gain: {gain_s}  |  Temp: {temp_s}F  |  Moon: {moon_s}%"
            line2 = f"LIVE SKY RADAR OVERLAYS:  {adsb_count} ADS-B Aircraft Overhead   |   {sats_count} Sunlit Satellites / Space Stations / Rocket Bodies in Dome"

            cv2.putText(
                out,
                line1,
                (int(24 * scale), int(34 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68 * scale,
                (245, 250, 255),
                max(2, int(round(2.1 * scale))),
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                line2,
                (int(24 * scale), int(68 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60 * scale,
                (255, 210, 80),
                max(1, int(round(1.9 * scale))),
                cv2.LINE_AA,
            )
        except Exception:
            pass

    return out


class AllskyPortalHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send_html(self, html_str: str):
        raw = html_str.encode("utf-8")
        use_gzip = len(raw) > 1024 and "gzip" in (self.headers.get("Accept-Encoding") or "")
        body = gzip.compress(raw, compresslevel=5) if use_gzip else raw
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict):
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        use_gzip = len(raw) > 1024 and "gzip" in (self.headers.get("Accept-Encoding") or "")
        body = gzip.compress(raw, compresslevel=5) if use_gzip else raw
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_cached_bytes(
        self,
        etag: str,
        raw_bytes: bytes,
        gzip_bytes: bytes,
        content_type: str = "application/json; charset=utf-8",
    ):
        inm = (self.headers.get("If-None-Match") or "").strip()
        if etag and inm == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return
        use_gzip = len(raw_bytes) > 1024 and "gzip" in (self.headers.get("Accept-Encoding") or "")
        body = gzip_bytes if use_gzip else raw_bytes
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Vary", "Accept-Encoding")
        if etag:
            self.send_header("ETag", etag)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static_file(self, fpath: str, content_type: str = "image/jpeg", max_age: int = 3600):
        try:
            st = os.stat(fpath)
        except OSError:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        etag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
        inm = (self.headers.get("If-None-Match") or "").strip()
        if inm == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", f"public, max-age={max_age}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        with open(fpath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", f"public, max-age={max_age}")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self._send_html(PORTAL_HTML)
            return

        if path in ("/live", "/full", "/fullscreen"):
            self._send_html(LIVE_FULLSCREEN_HTML)
            return

        if path in ("/discoveries", "/patrol", "/astrophysics"):
            try:
                import astrophysics_patrol
                self._send_html(astrophysics_patrol.get_discovery_patrol_html())
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
            return

        if path in ("/astrophoto", "/astrophotography", "/milkyway"):
            try:
                import deep_sky_astrophoto
                self._send_html(deep_sky_astrophoto.get_astrophoto_gallery_html())
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
            return

        if path == "/api/astrophoto":
            try:
                import deep_sky_astrophoto
                self._send_json(200, deep_sky_astrophoto.get_astrophoto_report())
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
            return

        if path == "/api/discoveries":
            try:
                import astrophysics_patrol
                self._send_json(200, astrophysics_patrol.build_discovery_patrol_report())
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
            return

        if path == "/api/deep-survey":
            try:
                import deep_sky_survey
                self._send_json(200, deep_sky_survey.get_deep_survey_report(auto_run_if_missing=True))
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
            return

        if path == "/api/realign/status":
            try:
                import camera_realign
                self._send_json(200, camera_realign.load_realign_state())
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
            return

        if path == "/api/version":
            self._send_json(200, {"status": "ok", "service": "AllskyAnomalyPortal", "version": "1.4.0"})
            return

        if path in ("/api/stats", "/api/events"):
            etag, raw_bytes, gzip_bytes = get_serialized_stats_response()
            self._send_cached_bytes(etag, raw_bytes, gzip_bytes)
            return

        if path == "/api/adsb":
            try:
                import adsb_tracker
                self._send_json(200, adsb_tracker.get_live_adsb_snapshot())
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
            return

        if path == "/api/live-radar":
            try:
                import detector
                import adsb_tracker
                import orbital_tracker

                cfg = detector.load_config()
                include_starlink = (query.get("starlink") or ["0"])[0] == "1"
                live_meta = {}
                if os.path.exists(LIVE_STATUS_FILE):
                    try:
                        with open(LIVE_STATUS_FILE, "r", encoding="utf-8") as f:
                            live_meta = json.load(f)
                    except Exception:
                        pass
                exp_sec = _parse_exposure_seconds(live_meta.get("exposure", "30s"))
                adsb_snap = adsb_tracker.get_live_adsb_snapshot()
                aircraft_in_dome = []
                for ac in adsb_snap.get("aircraft_in_dome", []):
                    ac_copy = dict(ac)
                    px, py = adsb_tracker.altaz_to_pixel(
                        float(ac.get("alt_deg") or 0.0),
                        float(ac.get("az_deg") or 0.0),
                        cfg,
                        (3552, 3552),
                    )
                    ac_copy["px"] = px
                    ac_copy["py"] = py
                    aircraft_in_dome.append(ac_copy)

                sats_in_dome = orbital_tracker.get_live_orbital_snapshot(
                    cfg=cfg,
                    exp_end_ts=time.time(),
                    exposure_sec=exp_sec,
                    include_starlink=include_starlink,
                    img_shape=(3552, 3552),
                )
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "live": live_meta,
                        "aircraft_in_dome": aircraft_in_dome,
                        "satellites_in_dome": sats_in_dome,
                    },
                )
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
            return

        if path == "/api/feedback":
            try:
                import feedback_learner
                self._send_json(200, feedback_learner.get_feedback_summary())
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
            return

        if path == "/api/top-interesting":
            data = get_events_and_stats()
            self._send_json(200, {"top_50": data["top_50"], "count": len(data["top_50"])})
            return

        if path == "/api/health":
            try:
                import allsky_workflow_guard
                if os.path.exists(allsky_workflow_guard.GUARD_STATUS_FILE):
                    with open(allsky_workflow_guard.GUARD_STATUS_FILE, "r", encoding="utf-8") as f:
                        self._send_json(200, json.load(f))
                        return
                rep = allsky_workflow_guard.run_guard_cycle(trigger_discord_alerts=False)
                self._send_json(200, rep)
                return
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
                return

        if path == "/api/cameras":
            try:
                import detector
                cfg = detector.load_config()
                cams = detector.get_configured_cameras(cfg, enabled_only=False)
                self._send_json(200, {"status": "ok", "count": len(cams), "cameras": cams})
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
            return

        if path == "/api/live-image":
            cam_id = str((query.get("camera_id") or ["cam1"])[0] or "cam1").strip()
            is_full = (query.get("full") or ["0"])[0] == "1"
            show_mask = (query.get("mask") or ["0"])[0] == "1"
            show_adsb = (query.get("adsb") or ["1"])[0] == "1"
            show_sats = (query.get("sats") or ["1"])[0] == "1"
            show_starlink = (query.get("starlink") or ["0"])[0] == "1"
            show_compass = (query.get("compass") or (["1"] if is_full else ["0"]))[0] == "1"
            show_detections = (query.get("detections") or ["1"])[0] == "1"
            show_hud = (query.get("hud") or (["1"] if is_full else ["0"]))[0] == "1"
            try:
                import detector
                live_frame_file = detector.get_camera_live_frame_path(cam_id)
                now_ts = time.time()
                frame_mtime = _safe_mtime(live_frame_file)
                render_key = (
                    cam_id,
                    is_full,
                    show_mask,
                    show_adsb,
                    show_sats,
                    show_starlink,
                    show_compass,
                    show_detections,
                    show_hud,
                    frame_mtime,
                )
                with _CACHE_LOCK:
                    cached_entry = _LIVE_IMG_RENDER_CACHE.get(render_key)
                if cached_entry is not None and (now_ts - cached_entry[0]) < 3.5:
                    jpg_bytes = cached_entry[1]
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg_bytes)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(jpg_bytes)
                    return

                cfg = detector.load_config()
                cams = detector.get_configured_cameras(cfg, enabled_only=False)
                cam_entry = next((c for c in cams if c["id"] == cam_id), None)
                cam_cfg = detector.get_effective_camera_cfg(cfg, cam_entry) if cam_entry else cfg
                raw_frame_bytes: Optional[bytes] = None
                if frame_mtime > 0 and (now_ts - frame_mtime) < 300.0:
                    try:
                        with open(live_frame_file, "rb") as f:
                            raw_frame_bytes = f.read()
                    except Exception:
                        raw_frame_bytes = None

                if not raw_frame_bytes:
                    live_url = cam_cfg.get("live_image_url", "http://192.168.1.100/current/tmp/image.jpg")
                    try:
                        r = requests.get(live_url, timeout=18)
                        if r.status_code == 200 and len(r.content) > 1000:
                            raw_frame_bytes = r.content
                            try:
                                tmp_live = live_frame_file + f".{threading.get_ident()}.tmp"
                                with open(tmp_live, "wb") as lf:
                                    lf.write(raw_frame_bytes)
                                os.replace(tmp_live, live_frame_file)
                                frame_mtime = _safe_mtime(live_frame_file)
                            except Exception:
                                pass
                    except Exception as fetch_err:
                        if frame_mtime > 0:
                            try:
                                with open(live_frame_file, "rb") as f:
                                    raw_frame_bytes = f.read()
                            except Exception:
                                pass
                        else:
                            raise fetch_err

                if raw_frame_bytes:
                    arr = np.frombuffer(raw_frame_bytes, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = render_live_full_frame_overlay(
                            img=img,
                            cfg=cam_cfg,
                            show_mask=show_mask,
                            show_compass=show_compass,
                            show_sats=show_sats,
                            show_starlink=show_starlink,
                            show_adsb=show_adsb,
                            show_detections=show_detections,
                            show_hud=show_hud,
                        )
                        if is_full:
                            out_img = img
                            jpg_qual = 93
                        else:
                            out_img = cv2.resize(img, (1080, 1080), interpolation=cv2.INTER_AREA)
                            jpg_qual = 86
                        ok, buf = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_qual])
                        if ok:
                            jpg_bytes = buf.tobytes()
                            final_key = (
                                cam_id,
                                is_full,
                                show_mask,
                                show_adsb,
                                show_sats,
                                show_starlink,
                                show_compass,
                                show_detections,
                                show_hud,
                                frame_mtime,
                            )
                            with _CACHE_LOCK:
                                if len(_LIVE_IMG_RENDER_CACHE) > 24:
                                    _LIVE_IMG_RENDER_CACHE.clear()
                                _LIVE_IMG_RENDER_CACHE[final_key] = (time.time(), jpg_bytes)
                            self.send_response(200)
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(jpg_bytes)))
                            self.send_header("Cache-Control", "no-cache")
                            self.end_headers()
                            self.wfile.write(jpg_bytes)
                            return
            except Exception as e:
                logger.warning(f"Live image proxy error ({cam_id}): {e}")
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if path.startswith("/snapshots/"):
            fname = os.path.basename(unquote(path[len("/snapshots/") :]))
            fpath = os.path.join(SNAPSHOT_DIR, fname)
            if os.path.isfile(fpath):
                self._serve_static_file(fpath, content_type="image/jpeg", max_age=3600)
                return
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if path.startswith("/survey_data/"):
            rel_part = unquote(path[len("/survey_data/") :]).replace("\\", "/").lstrip("/")
            if ".." not in rel_part:
                survey_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "survey_data")
                fpath = os.path.join(survey_root, rel_part)
                if os.path.isfile(fpath):
                    ctype = "application/json; charset=utf-8" if fpath.lower().endswith(".json") else ("image/png" if fpath.lower().endswith(".png") else "image/jpeg")
                    self._serve_static_file(fpath, content_type=ctype, max_age=3600)
                    return
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if path.startswith("/astrophoto_data/"):
            rel_part = unquote(path[len("/astrophoto_data/") :]).replace("\\", "/").lstrip("/")
            if ".." not in rel_part:
                astro_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "astrophoto_data")
                fpath = os.path.join(astro_root, rel_part)
                if os.path.isfile(fpath):
                    ctype = "application/json; charset=utf-8" if fpath.lower().endswith(".json") else ("image/png" if fpath.lower().endswith(".png") else "image/jpeg")
                    self._serve_static_file(fpath, content_type=ctype, max_age=3600)
                    return
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        self._send_json(404, {"status": "error", "detail": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body_raw = self.rfile.read(length).decode("utf-8", errors="replace") if length > 0 else "{}"
        try:
            data = json.loads(body_raw)
        except Exception:
            data = {}

        if path == "/api/reclassify":
            fname = str(data.get("frame_filename") or "").strip()
            new_cat = str(data.get("new_category") or "").strip()
            note = str(data.get("note") or "").strip()
            if not fname or not new_cat:
                self._send_json(400, {"status": "error", "detail": "Missing frame_filename or new_category"})
                return
            try:
                import feedback_learner
                res = feedback_learner.record_user_correction(
                    frame_filename=fname,
                    corrected_category=new_cat,
                    user_note=note or f"Reclassified to {new_cat}",
                    source="web_portal",
                    discord_user="WebObservatory",
                )
                if res:
                    bump_events_cache()
                    try:
                        import astrophysics_patrol
                        astrophysics_patrol.invalidate_patrol_cache()
                    except Exception:
                        pass
                    self._send_json(
                        200,
                        {
                            "status": "ok",
                            "detail": f"🧠 Learned! Reclassified {fname} from {res.get('old_category')} → {res.get('corrected_subject')} (Total training samples: {res.get('total_training_samples')})",
                            "result": res,
                        },
                    )
                else:
                    self._send_json(404, {"status": "error", "detail": f"Event {fname} not found or invalid category {new_cat}"})
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
            return

        if path == "/api/event-action":
            init_db()
            action = data.get("action")
            fname = data.get("frame_filename")
            if not fname:
                self._send_json(400, {"status": "error", "detail": "Missing frame_filename"})
                return
            with DB_LOCK:
                conn = get_db_connection()
                try:
                    if action == "star":
                        conn.execute(
                            "UPDATE sky_events SET is_starred = CASE WHEN is_starred = 1 THEN 0 ELSE 1 END WHERE frame_filename = ?",
                            (fname,),
                        )
                    elif action == "hide":
                        conn.execute("UPDATE sky_events SET is_hidden = 1 WHERE frame_filename = ?", (fname,))
                    conn.commit()
                finally:
                    conn.close()
            bump_events_cache()
            try:
                import astrophysics_patrol
                astrophysics_patrol.invalidate_patrol_cache()
            except Exception:
                pass
            self._send_json(200, {"status": "ok", "action": action, "frame_filename": fname})
            return

        if path == "/api/config":
            import detector
            cfg = detector.load_config()
            for key in ("night_k_sigma", "meteor_min_length_px", "alert_on_aircraft", "day_detection_enabled"):
                if key in data:
                    cfg[key] = data[key]

            wh_msg = ""
            if "allsky_webhook_url" in data:
                wh_url = str(data["allsky_webhook_url"] or "").strip()
                if wh_url and not wh_url.startswith("https://discord"):
                    self._send_json(400, {"status": "error", "detail": "Invalid Discord Webhook URL format."})
                    return
                cfg["discord_webhook_url"] = wh_url
            if "discord_bot_token" in data and str(data["discord_bot_token"] or "").strip():
                cfg["discord_bot_token"] = str(data["discord_bot_token"]).strip()
            if "discord_client_id" in data:
                cfg["discord_client_id"] = str(data["discord_client_id"] or "").strip()
            detector.save_config(cfg)

            if os.path.exists(DISCORD_CONFIG_FILE):
                try:
                    with open(DISCORD_CONFIG_FILE, "r", encoding="utf-8") as f:
                        d_cfg = json.load(f)
                    if "allsky_webhook_url" in data:
                        wh_url = str(data["allsky_webhook_url"] or "").strip()
                        if wh_url and not wh_url.startswith("https://discord"):
                            self._send_json(400, {"status": "error", "detail": "Invalid Discord Webhook URL format."})
                            return
                        d_cfg.setdefault("webhooks", {})["allsky"] = wh_url
                        wh_msg += " + #all-sky webhook updated"
                    if "discord_bot_token" in data and str(data["discord_bot_token"] or "").strip():
                        d_cfg["discord_bot_token"] = str(data["discord_bot_token"]).strip()
                        wh_msg += " + Discord Bot Token saved"
                    if "discord_client_id" in data:
                        d_cfg["discord_client_id"] = str(data["discord_client_id"] or "").strip()
                    with open(DISCORD_CONFIG_FILE, "w", encoding="utf-8") as f:
                        json.dump(d_cfg, f, indent=4)
                except Exception as e:
                    logger.error(f"Failed updating Discord config: {e}")

            bump_events_cache()
            self._send_json(200, {"status": "ok", "detail": f"Saved Allsky detector settings{wh_msg}!"})
            return

        if path == "/api/cameras":
            try:
                import detector
                cfg = detector.load_config()
                cams = list(cfg.get("cameras") or detector.get_configured_cameras(cfg, enabled_only=False))
                if isinstance(data.get("cameras"), list):
                    cfg["cameras"] = data["cameras"]
                elif data.get("id") or data.get("base_url"):
                    cid = str(data.get("id") or f"cam{len(cams) + 1}").strip()
                    updated = False
                    for idx_c, ex_c in enumerate(cams):
                        if str(ex_c.get("id")) == cid:
                            cams[idx_c] = {**ex_c, **data, "id": cid}
                            updated = True
                            break
                    if not updated:
                        cams.append({**data, "id": cid})
                    cfg["cameras"] = cams
                detector.save_config(cfg)
                bump_events_cache()
                final_cams = detector.get_configured_cameras(cfg, enabled_only=False)
                self._send_json(200, {"status": "ok", "count": len(final_cams), "cameras": final_cams})
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
            return

        if path == "/api/simulate":
            import detector
            sim_type = data.get("type", "meteor")
            send_d = bool(data.get("send_discord", True))
            ev = detector.run_synthetic_night_test(event_type=sim_type, send_discord=send_d)
            if ev:
                bump_events_cache()
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "detail": f"Detected {ev['subject']} @ Alt {ev['alt_deg']}° {ev['cardinal']} (SNR {ev['snr']}σ) — Discord sent: {ev.get('discord_sent')}",
                        "event": ev,
                    },
                )
            else:
                self._send_json(500, {"status": "error", "detail": "Synthetic night test did not produce an event."})
            return

        if path in ("/api/heal", "/api/workflow/heal"):
            try:
                import allsky_workflow_guard
                rep = allsky_workflow_guard.run_guard_cycle(trigger_discord_alerts=True)
                self._send_json(200, {"status": "ok", "report": rep})
                return
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
                return

        if path == "/api/deep-survey/scan":
            try:
                import deep_sky_survey
                force = bool(data.get("force_rebuild", False))
                rep = deep_sky_survey.run_multi_epoch_deep_survey(force_rebuild=force)
                self._send_json(200, {"status": "ok", "report": rep})
                return
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
                return

        if path == "/api/astrophoto/build":
            try:
                import deep_sky_astrophoto
                max_f = int(data.get("max_frames", 380))
                st = deep_sky_astrophoto.trigger_background_build(max_frames=max_f)
                self._send_json(200, {"status": "ok", "build_status": st})
                return
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
                return

        if path == "/api/realign/arm":
            try:
                import camera_realign
                update_mask = bool(data.get("update_day_mask", True))
                st = camera_realign.arm_camera_realignment(update_day_mask=update_mask)
                bump_events_cache()
                self._send_json(200, {"status": "ok", "realignment": st})
                return
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
                return

        if path == "/api/realign/cancel":
            try:
                import camera_realign
                st = camera_realign.cancel_camera_realignment()
                bump_events_cache()
                self._send_json(200, {"status": "ok", "realignment": st})
                return
            except Exception as e:
                self._send_json(500, {"status": "error", "detail": str(e)})
                return

        self._send_json(404, {"status": "error", "detail": "Unknown endpoint"})


class ExclusiveThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        if hasattr(socket, "IPPROTO_IPV6") and hasattr(socket, "IPV6_V6ONLY"):
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except Exception:
                pass
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            except Exception:
                pass
        self.socket.bind(self.server_address)
        self.server_address = self.socket.getsockname()
        self.server_name = "bluffsky"
        self.server_port = int(self.server_address[1])

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class ExclusiveIPv4ThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            except Exception:
                pass
        super().server_bind()


def _prewarm_caches_background() -> None:
    try:
        get_serialized_stats_response()
    except Exception as e:
        logger.debug(f"Stats cache pre-warm skipped: {e}")
    try:
        import orbital_tracker
        orbital_tracker.ensure_catalog_loaded()
    except Exception as e:
        logger.debug(f"Orbital catalog pre-warm skipped: {e}")
    try:
        if not os.path.exists(LATEST_LIVE_FRAME_FILE):
            import detector
            cfg = detector.load_config()
            live_url = cfg.get("live_image_url", "http://192.168.1.100/current/tmp/image.jpg")
            r = requests.get(live_url, timeout=22)
            if r.status_code == 200 and len(r.content) > 1000:
                tmp_path = LATEST_LIVE_FRAME_FILE + ".prewarm.tmp"
                with open(tmp_path, "wb") as f:
                    f.write(r.content)
                os.replace(tmp_path, LATEST_LIVE_FRAME_FILE)
    except Exception as e:
        logger.debug(f"Live frame pre-warm skipped: {e}")


def run_server(port: int = 8094) -> None:
    try:
        server = ExclusiveThreadingHTTPServer(("::", port), AllskyPortalHandler)
    except OSError as e:
        if getattr(e, "winerror", None) in (10048, 10013):
            logger.info(f"Port {port} is already bound by an active Allsky Web Portal instance ({e}); exiting duplicate process.")
            return
        try:
            server = ExclusiveIPv4ThreadingHTTPServer(("0.0.0.0", port), AllskyPortalHandler)
        except OSError as e4:
            logger.info(f"Port {port} is already bound by an active Allsky Web Portal instance ({e4}); exiting duplicate process.")
            return

    init_db()
    sync_from_history_json()
    threading.Thread(target=_prewarm_caches_background, name="PortalCachePrewarm", daemon=True).start()
    try:
        import feedback_learner
        feedback_learner.start_background_bot_thread()
    except Exception as e:
        logger.warning(f"Failed starting Discord feedback bot thread: {e}")
    try:
        import adsb_tracker
        adsb_tracker.start_background_adsb_thread()
    except Exception as e:
        logger.warning(f"Failed starting ADS-B tracker thread: {e}")
    try:
        import deep_sky_astrophoto
        deep_sky_astrophoto.start_background_auto_scheduler()
    except Exception as e:
        logger.warning(f"Failed starting astrophoto dawn auto-scheduler: {e}")
    logger.info(f"Allsky Anomaly & Meteor Observatory Web Portal listening (Dual-Stack IPv6+IPv4 HTTP/1.1) on http://localhost:{port}/")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Allsky Anomaly Observatory Web Portal")
    parser.add_argument("--port", type=int, default=8094)
    args = parser.parse_args()
    run_server(args.port)
