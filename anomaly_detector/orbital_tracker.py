"""
Orbital Satellite, Spent Rocket Body & Space Flash Identification Engine
========================================================================
Propagates ~16,500+ NORAD / CelesTrak + McCants Classified TLEs using C-accelerated
SGP4/SDP4 (`PyEphem`) across the exact Allsky camera shutter exposure window `[t_exp_start, t_exp_end]`.

Identifies:
  1. Space Stations (`ISS (ZARYA)`, `CSS (TIANHE)`, `DRAGON`, `SOYUZ`, `PROGRESS`)
  2. Bright Visual Targets & Tumbling Spent Rocket Bodies (`SL-16 R/B`, `SL-8 R/B`, `CZ-4B R/B`, `FALCON 9 R/B`, `HST`)
  3. Classified Military / Reconnaissance Satellites (`USA-xxx`, `Keyhole/KH-11`, `Lacrosse/Onyx`, `NOSS`, `X-37B`, `Yaogan`)
  4. Active Constellations & Payloads (`STARLINK`, `ONEWEB`, `IRIDIUM`, `BLUEWALKER 3`, `BLUEBIRD`, `RADARSAT`, `NOAA`, `COSMOS`, `GPS`, `GALILEO`)

Supports Two Matching Modes:
  - Mode 1 (Linear Streaks / Flares): Matches position, trajectory angle (`d_ang <= 14°`), and orbital velocity limit
    to identify satellite passes/flares while keeping true high-speed meteors separate.
  - Mode 2 (Optical Point Flashes / Space Glints): Matches sub-second flashes (`transient_flash`) against sunlit
    objects sweeping directly through `(cx, cy)` during `[t_exp_start, t_exp_end]`, prioritizing known tumblers (`R/B`),
    visual targets, and classified payloads.
"""

import io
import json
import logging
import math
import os
import re
import threading
import time
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import ephem
import numpy as np
import requests

import adsb_tracker

logger = logging.getLogger("allsky_orbital")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TLE_CACHE_DIR = os.path.join(BASE_DIR, "tle_cache")
ORBITAL_STATE_FILE = os.path.join(BASE_DIR, "orbital_live_state.json")

# Refresh cached TLE files at most once every 6 hours (21,600 seconds) to respect CelesTrak rate limits
TLE_CACHE_TTL_SEC = 6 * 3600

TLE_SOURCES = [
    {
        "key": "stations",
        "url": "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle",
        "kind": "tle",
        "priority": 100,
    },
    {
        "key": "classfd",
        "url": "https://mmccants.org/tles/classfd.zip",
        "kind": "zip",
        "priority": 95,
    },
    {
        "key": "visual",
        "url": "https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=tle",
        "kind": "tle",
        "priority": 90,
    },
    {
        "key": "inttles",
        "url": "https://mmccants.org/tles/inttles.zip",
        "kind": "zip",
        "priority": 85,
    },
    {
        "key": "analyst",
        "url": "https://celestrak.org/NORAD/elements/gp.php?GROUP=analyst&FORMAT=tle",
        "kind": "tle",
        "priority": 75,
    },
    {
        "key": "active",
        "url": "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
        "kind": "tle",
        "priority": 60,
    },
]

_LOCK = threading.Lock()
_CATALOG_OBJECTS: List[Dict[str, Any]] = []
_CATALOG_LOADED_AT: float = 0.0
_CATALOG_COUNTS: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# Satellite Category & Human-Friendly Description Classifier
# ---------------------------------------------------------------------------
def classify_orbital_object(raw_name: str, source_key: str) -> Dict[str, Any]:
    """
    Classifies a TLE object name + catalog source into a structured human-readable identity.
    """
    name = re.sub(r"\s+", " ", (raw_name or "").strip()).upper()
    # Strip trailing status markers like '[+]' or '[-]' occasionally present in CelesTrak names
    clean_name = re.sub(r"\s*\[[^\]]*\]\s*$", "", name).strip() or name

    is_rb = bool(
        re.search(r"\b(R/B|ROCKET|PKM|AKM|CENTAUR|BREEZE|FREGAT|AGEN[A]?)\b", clean_name)
    )
    is_deb = bool(re.search(r"\b(DEB|DEBRIS|FRAG)\b", clean_name))
    is_classified = (source_key == "classfd") or bool(
        re.search(r"\b(USA\s+\d+|NOSS|LACROSSE|ONYX|KEYHOLE|KH-\d+|MISTY|TOPAZ|X-37B|OTV|GSSAP|NROL)\b", clean_name)
    )
    is_visual_bright = source_key in ("stations", "visual", "inttles", "classfd")

    if source_key == "stations" or any(
        k in clean_name for k in ("ISS (ZARYA)", "ZARYA", "TIANHE", "TIANGONG", "WENTIAN", "MENGTIAN", "SHENZHOU", "TIANZHOU", "SOYUZ-MS", "PROGRESS-MS", "CREW DRAGON", "POISK", "NAUKA")
    ):
        obj_type = "space_station"
        type_desc = "Crewed Space Station / Visiting Spacecraft"
        priority_boost = 0.55
    elif is_rb:
        obj_type = "rocket_body"
        if "SL-16" in clean_name:
            type_desc = "Spent Zenit-2 Rocket Stage (SL-16 R/B Tumbler)"
        elif "SL-8" in clean_name:
            type_desc = "Spent Kosmos-3M Rocket Stage (SL-8 R/B Tumbler)"
        elif "SL-14" in clean_name:
            type_desc = "Spent Tsyklon-3 Rocket Stage (SL-14 R/B)"
        elif "SL-12" in clean_name:
            type_desc = "Spent Proton Rocket Stage (SL-12 R/B)"
        elif "SL-4" in clean_name or "SL-3" in clean_name:
            type_desc = "Spent Soyuz/Vostok Upper Stage"
        elif "CZ-" in clean_name:
            type_desc = "Spent Long March Rocket Stage (CZ R/B)"
        elif "FALCON" in clean_name:
            type_desc = "Spent SpaceX Falcon 9 Upper Stage"
        elif "ARIANE" in clean_name:
            type_desc = "Spent Ariane Rocket Stage"
        elif "ATLAS" in clean_name or "CENTAUR" in clean_name:
            type_desc = "Spent Atlas/Centaur Upper Stage"
        elif "DELTA" in clean_name:
            type_desc = "Spent Delta Rocket Stage"
        elif "H-2" in clean_name:
            type_desc = "Spent H-IIA/B Rocket Stage"
        else:
            type_desc = "Spent Orbital Rocket Body (Tumbling Booster Stage)"
        priority_boost = 0.62
    elif is_classified:
        obj_type = "classified"
        if "NOSS" in clean_name:
            type_desc = "Classified Naval Ocean Surveillance Satellite (NOSS)"
        elif "USA" in clean_name:
            type_desc = "Classified U.S. Reconnaissance / Defense Satellite"
        elif "YAOGAN" in clean_name:
            type_desc = "Military Reconnaissance Satellite (Yaogan)"
        else:
            type_desc = "Classified Reconnaissance / Defense Space Object"
        priority_boost = 0.65
    elif is_deb:
        obj_type = "debris"
        type_desc = "Orbital Debris Fragment"
        priority_boost = 0.90
    elif "STARLINK" in clean_name:
        obj_type = "starlink"
        type_desc = "SpaceX Starlink LEO Communications Satellite"
        priority_boost = 0.78
    elif "ONEWEB" in clean_name:
        obj_type = "oneweb"
        type_desc = "OneWeb LEO Broadband Satellite"
        priority_boost = 0.80
    elif "IRIDIUM" in clean_name:
        obj_type = "iridium"
        type_desc = "Iridium LEO Communications Satellite"
        priority_boost = 0.72
    elif "BLUEWALKER" in clean_name or "BLUEBIRD" in clean_name:
        obj_type = "giant_array"
        type_desc = "AST SpaceMobile Direct-to-Cell Phased Array"
        priority_boost = 0.60
    elif "HST" in clean_name or "HUBBLE" in clean_name:
        obj_type = "science"
        type_desc = "Hubble Space Telescope (HST)"
        priority_boost = 0.58
    elif "ENVISAT" in clean_name:
        obj_type = "defunct_large"
        type_desc = "Envisat Large Earth Observation Platform"
        priority_boost = 0.62
    elif any(k in clean_name for k in ("GPS", "NAVSTAR", "GLONASS", "GALILEO", "BEIDOU")):
        obj_type = "navigation"
        type_desc = "MEO Global Navigation Satellite (GNSS)"
        priority_boost = 0.82
    elif any(k in clean_name for k in ("NOAA", "METOP", "METEOR-M", "FENGYUN", "GOES", "SUOMI", "JPSS")):
        obj_type = "weather"
        type_desc = "Polar / Geostationary Weather Satellite"
        priority_boost = 0.72
    elif any(k in clean_name for k in ("RADARSAT", "SENTINEL", "LANDSAT", "TERRA", "AQUA", "AURA", "WORLDVIEW", "PLEIADES", "CAPELLA", "ICEYE", "YAOGAN")):
        obj_type = "earth_obs"
        type_desc = "Earth Observation / Radar Imaging Satellite"
        priority_boost = 0.72
    elif "COSMOS" in clean_name or "KOSMOS" in clean_name:
        obj_type = "cosmos"
        type_desc = "Russian Cosmos Military / Scientific Satellite"
        priority_boost = 0.74
    elif source_key == "analyst":
        obj_type = "analyst"
        type_desc = "Analyst-Tracked Uncorrelated Orbital Object"
        priority_boost = 0.82
    else:
        obj_type = "satellite"
        type_desc = "Bright Visual Orbital Satellite" if is_visual_bright else "Active Earth-Orbiting Satellite"
        priority_boost = 0.70 if is_visual_bright else 0.88

    return {
        "clean_name": clean_name,
        "obj_type": obj_type,
        "type_desc": type_desc,
        "is_rocket_body": is_rb,
        "is_classified": is_classified,
        "is_visual_bright": is_visual_bright,
        "priority_boost": priority_boost,
    }


# ---------------------------------------------------------------------------
# TLE Downloader, Disk Cache & SGP4 Catalog Loader
# ---------------------------------------------------------------------------
def _fetch_or_read_cached_tle(source: Dict[str, Any], force_refresh: bool = False) -> str:
    os.makedirs(TLE_CACHE_DIR, exist_ok=True)
    key = source["key"]
    cache_path = os.path.join(TLE_CACHE_DIR, f"{key}.tle")

    now = time.time()
    if not force_refresh and os.path.exists(cache_path):
        try:
            age = now - os.path.getmtime(cache_path)
            if age < TLE_CACHE_TTL_SEC and os.path.getsize(cache_path) > 100:
                with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
        except Exception:
            pass

    headers = {"User-Agent": "BluffSky-Allsky-Observatory/2.0 (Scientific Orbital Tracker)"}
    try:
        resp = requests.get(source["url"], headers=headers, timeout=20)
        if resp.status_code == 200 and len(resp.content) > 60:
            if source["kind"] == "zip":
                zf = zipfile.ZipFile(io.BytesIO(resp.content))
                texts = []
                for name in zf.namelist():
                    texts.append(zf.read(name).decode("utf-8", errors="replace"))
                tle_text = "\n".join(texts)
            else:
                tle_text = resp.text

            if "1 " in tle_text and "2 " in tle_text:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(tle_text)
                logger.info(f"Updated orbital TLE cache '{key}' ({len(tle_text.splitlines())} lines).")
                return tle_text
        elif key == "active" and not os.path.exists(cache_path):
            # Fallback to CelesTrak supplemental feeds if GROUP=active is within its 2-hour cooldown
            sup_urls = [
                "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=starlink&FORMAT=tle",
                "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=oneweb&FORMAT=tle",
                "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=planet&FORMAT=tle",
                "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=iridium&FORMAT=tle",
                "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=gps&FORMAT=tle",
                "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=glonass&FORMAT=tle",
                "https://celestrak.org/NORAD/elements/gp.php?GROUP=geo&FORMAT=tle",
                "https://celestrak.org/NORAD/elements/gp.php?GROUP=military&FORMAT=tle",
                "https://celestrak.org/NORAD/elements/gp.php?GROUP=radar&FORMAT=tle",
            ]
            chunks = []
            for su in sup_urls:
                try:
                    sr = requests.get(su, headers=headers, timeout=15)
                    if sr.status_code == 200 and "1 " in sr.text and "2 " in sr.text:
                        chunks.append(sr.text.strip())
                except Exception:
                    pass
            if chunks:
                tle_text = "\n".join(chunks) + "\n"
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(tle_text)
                return tle_text
    except Exception as e:
        logger.warning(f"Failed fetching TLE feed '{key}' ({source['url']}): {e}")

    # Fallback to existing disk cache if network fails
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            pass
    return ""


def _parse_tle_text(tle_text: str, source_key: str, priority: int) -> Dict[int, Dict[str, Any]]:
    lines = [ln.strip() for ln in tle_text.splitlines() if ln.strip()]
    parsed: Dict[int, Dict[str, Any]] = {}
    i = 0
    n = len(lines)
    while i < n - 1:
        # Locate standard 3-line TLE: Name, Line 1 ('1 NNNNN...'), Line 2 ('2 NNNNN...')
        if i + 2 < n and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 ") and len(lines[i + 1]) >= 60 and len(lines[i + 2]) >= 60:
            name_line = lines[i]
            l1 = lines[i + 1]
            l2 = lines[i + 2]
            i += 3
        elif lines[i].startswith("1 ") and lines[i + 1].startswith("2 ") and len(lines[i]) >= 60 and len(lines[i + 1]) >= 60:
            name_line = f"NORAD {lines[i][2:7].strip()}"
            l1 = lines[i]
            l2 = lines[i + 1]
            i += 2
        else:
            i += 1
            continue

        try:
            norad_str = l2[2:7].strip()
            norad_id = int(re.sub(r"[^0-9]", "", norad_str))
        except Exception:
            continue

        try:
            sat_obj = ephem.readtle(name_line, l1, l2)
        except Exception:
            continue

        try:
            mean_motion = float(l2[52:63].strip())
        except Exception:
            mean_motion = 15.0

        meta = classify_orbital_object(name_line, source_key)
        parsed[norad_id] = {
            "norad_id": norad_id,
            "name": meta["clean_name"],
            "raw_name": name_line,
            "line1": l1,
            "line2": l2,
            "source_key": source_key,
            "priority": priority,
            "mean_motion": mean_motion,
            "sat": sat_obj,
            **meta,
        }
    return parsed


def ensure_catalog_loaded(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Ensures the multi-source orbital catalog (CelesTrak + McCants Classified) is loaded in memory
    and refreshed every 6 hours.
    """
    global _CATALOG_OBJECTS, _CATALOG_LOADED_AT, _CATALOG_COUNTS
    now = time.time()
    if not force_refresh and _CATALOG_OBJECTS and (now - _CATALOG_LOADED_AT) < TLE_CACHE_TTL_SEC:
        return _CATALOG_OBJECTS

    with _LOCK:
        if not force_refresh and _CATALOG_OBJECTS and (now - _CATALOG_LOADED_AT) < TLE_CACHE_TTL_SEC:
            return _CATALOG_OBJECTS

        merged: Dict[int, Dict[str, Any]] = {}
        counts: Dict[str, int] = {}

        # Load in priority order (stations -> classfd -> visual -> inttles -> analyst -> active)
        for src in sorted(TLE_SOURCES, key=lambda s: s["priority"], reverse=True):
            txt = _fetch_or_read_cached_tle(src, force_refresh=force_refresh)
            if not txt:
                counts[src["key"]] = 0
                continue
            group_objs = _parse_tle_text(txt, src["key"], src["priority"])
            counts[src["key"]] = len(group_objs)
            for nid, obj in group_objs.items():
                if nid not in merged:
                    merged[nid] = obj
                else:
                    # Keep the higher-priority classification & bright visual flag,
                    # but if the new TLE is from 'active' and has a more descriptive name than an abbreviated McCants name, preserve both
                    existing = merged[nid]
                    if existing["source_key"] in ("visual", "inttles") and len(obj["name"]) > len(existing["name"]):
                        existing["name"] = obj["name"]

        _CATALOG_OBJECTS = list(merged.values())
        _CATALOG_LOADED_AT = now
        _CATALOG_COUNTS = counts
        logger.info(
            f"Loaded {len(_CATALOG_OBJECTS):,} unique orbital objects into SGP4 tracker "
            f"(stations={counts.get('stations', 0)}, classfd={counts.get('classfd', 0)}, "
            f"visual={counts.get('visual', 0)}, active={counts.get('active', 0)})."
        )
        return _CATALOG_OBJECTS


def _make_observer(lat_deg: float, lon_deg: float, alt_ft: float, unix_ts: float) -> ephem.Observer:
    obs = ephem.Observer()
    obs.lat = str(lat_deg)
    obs.lon = str(lon_deg)
    obs.elevation = float(alt_ft) * 0.3048
    # Disable horizon refraction so geometric topocentric Alt/Az matches our calibrated fisheye model
    obs.pressure = 0
    dt_utc = datetime.fromtimestamp(float(unix_ts), tz=timezone.utc)
    obs.date = ephem.Date(dt_utc.strftime("%Y/%m/%d %H:%M:%S.%f"))
    return obs


def _orbital_speed_kms(alt_km: float) -> float:
    """Computes circular orbital speed in km/s at altitude `alt_km` above Earth's surface."""
    mu_earth = 398600.4418  # km^3 / s^2
    r_km = max(6500.0, 6371.0 + float(alt_km))
    return round(math.sqrt(mu_earth / r_km), 2)


# ---------------------------------------------------------------------------
# Polyline Distance & Tangent Helper Across Multi-Point Orbital Track
# ---------------------------------------------------------------------------
def _point_to_polyline_px(
    px: float,
    py: float,
    pts_px: List[Tuple[float, float]],
) -> Tuple[float, float, float, float, float, float]:
    """
    Finds the closest point on a multi-segment polyline `pts_px` to `(px, py)`.
    Returns `(min_dist_px, proj_x, proj_y, seg_dx, seg_dy, frac_along_track)`.
    """
    best_dist = 999999.0
    best_x, best_y = pts_px[0]
    best_dx, best_dy = 1.0, 0.0
    best_frac = 0.0
    n_seg = max(1, len(pts_px) - 1)

    for i in range(len(pts_px) - 1):
        x1, y1 = pts_px[i]
        x2, y2 = pts_px[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-4:
            t = 0.0
            proj_x, proj_y = x1, y1
        else:
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
        d = math.hypot(px - proj_x, py - proj_y)
        if d < best_dist:
            best_dist = d
            best_x, best_y = proj_x, proj_y
            best_dx, best_dy = dx, dy
            best_frac = (i + t) / float(n_seg)

    return best_dist, best_x, best_y, best_dx, best_dy, best_frac


# ---------------------------------------------------------------------------
# Core SGP4 Orbital Matcher for Both Streaks (`space_object`) and Flashes (`transient_flash`)
# ---------------------------------------------------------------------------
def match_detection_to_satellite(
    candidate: Dict[str, Any],
    cfg: Optional[dict] = None,
    exposure_sec: float = 20.0,
    frame_timestamp: Optional[float] = None,
    exposure_start_ts: Optional[float] = None,
    exposure_end_ts: Optional[float] = None,
    shape: Tuple[int, int] = (3552, 3552),
) -> Optional[Dict[str, Any]]:
    """
    Propagates all cataloged orbital objects across `[exposure_start_ts, exposure_end_ts]`
    and identifies the specific sunlit satellite, rocket body, or classified object
    responsible for either:
      - Mode 1: An orbital streak or partial-pass flare (`cand_len >= 20 px`)
      - Mode 2: A high-contrast optical flash / tumbling specular glint (`cand_len < 20 px`)
    """
    if cfg is None:
        cfg = adsb_tracker.load_config()
    if not cfg.get("orbital_tracking_enabled", True):
        return None

    catalog = ensure_catalog_loaded()
    if not catalog:
        return None

    t_end = float(exposure_end_ts or frame_timestamp or time.time())
    exp_dur = max(0.1, float(exposure_sec))
    t_start = float(exposure_start_ts) if exposure_start_ts is not None else (t_end - exp_dur)
    t_mid = 0.5 * (t_start + t_end)

    obs_lat = float(cfg.get("camera_lat", adsb_tracker.DEFAULT_LAT))
    obs_lon = float(cfg.get("camera_lon", adsb_tracker.DEFAULT_LON))
    obs_alt_ft = float(cfg.get("camera_alt_ft", adsb_tracker.DEFAULT_ALT_FT))

    cx, cy = candidate.get("center", (0.0, 0.0))
    if float(cx) > 0.0 and float(cy) > 0.0:
        cand_alt, cand_az, _ = adsb_tracker.pixel_to_alt_az(float(cx), float(cy), cfg, shape)
    else:
        cand_alt = float(candidate.get("alt_deg") or 0.0)
        cand_az = float(candidate.get("az_deg") or 0.0)
    cand_len = float(candidate.get("length_px") or 0.0)
    cand_lin = float(candidate.get("linearity") or 0.0)
    cand_ang = candidate.get("angle_deg")
    cand_cat = str(candidate.get("category") or "")

    is_linear_streak = (cand_len >= 20.0)
    # A linear streak must have a resolved track angle to verify orbital trajectory alignment
    if is_linear_streak and cand_ang is None:
        return None

    # Stage 1: Fast Mid-Exposure Angular Proximity Pre-Filter across ~13,200+ objects (~30 ms)
    obs_mid = _make_observer(obs_lat, obs_lon, obs_alt_ft, t_mid)
    shortlist: List[Dict[str, Any]] = []

    for obj in catalog:
        sat = obj["sat"]
        try:
            sat.compute(obs_mid)
        except Exception:
            continue

        el_mid = math.degrees(float(sat.alt))
        if el_mid < 2.5:
            continue

        az_mid = math.degrees(float(sat.az)) % 360.0
        ang_sep_mid = adsb_tracker.angular_separation_deg(cand_alt, cand_az, el_mid, az_mid)
        # Low-orbit satellite at 300 km moves at most ~18 deg from t_mid during a 35s exposure (+ along-track TLE margin)
        if ang_sep_mid <= 26.0:
            shortlist.append(obj)

    if not shortlist:
        return None

    # Prepare 5-point sample observers across [t_start, t_end] + extended ±22s along-track observers for TLE epoch drift
    n_steps = 5
    sample_ts = [t_start + (t_end - t_start) * (i / float(n_steps - 1)) for i in range(n_steps)]
    sample_obs = [_make_observer(obs_lat, obs_lon, obs_alt_ft, ts) for ts in sample_ts]
    ext_pre_obs = [_make_observer(obs_lat, obs_lon, obs_alt_ft, ts) for ts in (t_start - 22.0, t_start - 11.0)]
    ext_post_obs = [_make_observer(obs_lat, obs_lon, obs_alt_ft, ts) for ts in (t_end + 11.0, t_end + 22.0)]

    best_match: Optional[Dict[str, Any]] = None
    best_weighted_score = 999999.0

    scale_factor = shape[0] / 3552.0

    for obj in shortlist:
        sat = obj["sat"]
        pts_px: List[Tuple[float, float]] = []
        pts_altaz: List[Tuple[float, float]] = []
        eclipsed_all = True
        alt_km_mid = 550.0
        range_km_mid = 800.0

        valid_track = True
        for idx, obs in enumerate(sample_obs):
            try:
                sat.compute(obs)
            except Exception:
                valid_track = False
                break
            if not bool(sat.eclipsed):
                eclipsed_all = False
            el_d = math.degrees(float(sat.alt))
            az_d = math.degrees(float(sat.az)) % 360.0
            px, py = adsb_tracker.altaz_to_pixel(el_d, az_d, cfg, shape)
            pts_px.append((float(px), float(py)))
            pts_altaz.append((el_d, az_d))
            if idx == n_steps // 2:
                alt_km_mid = round(float(sat.elevation) / 1000.0, 1)
                range_km_mid = round(float(sat.range) / 1000.0, 1)

        if not valid_track or max(p[0] for p in pts_altaz) < 3.0:
            continue

        # Build extended ±22s along-track rail to catch active LEO satellites with slight TLE epoch drift
        pts_px_pre: List[Tuple[float, float]] = []
        pts_px_post: List[Tuple[float, float]] = []
        for obs_e in ext_pre_obs:
            try:
                sat.compute(obs_e)
                if not bool(sat.eclipsed):
                    eclipsed_all = False
                px_e, py_e = adsb_tracker.altaz_to_pixel(math.degrees(float(sat.alt)), math.degrees(float(sat.az)) % 360.0, cfg, shape)
                pts_px_pre.append((float(px_e), float(py_e)))
            except Exception:
                pass
        for obs_e in ext_post_obs:
            try:
                sat.compute(obs_e)
                if not bool(sat.eclipsed):
                    eclipsed_all = False
                px_e, py_e = adsb_tracker.altaz_to_pixel(math.degrees(float(sat.alt)), math.degrees(float(sat.az)) % 360.0, cfg, shape)
                pts_px_post.append((float(px_e), float(py_e)))
            except Exception:
                pass
        pts_px_ext = pts_px_pre + pts_px + pts_px_post

        # Compute closest approach of candidate center (cx, cy) to the 5-point orbital polyline
        dist_px, proj_x, proj_y, seg_dx, seg_dy, frac_t = _point_to_polyline_px(float(cx), float(cy), pts_px)
        overall_dx = pts_px[-1][0] - pts_px[0][0]
        overall_dy = pts_px[-1][1] - pts_px[0][1]
        sat_chord_len_px = math.hypot(overall_dx, overall_dy)

        # Interpolate alt/az at the closest point along the track
        seg_idx = min(n_steps - 2, max(0, int(frac_t * (n_steps - 1))))
        local_w = (frac_t * (n_steps - 1)) - seg_idx
        el_closest = pts_altaz[seg_idx][0] * (1.0 - local_w) + pts_altaz[seg_idx + 1][0] * local_w
        az_closest = pts_altaz[seg_idx][1]
        ang_sep = adsb_tracker.angular_separation_deg(cand_alt, cand_az, el_closest, az_closest)

        # Tangent direction at closest point (fallback to overall chord)
        tdx = seg_dx if math.hypot(seg_dx, seg_dy) >= 2.0 else overall_dx
        tdy = seg_dy if math.hypot(seg_dx, seg_dy) >= 2.0 else overall_dy
        sat_ang_180 = math.degrees(math.atan2(tdy, tdx)) % 180.0

        is_sunlit = not eclipsed_all
        d_ang = 0.0
        matched_as_streak = False

        if is_linear_streak and cand_ang is not None:
            # Mode 1: Orbital Streak / Flare Matcher
            d_ang = abs(float(cand_ang) - sat_ang_180) % 180.0
            d_ang = min(d_ang, 180.0 - d_ang)
            max_d_ang = 14.0 if cand_len >= 40.0 else 16.0
            if d_ang <= max_d_ang:
                s_dist_px = dist_px
                p1_opt = candidate.get("p1")
                p2_opt = candidate.get("p2")
                if p1_opt and p2_opt:
                    d1, _, _, _, _, _ = _point_to_polyline_px(float(p1_opt[0]), float(p1_opt[1]), pts_px)
                    d2, _, _, _, _, _ = _point_to_polyline_px(float(p2_opt[0]), float(p2_opt[1]), pts_px)
                    s_dist_px = max(s_dist_px, 0.5 * (d1 + d2))

                # Also check extended ±22s along-track rail for long, steady satellite streaks with slight TLE epoch drift
                cand_snr = float(candidate.get("snr") or 0.0)
                if is_sunlit and cand_len >= 55.0 * scale_factor and cand_snr < 35.0 and d_ang <= 12.0 and len(pts_px_ext) > len(pts_px):
                    d_ext_c, px_e, py_e, edx, edy, _ = _point_to_polyline_px(float(cx), float(cy), pts_px_ext)
                    s_dist_ext = d_ext_c
                    if p1_opt and p2_opt:
                        de1, _, _, _, _, _ = _point_to_polyline_px(float(p1_opt[0]), float(p1_opt[1]), pts_px_ext)
                        de2, _, _, _, _, _ = _point_to_polyline_px(float(p2_opt[0]), float(p2_opt[1]), pts_px_ext)
                        s_dist_ext = max(s_dist_ext, 0.5 * (de1 + de2))
                    if s_dist_ext <= 22.0 * scale_factor and (s_dist_ext + 6.0 * scale_factor) < s_dist_px:
                        s_dist_px = s_dist_ext + 6.0 * scale_factor
                        proj_x, proj_y = px_e, py_e
                        if math.hypot(edx, edy) >= 2.0:
                            tdx, tdy = edx, edy

                # Prevent short, high-SNR spindle meteors from stealing a satellite label
                is_short_bright_meteor = (
                    (cand_len < 38.0 * scale_factor or cand_len < 0.50 * sat_chord_len_px)
                    and cand_snr >= 35.0
                )
                if not is_short_bright_meteor and cand_len <= (sat_chord_len_px * 1.55 + 55.0 * scale_factor):
                    max_px_thresh = (42.0 if is_sunlit else 24.0) * scale_factor
                    if s_dist_px <= max_px_thresh or ang_sep <= (1.75 if is_sunlit else 1.1):
                        matched_as_streak = True
                        dist_px = s_dist_px
                        raw_score = dist_px + d_ang * 2.5

        if not matched_as_streak:
            # Mode 2: Optical Point Flash / Deep-Space or Low-Orbit Glint Matcher
            # Applies ONLY to compact stellar-PSF flashes (cand_len < 22px, aspect <= 2.4),
            # never to high-aspect shooting star / meteor streaks or diffuse cloud puffs!
            cand_aspect = float(candidate.get("aspect") or 1.0)
            cand_width = float(candidate.get("width_px") or 0.0)
            if (
                cand_len > 24.0
                or cand_aspect > 2.5
                or cand_width > 9.0
                or cand_cat in ("meteor", "fireball", "aircraft")
            ):
                continue
            max_px_thresh = (28.0 if is_sunlit else 16.0) * scale_factor
            if dist_px > max_px_thresh and ang_sep > (1.15 if is_sunlit else 0.7):
                continue
            raw_score = dist_px

        # Always prefer any valid sunlit match over any shadowed/penumbral match (penalty +10,000 for non-sunlit)
        weighted_score = raw_score * float(obj.get("priority_boost", 0.85)) + (0.0 if is_sunlit else 10000.0)
        if weighted_score < best_weighted_score:
            best_weighted_score = weighted_score
            speed_kms = _orbital_speed_kms(alt_km_mid)
            speed_mph = int(round(speed_kms * 2236.936))
            alt_miles = round(alt_km_mid * 0.621371, 1)
            range_miles = round(range_km_mid * 0.621371, 1)

            # Classify orbital altitude regime (LEO vs MEO vs HEO/GEO Deep Space)
            if alt_km_mid >= 33000.0:
                orbit_regime = "GEO Deep-Space"
                orbit_short = "GEO"
            elif alt_km_mid >= 15000.0:
                orbit_regime = "Deep-Space MEO/HEO"
                orbit_short = "DEEP-SPACE"
            elif alt_km_mid >= 2000.0:
                orbit_regime = "Mid-Earth Orbit (MEO)"
                orbit_short = "MEO"
            else:
                orbit_regime = "Low-Earth Orbit (LEO)"
                orbit_short = "LEO"

            # Determine arrow start/end coordinates:
            # - If it's a resolved linear streak, align arrow along the optical streak (p1 -> p2) in orbital direction
            # - If it's a point flash, place the directional trajectory vector OUTSIDE the flash box so the flash itself is never covered!
            p1_opt = candidate.get("p1")
            p2_opt = candidate.get("p2")
            if is_linear_streak and p1_opt and p2_opt and cand_len >= 22.0:
                opt_dx = float(p2_opt[0]) - float(p1_opt[0])
                opt_dy = float(p2_opt[1]) - float(p1_opt[1])
                if (opt_dx * tdx + opt_dy * tdy) >= 0.0:
                    draw_start = [int(round(p1_opt[0])), int(round(p1_opt[1]))]
                    draw_end = [int(round(p2_opt[0])), int(round(p2_opt[1]))]
                else:
                    draw_start = [int(round(p2_opt[0])), int(round(p2_opt[1]))]
                    draw_end = [int(round(p1_opt[0])), int(round(p1_opt[1]))]
            else:
                # Point flash or short glint: start the arrow outside the flash box on the leading side so the flash dot inside the box is 100% clear!
                mag = max(1e-3, math.hypot(tdx, tdy))
                ux, uy = tdx / mag, tdy / mag
                draw_start = [int(round(float(cx) + ux * 36.0 * scale_factor)), int(round(float(cy) + uy * 36.0 * scale_factor))]
                vec_len = max(55.0 * scale_factor, min(140.0 * scale_factor, sat_chord_len_px * 0.65))
                draw_end = [int(round(float(cx) + ux * (36.0 * scale_factor + vec_len))), int(round(float(cy) + uy * (36.0 * scale_factor + vec_len)))]

            sat_name = obj["name"]
            norad_id = obj["norad_id"]
            type_desc = obj["type_desc"]

            if is_linear_streak:
                subject_line = f"🛰️ {sat_name} [NORAD #{norad_id}] ({orbit_short} {alt_km_mid:,.0f} km / {alt_miles:,.0f} mi)"
                box_header = f"SAT: {sat_name} [#{norad_id}] ({orbit_short} {alt_km_mid:,.0f}km)"
                desc_line = (
                    f"Identified {orbit_regime} pass/flare of {sat_name} (NORAD #{norad_id} — {type_desc}) "
                    f"at {alt_km_mid:,.0f} km ({alt_miles:,.0f} mi) altitude traveling at "
                    f"{speed_kms:.2f} km/s ({speed_mph:,} mph), slant range {range_km_mid:,.0f} km ({range_miles:,.0f} mi)."
                )
            else:
                flash_kind = "Deep-Space Flash" if alt_km_mid >= 2500.0 else "Orbital Glint"
                subject_line = f"🛰️ {flash_kind}: {sat_name} [NORAD #{norad_id}] ({orbit_short} {alt_km_mid:,.0f} km)"
                box_header = f"{orbit_short} FLASH: {sat_name} [#{norad_id}] ({alt_km_mid:,.0f}km)"
                desc_line = (
                    f"Identified {orbit_regime} specular glint / flash from {sat_name} "
                    f"(NORAD #{norad_id} — {type_desc}) at {alt_km_mid:,.0f} km ({alt_miles:,.0f} mi) altitude "
                    f"({speed_kms:.2f} km/s / {speed_mph:,} mph, slant range {range_km_mid:,.0f} km / {range_miles:,.0f} mi)."
                )

            best_match = {
                "norad_id": norad_id,
                "name": sat_name,
                "obj_type": obj["obj_type"],
                "type_desc": type_desc,
                "source_key": obj["source_key"],
                "is_rocket_body": obj["is_rocket_body"],
                "is_classified": obj["is_classified"],
                "is_visual_bright": obj["is_visual_bright"],
                "is_sunlit": is_sunlit,
                "orbit_regime": orbit_regime,
                "orbit_short": orbit_short,
                "orbit_alt_km": alt_km_mid,
                "orbit_alt_miles": alt_miles,
                "slant_range_km": range_km_mid,
                "slant_range_miles": range_miles,
                "speed_kms": speed_kms,
                "speed_mph": speed_mph,
                "match_mode": "streak" if is_linear_streak else "flash",
                "match_dist_px": round(dist_px, 1),
                "match_ang_deg": round(ang_sep, 2),
                "track_angle_diff_deg": round(d_ang, 1),
                "seg_start_px": draw_start,
                "seg_end_px": draw_end,
                "orbit_seg_start_px": [int(round(pts_px[0][0])), int(round(pts_px[0][1]))],
                "orbit_seg_end_px": [int(round(pts_px[-1][0])), int(round(pts_px[-1][1]))],
                "matched_proj_px": [int(round(proj_x)), int(round(proj_y))],
                "flash_fraction_along_exposure": round(frac_t, 2),
                "box_header": box_header,
                "subject_line": subject_line,
                "description_line": desc_line,
            }

    return best_match


_LIVE_ORBITAL_CACHE: Dict[str, Any] = {"key": None, "ts": 0.0, "data": []}


def get_live_orbital_snapshot(
    cfg: Optional[dict] = None,
    exp_end_ts: Optional[float] = None,
    exposure_sec: float = 30.0,
    include_starlink: bool = False,
    img_shape: Tuple[int, int] = (3552, 3552),
) -> List[Dict[str, Any]]:
    """
    Computes all sunlit orbital satellites, space stations, rocket bodies, and
    classified reconnaissance objects currently in the Allsky camera dome over
    the exposure window [exp_end_ts - exposure_sec, exp_end_ts].
    """
    if ephem is None:
        return []
    if cfg is None:
        cfg = adsb_tracker.load_config()
    if exp_end_ts is None:
        exp_end_ts = time.time()

    exp_dur = max(5.0, min(120.0, float(exposure_sec or 30.0)))
    cache_key = (int(exp_end_ts // 8), bool(include_starlink), img_shape[0], img_shape[1])
    now_mono = time.time()
    if _LIVE_ORBITAL_CACHE.get("key") == cache_key and (now_mono - _LIVE_ORBITAL_CACHE.get("ts", 0.0)) < 8.0:
        return _LIVE_ORBITAL_CACHE["data"]

    catalog = ensure_catalog_loaded(force_refresh=False)
    if not catalog:
        return []

    cam_lat = float(cfg.get("camera_lat", adsb_tracker.DEFAULT_LAT))
    cam_lon = float(cfg.get("camera_lon", adsb_tracker.DEFAULT_LON))
    cam_alt_ft = float(cfg.get("camera_alt_ft", adsb_tracker.DEFAULT_ALT_FT))

    t_start = exp_end_ts - exp_dur
    t_mid = exp_end_ts - 0.5 * exp_dur
    t_end = exp_end_ts

    obs_start = _make_observer(cam_lat, cam_lon, cam_alt_ft, t_start)
    obs_mid = _make_observer(cam_lat, cam_lon, cam_alt_ft, t_mid)
    obs_end = _make_observer(cam_lat, cam_lon, cam_alt_ft, t_end)

    h, w = img_shape[:2]
    leo_meo_results: List[Dict[str, Any]] = []
    geo_results: List[Dict[str, Any]] = []
    constellation_sats: List[Dict[str, Any]] = []

    for obj in catalog:
        norad_id = obj["norad_id"]
        body = obj["sat"]
        try:
            body.compute(obs_mid)
        except Exception:
            continue

        if bool(getattr(body, "eclipsed", False)):
            continue

        alt_mid_deg = math.degrees(float(body.alt))
        alt_km = round(float(getattr(body, "elevation", 550000.0)) / 1000.0, 1)
        is_leo = alt_km <= 2500.0
        is_priority = (
            obj["is_visual_bright"]
            or obj["is_classified"]
            or obj["is_rocket_body"]
            or obj["source_key"] in ("stations", "visual", "classfd", "inttles", "analyst", "military", "radar")
        )
        min_el = 10.0 if (is_priority or is_leo) else 18.0
        if alt_mid_deg < min_el:
            continue

        az_mid_deg = math.degrees(float(body.az)) % 360.0
        px_mid, py_mid = adsb_tracker.altaz_to_pixel(alt_mid_deg, az_mid_deg, cfg, (h, w))
        if not (20 <= px_mid < w - 20 and 20 <= py_mid < h - 20):
            continue

        try:
            body.compute(obs_start)
            alt_s_deg = math.degrees(float(body.alt))
            az_s_deg = math.degrees(float(body.az)) % 360.0
            px_s, py_s = adsb_tracker.altaz_to_pixel(alt_s_deg, az_s_deg, cfg, (h, w))

            body.compute(obs_end)
            alt_e_deg = math.degrees(float(body.alt))
            az_e_deg = math.degrees(float(body.az)) % 360.0
            px_e, py_e = adsb_tracker.altaz_to_pixel(alt_e_deg, az_e_deg, cfg, (h, w))

            body.compute(obs_mid)
        except Exception:
            continue

        range_km = round(float(getattr(body, "range", 800000.0)) / 1000.0, 1)
        speed_kms = _orbital_speed_kms(alt_km)
        speed_mph = int(round(speed_kms * 2236.936))

        entry = {
            "norad_id": norad_id,
            "name": obj["name"],
            "obj_type": obj["obj_type"],
            "type_desc": obj["type_desc"],
            "source_key": obj["source_key"],
            "is_rocket_body": obj["is_rocket_body"],
            "is_classified": obj["is_classified"],
            "is_visual_bright": obj["is_visual_bright"],
            "is_priority": is_priority,
            "alt_deg": round(alt_mid_deg, 1),
            "az_deg": round(az_mid_deg, 1),
            "orbit_alt_km": alt_km,
            "orbit_alt_miles": round(alt_km * 0.621371, 1),
            "slant_range_km": range_km,
            "speed_kms": speed_kms,
            "speed_mph": speed_mph,
            "px_start": int(round(px_s)),
            "py_start": int(round(py_s)),
            "px_mid": int(round(px_mid)),
            "py_mid": int(round(py_mid)),
            "px_end": int(round(px_e)),
            "py_end": int(round(py_e)),
        }

        if alt_km > 22500.0:
            if is_priority:
                geo_results.append(entry)
        elif (
            obj["source_key"] in ("starlink", "oneweb", "planet")
            or "STARLINK" in obj["name"].upper()
            or "ONEWEB" in obj["name"].upper()
        ) and not is_priority:
            constellation_sats.append(entry)
        else:
            leo_meo_results.append(entry)

    # Sort LEO/MEO objects: LEO (<=2500km) first, then visual/classified/R/B, then elevation
    leo_meo_results.sort(
        key=lambda x: (
            1 if x["source_key"] == "stations" else 0,
            1 if x["orbit_alt_km"] <= 2500.0 else 0,
            1 if x["is_visual_bright"] else 0,
            1 if x["is_classified"] else 0,
            1 if x["is_rocket_body"] else 0,
            x["alt_deg"],
        ),
        reverse=True,
    )
    results = leo_meo_results[:22]

    # If few non-constellation LEO objects are sunlit (e.g., around solar midnight), include top sunlit LEO constellation passes
    constellation_sats.sort(key=lambda x: (1 if x["orbit_alt_km"] <= 2000.0 else 0, x["alt_deg"]), reverse=True)
    if include_starlink:
        results.extend(constellation_sats[:30])
    elif sum(1 for r in results if r["orbit_alt_km"] <= 2500.0) < 6:
        results.extend(constellation_sats[:6])

    # Include up to 4 high-altitude HEO/GEO classified objects without crowding the southern belt
    geo_results.sort(key=lambda x: (1 if x["orbit_alt_km"] < 34000.0 else 0, x["alt_deg"]), reverse=True)
    results.extend(geo_results[:4])

    _LIVE_ORBITAL_CACHE["key"] = cache_key
    _LIVE_ORBITAL_CACHE["ts"] = now_mono
    _LIVE_ORBITAL_CACHE["data"] = results
    return results


def draw_orbital_overlay_on_image(
    img: np.ndarray,
    cfg: Optional[dict] = None,
    exp_end_ts: Optional[float] = None,
    exposure_sec: float = 30.0,
    include_starlink: bool = False,
) -> np.ndarray:
    """
    Draws sunlit orbital satellites, space stations, rocket bodies, and classified
    satellites with their exposure-window trajectory vectors and NORAD telemetry pills.
    """
    import cv2

    if cfg is None:
        cfg = adsb_tracker.load_config()
    h, w = img.shape[:2]
    sats = get_live_orbital_snapshot(
        cfg=cfg,
        exp_end_ts=exp_end_ts,
        exposure_sec=exposure_sec,
        include_starlink=include_starlink,
        img_shape=(h, w),
    )
    if not sats:
        return img

    out = img.copy()
    scale = max(0.45, w / 2200.0)
    thick = max(2, int(round(scale * 2.2)))
    placed_boxes: List[Tuple[int, int, int, int]] = []

    for sat in sats:
        px_s, py_s = sat["px_start"], sat["py_start"]
        px_m, py_m = sat["px_mid"], sat["py_mid"]
        px_e, py_e = sat["px_end"], sat["py_end"]

        if sat["source_key"] == "stations" or sat["is_visual_bright"]:
            color = (255, 235, 40)   # Bright Cyan-Aqua
        elif sat["is_classified"]:
            color = (235, 115, 255)  # Magenta-Violet for classified recon
        elif sat["is_rocket_body"]:
            color = (70, 165, 255)   # Warm Orange for Rocket Bodies
        else:
            color = (255, 195, 55)   # Sky Blue-Cyan for standard payloads

        # Draw a clear target box around the satellite so any streak/glint inside the box is unobstructed
        box_r = int(16 * scale)
        cv2.rectangle(out, (px_m - box_r - 1, py_m - box_r - 1), (px_m + box_r + 1, py_m + box_r + 1), (12, 16, 26), thick + 1, cv2.LINE_AA)
        cv2.rectangle(out, (px_m - box_r, py_m - box_r), (px_m + box_r, py_m + box_r), color, thick, cv2.LINE_AA)

        # Draw orbital motion vector starting outside the box toward the leading edge
        chord = math.hypot(px_e - px_s, py_e - py_s)
        if chord >= 8 * scale:
            ux = (px_e - px_s) / chord
            uy = (py_e - py_s) / chord
            ax1 = int(round(px_m + ux * (box_r + 3 * scale)))
            ay1 = int(round(py_m + uy * (box_r + 3 * scale)))
            arrow_len = max(18.0 * scale, min(55.0 * scale, chord * 0.55))
            ax2 = int(round(ax1 + ux * arrow_len))
            ay2 = int(round(ay1 + uy * arrow_len))
            cv2.arrowedLine(out, (ax1, ay1), (ax2, ay2), color, thick, cv2.LINE_AA, tipLength=0.32)

        alt_km_v = sat["orbit_alt_km"]
        reg_lbl = "GEO" if alt_km_v >= 33000 else ("MEO" if alt_km_v >= 2000 else "LEO")
        if sat["is_classified"]:
            reg_lbl = f"RECON {reg_lbl}"
        elif sat["is_rocket_body"]:
            reg_lbl = f"R/B {reg_lbl}"
        label = f"SAT: {sat['name']} (#{sat['norad_id']}) | {reg_lbl} {int(round(alt_km_v)):,}km | {int(sat['speed_mph']):,}mph"
        font_scale = 0.55 * scale
        font_thick = max(1, int(round(scale * 1.8)))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

        # Candidate label anchor positions with collision avoidance
        lx = max(12, min(w - tw - 18, px_m + box_r + int(10 * scale)))
        ly = max(th + 14, min(h - 14, py_m - int(6 * scale)))
        for dy_try in (0, int(26 * scale), -int(26 * scale), int(52 * scale), -int(52 * scale)):
            cand_y = max(th + 14, min(h - 14, ly + dy_try))
            box = (lx - 6, cand_y - th - 6, lx + tw + 6, cand_y + 6)
            if not any(
                not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3])
                for b in placed_boxes
            ):
                ly = cand_y
                placed_boxes.append(box)
                break

        cv2.rectangle(out, (lx - 6, ly - th - 6), (lx + tw + 6, ly + 6), (10, 15, 28), -1)
        cv2.rectangle(out, (lx - 6, ly - th - 6), (lx + tw + 6, ly + 6), color, max(1, font_thick - 1))
        cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick, cv2.LINE_AA)

    return out

