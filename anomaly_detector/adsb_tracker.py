from datetime import datetime, timezone
import json
import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
ADSB_STATE_FILE = os.path.join(BASE_DIR, "adsb_live_state.json")
CALIBRATION_FILE = os.path.join(BASE_DIR, "camera_calibration.json")

logger = logging.getLogger("AllskyADSBTracker")

# Exact house / camera / antenna coordinates provided by user
DEFAULT_LAT = 45.0000
DEFAULT_LON = -100.0000
DEFAULT_ALT_FT = 130.0

CARDINALS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]

# Common ICAO Aircraft Type Designators -> Human-readable aircraft names
ICAO_TYPE_NAMES: Dict[str, str] = {
    "C150": "Cessna 150",
    "C152": "Cessna 152",
    "C172": "Cessna 172 Skyhawk",
    "C182": "Cessna 182 Skylane",
    "C206": "Cessna 206 Stationair",
    "C208": "Cessna 208 Caravan",
    "C25A": "Cessna Citation CJ2",
    "C25B": "Cessna Citation CJ3",
    "C56X": "Cessna Citation Excel",
    "PA28": "Piper PA-28 Cherokee",
    "PA31": "Piper PA-31 Navajo",
    "PA32": "Piper PA-32 Saratoga",
    "PA34": "Piper PA-34 Seneca",
    "PA46": "Piper PA-46 Malibu",
    "SR20": "Cirrus SR20",
    "SR22": "Cirrus SR22",
    "PC12": "Pilatus PC-12",
    "PC24": "Pilatus PC-24",
    "BE20": "Beechcraft King Air 200",
    "BE30": "Beechcraft King Air 300",
    "BE35": "Beechcraft Bonanza",
    "BE36": "Beechcraft Bonanza 36",
    "BE58": "Beechcraft Baron 58",
    "BE9L": "Beechcraft King Air C90",
    "DA40": "Diamond DA40",
    "DA42": "Diamond DA42 Twin Star",
    "DA62": "Diamond DA62",
    "DH8A": "De Havilland Dash 8-100",
    "DH8B": "De Havilland Dash 8-200",
    "DH8C": "De Havilland Dash 8-300",
    "DH8D": "De Havilland Dash 8-400 (Q400)",
    "DHC2": "De Havilland DHC-2 Beaver",
    "DHC3": "De Havilland DHC-3 Otter",
    "DHC6": "De Havilland DHC-6 Twin Otter",
    "E170": "Embraer E170",
    "E75L": "Embraer E175",
    "E75S": "Embraer E175",
    "E190": "Embraer E190",
    "CRJ2": "Bombardier CRJ-200",
    "CRJ7": "Bombardier CRJ-700",
    "CRJ9": "Bombardier CRJ-900",
    "BCS1": "Airbus A220-100",
    "BCS3": "Airbus A220-300",
    "A319": "Airbus A319",
    "A320": "Airbus A320",
    "A20N": "Airbus A320neo",
    "A321": "Airbus A321",
    "A21N": "Airbus A321neo",
    "A332": "Airbus A330-200",
    "A333": "Airbus A330-300",
    "A339": "Airbus A330-900neo",
    "A359": "Airbus A350-900",
    "A35K": "Airbus A350-1000",
    "B737": "Boeing 737-700",
    "B738": "Boeing 737-800",
    "B739": "Boeing 737-900",
    "B38M": "Boeing 737 MAX 8",
    "B39M": "Boeing 737 MAX 9",
    "B752": "Boeing 757-200",
    "B753": "Boeing 757-300",
    "B763": "Boeing 767-300",
    "B772": "Boeing 777-200",
    "B77W": "Boeing 777-300ER",
    "B77L": "Boeing 777F",
    "B788": "Boeing 787-8 Dreamliner",
    "B789": "Boeing 787-9 Dreamliner",
    "B78X": "Boeing 787-10 Dreamliner",
    "B744": "Boeing 747-400",
    "B748": "Boeing 747-8",
    "AS65": "USCG MH-65 Dolphin Helicopter",
    "H60": "Sikorsky UH-60 / MH-60 Jayhawk",
    "EC35": "Airbus H135 Helicopter",
    "EC45": "Airbus H145 Helicopter",
    "A109": "AgustaWestland AW109",
    "B06": "Bell 206 JetRanger",
    "B407": "Bell 407 Helicopter",
    "R44": "Robinson R44",
    "R66": "Robinson R66",
    "P8": "Boeing P-8A Poseidon (US Navy)",
    "C17": "Boeing C-17A Globemaster III (USAF)",
    "C27J": "Alenia C-27J Spartan (USCG)",
    "K35R": "Boeing KC-135R Stratotanker",
    "EA18": "Boeing EA-18G Growler (NAS Whidbey)",
    "F18S": "Boeing F/A-18 Super Hornet",
    "B350": "Beechcraft Super King Air 350",
    "GLEX": "Bombardier Global Express",
    "M20P": "Mooney M20",
    "TWEN": "Tecnam P2010 Twenty",
}

# Common 3-letter ICAO airline callsign prefixes in the Pacific Northwest
AIRLINE_PREFIXES: Dict[str, str] = {
    "ASA": "Alaska Airlines",
    "QXE": "Horizon Air",
    "SKW": "SkyWest Airlines",
    "DAL": "Delta Air Lines",
    "UAL": "United Airlines",
    "AAL": "American Airlines",
    "SWA": "Southwest Airlines",
    "ACA": "Air Canada",
    "ROU": "Air Canada Rouge",
    "QK": "Air Canada Express (Jazz)",
    "JZA": "Air Canada Express (Jazz)",
    "WJA": "WestJet",
    "WEN": "WestJet Encore",
    "FLE": "Flair Airlines",
    "PCO": "Pacific Coastal Airlines",
    "FDX": "FedEx Express",
    "UPS": "UPS Airlines",
    "GTI": "Atlas Air",
    "ABX": "ABX Air",
    "ATN": "Air Transport International",
    "NCR": "National Airlines",
    "CKK": "China Cargo Airlines",
    "CJT": "Cargojet Airways",
    "HAL": "Hawaiian Airlines",
    "JBU": "JetBlue",
    "FFT": "Frontier Airlines",
    "NKS": "Spirit Airlines",
    "SCX": "Sun Country Airlines",
    "AMX": "Aeroméxico",
    "PAL": "Philippine Airlines",
    "SJX": "Starlux Airlines",
    "EVA": "EVA Air",
    "CAL": "China Airlines",
    "ANA": "All Nippon Airways",
    "JAL": "Japan Airlines",
    "KAL": "Korean Air",
    "CPA": "Cathay Pacific",
    "SIA": "Singapore Airlines",
    "BAW": "British Airways",
    "DLH": "Lufthansa",
    "AFR": "Air France",
    "KLM": "KLM Royal Dutch Airlines",
    "BOE": "Boeing Flight Test",
    "C27": "US Coast Guard",
    "KEN": "Kenmore Air",
    "HAR": "Harbour Air Seaplanes",
}

# Navigation stars for Astrometric Star-Field Calibration
STAR_CATALOG = [
    ("Polaris", 2.5303, 89.2641, 1.98),
    ("Sirius", 6.7525, -16.7161, -1.46),
    ("Capella", 5.2782, 45.9980, 0.08),
    ("Vega", 18.6156, 38.7837, 0.03),
    ("Arcturus", 14.2610, 19.1822, -0.05),
    ("Rigel", 5.2423, -8.2016, 0.13),
    ("Procyon", 7.6550, 5.2250, 0.34),
    ("Betelgeuse", 5.9195, 7.4071, 0.42),
    ("Altair", 19.8464, 8.8683, 0.76),
    ("Aldebaran", 4.5987, 16.5093, 0.86),
    ("Pollux", 7.7553, 28.0262, 1.14),
    ("Fomalhaut", 22.9608, -29.6222, 1.16),
    ("Deneb", 20.6905, 45.2803, 1.25),
    ("Castor", 7.5767, 31.8883, 1.58),
    ("Pleiades", 3.7914, 24.1051, 1.60),
    ("Bellatrix", 5.4189, 6.3497, 1.64),
    ("Elnath", 5.4382, 28.6075, 1.65),
    ("Alioth", 12.9005, 55.9598, 1.77),
    ("Mirfak", 3.4054, 49.8612, 1.79),
    ("Dubhe", 11.0621, 61.7510, 1.79),
    ("Alkaid", 13.7923, 49.3133, 1.86),
    ("Hamal", 2.1196, 23.4624, 2.00),
    ("Alpheratz", 0.1398, 29.0904, 2.06),
    ("Mirach", 1.1622, 35.6206, 2.07),
    ("Schedar", 0.6751, 56.5373, 2.24),
    ("Caph", 0.1529, 59.1498, 2.28),
    ("Gamma Cas", 0.9451, 60.7167, 2.15),
    ("Ruchbah", 1.4303, 60.2353, 2.68),
    ("Segin", 1.9066, 63.6701, 3.35),
]

_LOCK = threading.Lock()
_TRACKER_THREAD_STARTED = False
# Rolling history of aircraft positions over the last 240 seconds: hex -> list of samples
_AIRCRAFT_HISTORY: Dict[str, List[Dict[str, Any]]] = {}
# Cache of aircraft registration/type enrichment from regional feed: hex -> dict
_ENRICHMENT_CACHE: Dict[str, Dict[str, str]] = {}
_LAST_REGIONAL_FETCH = 0.0
_LATEST_SKY_SNAPSHOT: Dict[str, Any] = {
    "updated_at": 0.0,
    "local_receiver_ok": False,
    "local_messages": 0,
    "aircraft_in_range": [],
    "aircraft_in_dome": [],
}


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# 3D Geodetic <-> Topocentric Sky (Alt, Az) <-> Fisheye Sensor (x, y) Math
# ---------------------------------------------------------------------------
def geodetic_to_altaz(
    lat: float,
    lon: float,
    alt_ft: float,
    obs_lat: float = DEFAULT_LAT,
    obs_lon: float = DEFAULT_LON,
    obs_alt_ft: float = DEFAULT_ALT_FT,
) -> Tuple[float, float, float, float]:
    """
    Converts an aircraft's WGS84 (lat, lon, alt_ft) to topocentric sky coordinates
    relative to the Allsky camera / ADS-B antenna at (obs_lat, obs_lon, obs_alt_ft):
      Returns (elevation_deg, azimuth_deg, slant_range_km, ground_dist_km)
    Includes Earth curvature drop over distance.
    """
    R_earth = 6371000.0
    alt_m = float(alt_ft) * 0.3048
    obs_alt_m = float(obs_alt_ft) * 0.3048

    dlat_rad = math.radians(float(lat) - float(obs_lat))
    dlon_rad = math.radians(float(lon) - float(obs_lon))
    north_m = dlat_rad * R_earth
    east_m = dlon_rad * R_earth * math.cos(math.radians(float(obs_lat)))
    horiz_m = math.hypot(north_m, east_m)

    # Earth curvature + standard atmospheric refraction correction
    curvature_drop_m = (horiz_m ** 2) / (2.0 * R_earth * 1.14)
    up_m = (alt_m - obs_alt_m) - curvature_drop_m

    az_deg = (math.degrees(math.atan2(east_m, north_m)) + 360.0) % 360.0
    el_deg = math.degrees(math.atan2(up_m, max(1.0, horiz_m)))
    slant_km = math.hypot(horiz_m, up_m) / 1000.0
    ground_km = horiz_m / 1000.0
    return round(el_deg, 2), round(az_deg, 2), round(slant_km, 2), round(ground_km, 2)


def altaz_to_pixel(
    alt_deg: float,
    az_deg: float,
    cfg: Optional[dict] = None,
    shape: Tuple[int, int] = (3552, 3552),
) -> Tuple[int, int]:
    """
    Maps topocentric (alt_deg, az_deg) to pixel (x, y) on the 3552x3552 Allsky fisheye sensor
    using the 22-star 3D Spherical Tilt + 5th-Order Radial Fisheye calibration.
    """
    if cfg is None:
        cfg = load_config()
    h, w = shape[:2]
    cx, cy = cfg.get("zenith_center", [1750.95, 1690.25])
    r_horiz = float(cfg.get("horizon_radius_px", 2007.74))
    bottom_az = float(cfg.get("bottom_azimuth_deg", 265.055))
    tilt_x = math.radians(float(cfg.get("tilt_x_deg", 6.292)))
    tilt_y = math.radians(float(cfg.get("tilt_y_deg", -0.947)))
    k2 = float(cfg.get("radial_k2", -0.14857))
    k3 = float(cfg.get("radial_k3", 0.03125))

    # Scale for non-3552 image sizes (e.g. 1080x1080 or 1920x1920 snapshots)
    scale_x = w / 3552.0
    scale_y = h / 3552.0
    cx_s = float(cx) * scale_x
    cy_s = float(cy) * scale_y
    r_horiz_s = r_horiz * ((scale_x + scale_y) * 0.5)

    alt_r = math.radians(max(-8.0, min(90.0, float(alt_deg))))
    az_r = math.radians(float(az_deg))
    east = math.cos(alt_r) * math.sin(az_r)
    north = math.cos(alt_r) * math.cos(az_r)
    up = math.sin(alt_r)

    phi = math.radians(bottom_az)
    cb, sb = math.cos(phi), math.sin(phi)
    vx0 = cb * east - sb * north
    vy0 = sb * east + cb * north
    vz0 = up

    ctx, stx = math.cos(tilt_x), math.sin(tilt_x)
    vx1 = vx0
    vy1 = ctx * vy0 - stx * vz0
    vz1 = stx * vy0 + ctx * vz0

    cty, sty = math.cos(tilt_y), math.sin(tilt_y)
    vx_c = cty * vx1 + sty * vz1
    vy_c = vy1
    vz_c = -sty * vx1 + cty * vz1

    theta = math.acos(max(-1.0, min(1.0, vz_c)))
    u = theta / (math.pi / 2.0)
    r = r_horiz_s * (u + k2 * (u ** 3) + k3 * (u ** 5))

    rho = math.hypot(vx_c, vy_c)
    if rho > 1e-9:
        px = int(round(cx_s + r * (vx_c / rho)))
        py = int(round(cy_s + r * (vy_c / rho)))
    else:
        px = int(round(cx_s))
        py = int(round(cy_s))
    return px, py


def pixel_to_alt_az(
    x: float,
    y: float,
    cfg: Optional[dict] = None,
    shape: Tuple[int, int] = (3552, 3552),
) -> Tuple[float, float, str]:
    """
    Exact inverse of `altaz_to_pixel`: converts pixel (x, y) on the circular fisheye sensor to
    (altitude_deg [0..90], azimuth_deg [0..360], cardinal_str) using the 3D Spherical Tilt + 5th-order model.
    """
    if cfg is None:
        cfg = load_config()
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


def angular_separation_deg(alt1: float, az1: float, alt2: float, az2: float) -> float:
    """Computes true great-circle angular separation (in degrees) between two sky (Alt, Az) points."""
    a1, z1 = math.radians(alt1), math.radians(az1)
    a2, z2 = math.radians(alt2), math.radians(az2)
    cos_sep = math.sin(a1) * math.sin(a2) + math.cos(a1) * math.cos(a2) * math.cos(z1 - z2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_sep))))


def propagate_latlon(lat: float, lon: float, gs_kt: float, track_deg: float, dt_sec: float) -> Tuple[float, float]:
    """Dead-reckons aircraft (lat, lon) forward or backward by `dt_sec` seconds using groundspeed and track."""
    if abs(dt_sec) < 0.05 or gs_kt <= 1.0:
        return lat, lon
    speed_mps = float(gs_kt) * 0.514444
    dist_m = speed_mps * float(dt_sec)
    R = 6371000.0
    trk_rad = math.radians(float(track_deg))
    d_north = dist_m * math.cos(trk_rad)
    d_east = dist_m * math.sin(trk_rad)
    new_lat = lat + math.degrees(d_north / R)
    new_lon = lon + math.degrees(d_east / (R * max(0.1, math.cos(math.radians(lat)))))
    return new_lat, new_lon


def format_flight_identity(ac: Dict[str, Any]) -> Dict[str, str]:
    """
    Builds human-friendly flight identification strings from raw ADS-B + enriched metadata:
      - callsign (e.g. 'ASA412' or 'N560PC')
      - airline (e.g. 'Alaska Airlines')
      - registration (e.g. 'N560PC')
      - type_code (e.g. 'C152', 'B738')
      - model_name (e.g. 'Cessna 152', 'Boeing 737-800')
      - label_short (for image bounding-box tag, e.g. 'N560PC (C152) 1,500ft')
      - subject_line (for event card title & Discord alert)
      - description_line (for detailed telemetry narrative)
    """
    hex_id = str(ac.get("hex") or "").upper()
    flight_raw = str(ac.get("flight") or "").strip().upper()
    reg = str(ac.get("r") or ac.get("registration") or "").strip().upper()
    t_code = str(ac.get("t") or ac.get("type_code") or "").strip().upper()
    desc_raw = str(ac.get("desc") or "").strip()

    # Check enrichment cache if missing
    if hex_id in _ENRICHMENT_CACHE:
        cached = _ENRICHMENT_CACHE[hex_id]
        if not reg:
            reg = cached.get("r", "")
        if not t_code:
            t_code = cached.get("t", "")
        if not desc_raw:
            desc_raw = cached.get("desc", "")
        if not flight_raw:
            flight_raw = cached.get("flight", "")

    callsign = flight_raw or reg or f"ICAO-{hex_id}"
    airline = ""
    if len(callsign) >= 4 and callsign[:3].isalpha() and callsign[3:].isdigit():
        airline = AIRLINE_PREFIXES.get(callsign[:3], "")

    model_name = desc_raw or ICAO_TYPE_NAMES.get(t_code, t_code)
    alt_ft = ac.get("alt_geom") or ac.get("alt_baro") or 0
    alt_str = f"{int(alt_ft):,} ft" if isinstance(alt_ft, (int, float)) and alt_ft > 0 else "Low Alt"
    gs_kt = float(ac.get("gs") or 0.0)
    trk = float(ac.get("track") or 0.0)
    dist_mi = float(ac.get("slant_km") or 0.0) * 0.621371

    type_badge = f" ({t_code})" if t_code else ""
    reg_badge = f" [{reg}]" if reg and reg != callsign else ""
    airline_prefix = f"{airline} " if airline else ""

    label_short = f"{callsign}{type_badge} • {alt_str}"
    subject_line = f"✈️ {airline_prefix}Flight {callsign}"
    if model_name:
        subject_line += f" ({model_name}{reg_badge})"
    elif reg_badge:
        subject_line += reg_badge

    cardinal_idx = int(round(trk / 22.5)) % 16
    hdg_card = CARDINALS[cardinal_idx]
    description_line = (
        f"ADS-B Verified Aircraft: {airline_prefix}{callsign}"
        f"{' (' + model_name + ')' if model_name else ''}"
        f"{' Reg ' + reg if reg and reg != callsign else ''} "
        f"at {alt_str} MSL, groundspeed {gs_kt:.0f} kt heading {trk:.0f}° ({hdg_card}), "
        f"slant range {dist_mi:.1f} mi ({ac.get('slant_km', 0):.1f} km) [ICAO hex {hex_id}]."
    )

    return {
        "hex": hex_id,
        "callsign": callsign,
        "airline": airline,
        "registration": reg,
        "type_code": t_code,
        "model_name": model_name,
        "label_short": label_short,
        "subject_line": subject_line,
        "description_line": description_line,
    }


def _effective_camera_geodetic(cfg: dict) -> Tuple[float, float, float]:
    """
    Returns (lat, lon, alt_ft) for `cfg`, applying any 3D baseline meter offsets
    (east_offset_m, north_offset_m, up_offset_m) for multi-camera (1..N) setups.
    """
    base_lat = float(cfg.get("camera_lat", DEFAULT_LAT))
    base_lon = float(cfg.get("camera_lon", DEFAULT_LON))
    base_alt_ft = float(cfg.get("camera_alt_ft", DEFAULT_ALT_FT))
    east_m = float(cfg.get("east_offset_m") or 0.0)
    north_m = float(cfg.get("north_offset_m") or 0.0)
    up_m = float(cfg.get("up_offset_m") or 0.0)
    if abs(east_m) > 1e-3 or abs(north_m) > 1e-3 or abs(up_m) > 1e-3:
        r_earth = 6371000.0
        base_lat += math.degrees(north_m / r_earth)
        base_lon += math.degrees(east_m / (r_earth * max(0.1, math.cos(math.radians(base_lat)))))
        base_alt_ft += up_m * 3.28084
    return base_lat, base_lon, base_alt_ft


# ---------------------------------------------------------------------------
# Live Polling of Local ADS-B Receiver(s) (1..N Camera Pis + Dedicated Host)
# ---------------------------------------------------------------------------
def poll_adsb_once(cfg: Optional[dict] = None) -> Dict[str, Any]:
    global _LAST_REGIONAL_FETCH, _LATEST_SKY_SNAPSHOT
    if cfg is None:
        cfg = load_config()

    obs_lat, obs_lon, obs_alt_ft = _effective_camera_geodetic(cfg)
    adsb_host = str(cfg.get("adsb_tracker_host") or "").strip()

    # Build list of receiver endpoint groups (each group represents one physical Pi receiver;
    # we try candidate URLs within a group until one responds, and union-merge across all groups!)
    receiver_groups: List[List[str]] = []

    if cfg.get("adsb_local_urls"):
        for u in cfg["adsb_local_urls"]:
            if u:
                receiver_groups.append([str(u)])
    elif adsb_host:
        receiver_groups.append(
            [
                f"http://{adsb_host}/tar1090/data/aircraft.json",
                f"http://{adsb_host}/adsbx/data/aircraft.json",
                f"http://{adsb_host}:8096/api/adsb",
                "http://ADBSTracker/tar1090/data/aircraft.json",
            ]
        )

    # Also include any 1..N Camera Pi nodes where co-located ADS-B ("adsb_enabled": true) is enabled
    for cam in cfg.get("cameras") or []:
        if isinstance(cam, dict) and cam.get("enabled", True) and cam.get("adsb_enabled", False):
            b_url = str(cam.get("allsky_base_url") or cam.get("base_url") or "").rstrip("/")
            api_u = str(cam.get("adsb_api_url") or (f"{b_url}:8096/api/adsb" if b_url else "")).strip()
            group = []
            if api_u:
                group.append(api_u)
            if b_url:
                group.append(f"{b_url}/tar1090/data/aircraft.json")
            if group:
                receiver_groups.append(group)

    now_ts = time.time()
    merged_by_hex: Dict[str, Dict[str, Any]] = {}
    local_ok = False
    local_messages = 0

    # 1. Poll all configured local ADS-B receivers and union-merge by ICAO hex (freshest seen_pos wins)
    for group in receiver_groups:
        for url in group:
            try:
                r = requests.get(url, timeout=1.8)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("enabled") is False:
                        break
                    local_ok = True
                    local_messages += int(data.get("messages") or 0)
                    for ac in data.get("aircraft", []):
                        hx = str(ac.get("hex") or "").strip().lower()
                        if not hx:
                            continue
                        ac["source"] = "local_adsbtracker"
                        if hx not in merged_by_hex:
                            merged_by_hex[hx] = ac
                        else:
                            # Keep whichever receiver saw this aircraft more recently
                            prev = merged_by_hex[hx]
                            seen_new = float(ac.get("seen_pos") if ac.get("seen_pos") is not None else 999.0)
                            seen_old = float(prev.get("seen_pos") if prev.get("seen_pos") is not None else 999.0)
                            if seen_new < seen_old and ac.get("lat") is not None and ac.get("lon") is not None:
                                for k in ("flight", "r", "t", "desc"):
                                    if prev.get(k) and not ac.get(k):
                                        ac[k] = prev[k]
                                merged_by_hex[hx] = ac
                            else:
                                for k in ("flight", "r", "t", "desc"):
                                    if ac.get(k) and not prev.get(k):
                                        prev[k] = ac[k]
                    break
            except Exception:
                continue

    # 2. Enrich / supplement every 6 seconds from regional ADS-B network within 45 nm
    if now_ts - _LAST_REGIONAL_FETCH >= 6.0:
        _LAST_REGIONAL_FETCH = now_ts
        reg_url = f"https://api.adsb.lol/v2/point/{obs_lat:.5f}/{obs_lon:.5f}/45"
        try:
            r_reg = requests.get(reg_url, timeout=3.0)
            if r_reg.status_code == 200:
                reg_data = r_reg.json()
                for ac in reg_data.get("ac") or reg_data.get("aircraft") or []:
                    hx = str(ac.get("hex") or "").strip().lower()
                    if not hx:
                        continue
                    _ENRICHMENT_CACHE[hx.upper()] = {
                        "r": str(ac.get("r") or "").strip().upper(),
                        "t": str(ac.get("t") or "").strip().upper(),
                        "desc": str(ac.get("desc") or "").strip(),
                        "flight": str(ac.get("flight") or "").strip().upper(),
                    }
                    if hx not in merged_by_hex:
                        ac["source"] = "regional_adsb"
                        merged_by_hex[hx] = ac
                    else:
                        # Copy registration/type fields into local ADBSTracker record
                        for k in ("r", "t", "desc"):
                            if ac.get(k) and not merged_by_hex[hx].get(k):
                                merged_by_hex[hx][k] = ac[k]
                        if ac.get("flight") and not merged_by_hex[hx].get("flight"):
                            merged_by_hex[hx]["flight"] = ac["flight"]
        except Exception:
            pass

    aircraft_in_range: List[Dict[str, Any]] = []
    aircraft_in_dome: List[Dict[str, Any]] = []

    with _LOCK:
        for hx, ac in merged_by_hex.items():
            lat = ac.get("lat")
            lon = ac.get("lon")
            seen_pos = float(ac.get("seen_pos") or ac.get("seen") or 0.0)
            if lat is None or lon is None or seen_pos > 45.0:
                continue

            alt_val = ac.get("alt_geom")
            if alt_val is None or alt_val == "ground":
                alt_val = ac.get("alt_baro")
            if alt_val == "ground" or alt_val is None:
                continue
            alt_ft = float(alt_val)

            gs_kt = float(ac.get("gs") or 0.0)
            trk_deg = float(ac.get("track") or 0.0)

            # Dead-reckon forward by `seen_pos` seconds so position is exact at `now_ts`
            cur_lat, cur_lon = propagate_latlon(float(lat), float(lon), gs_kt, trk_deg, min(15.0, seen_pos))
            el_deg, az_deg, slant_km, ground_km = geodetic_to_altaz(
                cur_lat, cur_lon, alt_ft, obs_lat, obs_lon, obs_alt_ft
            )
            px, py = altaz_to_pixel(el_deg, az_deg, cfg, (3552, 3552))
            idx_c = int(round(az_deg / 22.5)) % 16

            enriched = dict(ac)
            enriched.update(
                {
                    "hex": hx.upper(),
                    "lat": round(cur_lat, 6),
                    "lon": round(cur_lon, 6),
                    "alt_ft": round(alt_ft, 0),
                    "gs_kt": round(gs_kt, 1),
                    "track_deg": round(trk_deg, 1),
                    "alt_deg": el_deg,
                    "az_deg": az_deg,
                    "cardinal": CARDINALS[idx_c],
                    "slant_km": slant_km,
                    "slant_miles": round(slant_km * 0.621371, 2),
                    "ground_km": ground_km,
                    "dome_x": px,
                    "dome_y": py,
                    "timestamp": now_ts,
                }
            )
            ident = format_flight_identity(enriched)
            enriched.update(ident)

            aircraft_in_range.append(enriched)
            if el_deg >= 3.0:
                aircraft_in_dome.append(enriched)

            hist_list = _AIRCRAFT_HISTORY.setdefault(hx.upper(), [])
            hist_list.append(enriched)
            # Prune samples older than 240 seconds
            cutoff = now_ts - 240.0
            _AIRCRAFT_HISTORY[hx.upper()] = [s for s in hist_list if s["timestamp"] >= cutoff]

        # Clean up stale aircraft keys
        stale_keys = [
            k for k, v in _AIRCRAFT_HISTORY.items() if not v or (now_ts - v[-1]["timestamp"]) > 240.0
        ]
        for k in stale_keys:
            del _AIRCRAFT_HISTORY[k]

        aircraft_in_range.sort(key=lambda x: x["slant_km"])
        aircraft_in_dome.sort(key=lambda x: x["alt_deg"], reverse=True)

        _LATEST_SKY_SNAPSHOT = {
            "updated_at": now_ts,
            "updated_iso": datetime.now().isoformat(),
            "local_receiver_ok": local_ok,
            "local_messages": local_messages,
            "observer": {"lat": obs_lat, "lon": obs_lon, "alt_ft": obs_alt_ft},
            "aircraft_in_range_count": len(aircraft_in_range),
            "aircraft_in_dome_count": len(aircraft_in_dome),
            "aircraft_in_range": aircraft_in_range,
            "aircraft_in_dome": aircraft_in_dome,
        }

    try:
        with open(ADSB_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(_LATEST_SKY_SNAPSHOT, f, indent=2)
    except Exception:
        pass

    return _LATEST_SKY_SNAPSHOT


def get_live_adsb_snapshot() -> Dict[str, Any]:
    if time.time() - float(_LATEST_SKY_SNAPSHOT.get("updated_at") or 0.0) > 4.0:
        return poll_adsb_once()
    return _LATEST_SKY_SNAPSHOT


# ---------------------------------------------------------------------------
# Point-to-Segment Distance Helper on 3552x3552 Dome
# ---------------------------------------------------------------------------
def _point_to_segment_dist_px(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float]:
    """Returns (min_distance_px, closest_x, closest_y) from (px, py) to segment (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-3:
        return math.hypot(px - x1, py - y1), x1, y1
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y), proj_x, proj_y


def _estimate_aircraft_state_at(
    samples: List[Dict[str, Any]],
    target_ts: float,
) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Returns (lat, lon, alt_ft, gs_kt, track_deg) at `target_ts` by interpolating between
    bracketing historical ADS-B samples in `samples`, or dead-reckoning from the closest sample.
    """
    if not samples:
        return None
    for i in range(len(samples) - 1):
        sa = samples[i]
        sb = samples[i + 1]
        ta = float(sa["timestamp"])
        tb = float(sb["timestamp"])
        if ta <= target_ts <= tb and (tb - ta) > 1e-3:
            w = (target_ts - ta) / (tb - ta)
            lat = float(sa["lat"]) * (1.0 - w) + float(sb["lat"]) * w
            lon = float(sa["lon"]) * (1.0 - w) + float(sb["lon"]) * w
            alt_ft = float(sa["alt_ft"]) * (1.0 - w) + float(sb["alt_ft"]) * w
            gs_kt = float(sa["gs_kt"]) * (1.0 - w) + float(sb["gs_kt"]) * w
            da = ((float(sb["track_deg"]) - float(sa["track_deg"]) + 180.0) % 360.0) - 180.0
            trk = (float(sa["track_deg"]) + w * da) % 360.0
            return lat, lon, alt_ft, gs_kt, trk

    closest = min(samples, key=lambda s: abs(target_ts - float(s["timestamp"])))
    dt = target_ts - float(closest["timestamp"])
    if abs(dt) > 110.0:
        return None
    lat, lon = propagate_latlon(
        float(closest["lat"]),
        float(closest["lon"]),
        float(closest["gs_kt"]),
        float(closest["track_deg"]),
        dt,
    )
    return lat, lon, float(closest["alt_ft"]), float(closest["gs_kt"]), float(closest["track_deg"])


# ---------------------------------------------------------------------------
# Match an Allsky Anomaly Detection Against Active ADS-B Aircraft Trajectories
# ---------------------------------------------------------------------------
def match_detection_to_adsb(
    candidate: Dict[str, Any],
    cfg: Optional[dict] = None,
    exposure_sec: float = 20.0,
    frame_timestamp: Optional[float] = None,
    exposure_start_ts: Optional[float] = None,
    exposure_end_ts: Optional[float] = None,
    shape: Tuple[int, int] = (3552, 3552),
) -> Optional[Dict[str, Any]]:
    """
    Checks if a detected night/day sky anomaly at `candidate` ('center', 'alt_deg', 'az_deg', 'bbox')
    coincides with any live ADS-B aircraft trajectory during the exact shutter open interval
    `[exposure_start_ts, exposure_end_ts]`.
    If matched:
      - Returns the matched aircraft dict + trajectory segment `((x1, y1), (x2, y2))` + match metrics.
      - Logs the correspondence to `camera_calibration.json` to continuously refine camera alignment!
    """
    if cfg is None:
        cfg = load_config()
    poll_adsb_once(cfg)

    t_end = float(exposure_end_ts or frame_timestamp or time.time())
    exp_dur = max(0.05, float(exposure_sec))
    t_start = float(exposure_start_ts) if exposure_start_ts is not None else (t_end - exp_dur)
    t_mid = 0.5 * (t_start + t_end)

    # For short daytime exposures (< 2.5s), compute a 10-second flight segment around t_mid
    # so the trajectory arrow clearly indicates the aircraft's flight path and heading.
    if (t_end - t_start) < 2.5:
        seg_t1 = t_mid - 2.0
        seg_t2 = t_mid + 8.0
    else:
        seg_t1 = t_start
        seg_t2 = t_end

    cx, cy = candidate.get("center", (0.0, 0.0))
    cand_alt = float(candidate.get("alt_deg") or 0.0)
    cand_az = float(candidate.get("az_deg") or 0.0)
    cand_len = float(candidate.get("length_px") or 0.0)
    cand_lin = float(candidate.get("linearity") or 0.0)
    cand_ang = candidate.get("angle_deg")

    obs_lat, obs_lon, obs_alt_ft = _effective_camera_geodetic(cfg)

    best_match: Optional[Dict[str, Any]] = None
    best_score_px = 99999.0

    with _LOCK:
        histories = {k: list(v) for k, v in _AIRCRAFT_HISTORY.items()}

    for hx, samples in histories.items():
        if not samples:
            continue
        st1 = _estimate_aircraft_state_at(samples, seg_t1)
        stm = _estimate_aircraft_state_at(samples, t_mid)
        st2 = _estimate_aircraft_state_at(samples, seg_t2)
        if st1 is None or stm is None or st2 is None:
            continue

        el1, az1, _, _ = geodetic_to_altaz(st1[0], st1[1], st1[2], obs_lat, obs_lon, obs_alt_ft)
        elm, azm, slant_km, _ = geodetic_to_altaz(stm[0], stm[1], stm[2], obs_lat, obs_lon, obs_alt_ft)
        el2, az2, _, _ = geodetic_to_altaz(st2[0], st2[1], st2[2], obs_lat, obs_lon, obs_alt_ft)
        if max(el1, elm, el2) < 1.5:
            continue

        x1, y1 = altaz_to_pixel(el1, az1, cfg, shape)
        xm, ym = altaz_to_pixel(elm, azm, cfg, shape)
        x2, y2 = altaz_to_pixel(el2, az2, cfg, shape)

        dist_px, proj_x, proj_y = _point_to_segment_dist_px(float(cx), float(cy), float(x1), float(y1), float(x2), float(y2))
        ang_sep1 = angular_separation_deg(cand_alt, cand_az, el1, az1)
        ang_sepm = angular_separation_deg(cand_alt, cand_az, elm, azm)
        ang_sep2 = angular_separation_deg(cand_alt, cand_az, el2, az2)
        ang_sep = min(ang_sep1, ang_sepm, ang_sep2)

        # If the optical detection is a resolved linear streak, verify its axis aligns with the ADS-B trajectory
        adsb_dx = float(x2 - x1)
        adsb_dy = float(y2 - y1)
        adsb_seg_len = math.hypot(adsb_dx, adsb_dy)
        if cand_len >= 45.0 and cand_lin >= 0.88 and cand_ang is not None and adsb_seg_len >= 25.0:
            adsb_ang_180 = math.degrees(math.atan2(adsb_dy, adsb_dx)) % 180.0
            d_ang = abs(float(cand_ang) - adsb_ang_180) % 180.0
            d_ang = min(d_ang, 180.0 - d_ang)
            if d_ang > 28.0:
                continue

        # Match threshold: within 285 px on 3552x3552 sensor OR within 13.5 deg angular separation
        max_px_thresh = 285.0 * (shape[0] / 3552.0)
        if dist_px <= max_px_thresh or ang_sep <= 13.5:
            if dist_px < best_score_px:
                best_score_px = dist_px
                latest = samples[-1]
                best_match = dict(latest)

                # Fuse optical trail endpoints with ADS-B flight direction so the trajectory arrow
                # aligns cleanly with the optical detection while pointing in the exact direction of travel:
                p1_opt = candidate.get("p1")
                p2_opt = candidate.get("p2")
                if p1_opt and p2_opt and cand_len >= 35.0 and adsb_seg_len >= 8.0:
                    opt_dx = float(p2_opt[0]) - float(p1_opt[0])
                    opt_dy = float(p2_opt[1]) - float(p1_opt[1])
                    if (opt_dx * adsb_dx + opt_dy * adsb_dy) >= 0.0:
                        draw_start = [int(round(p1_opt[0])), int(round(p1_opt[1]))]
                        draw_end = [int(round(p2_opt[0])), int(round(p2_opt[1]))]
                    else:
                        draw_start = [int(round(p2_opt[0])), int(round(p2_opt[1]))]
                        draw_end = [int(round(p1_opt[0])), int(round(p1_opt[1]))]
                elif adsb_seg_len >= 8.0:
                    # Point/daytime aircraft: anchor arrow at optical center (cx, cy) along ADS-B velocity vector
                    draw_start = [int(round(cx)), int(round(cy))]
                    draw_end = [int(round(cx + adsb_dx)), int(round(cy + adsb_dy))]
                else:
                    draw_start = [int(x1), int(y1)]
                    draw_end = [int(x2), int(y2)]

                best_match.update(
                    {
                        "match_dist_px": round(dist_px, 1),
                        "match_ang_deg": round(ang_sep, 2),
                        "seg_start_px": draw_start,
                        "seg_end_px": draw_end,
                        "adsb_seg_start_px": [int(x1), int(y1)],
                        "adsb_seg_mid_px": [int(xm), int(ym)],
                        "adsb_seg_end_px": [int(x2), int(y2)],
                        "matched_proj_px": [int(round(proj_x)), int(round(proj_y))],
                        "exposure_mid_ts": round(t_mid, 3),
                        "exposure_el_start": el1,
                        "exposure_az_start": az1,
                        "exposure_el_mid": elm,
                        "exposure_az_mid": azm,
                        "exposure_el_end": el2,
                        "exposure_az_end": az2,
                    }
                )

    if best_match is not None:
        _record_calibration_observation(candidate, best_match)

    return best_match


def _record_calibration_observation(candidate: Dict[str, Any], adsb_match: Dict[str, Any]) -> None:
    """
    Records paired (Camera Pixel Detection <-> ADS-B WAAS GPS Sky Position) observations
    at the exact midpoint of the shutter exposure so the camera's alignment can self-heal.
    """
    try:
        cal = {"pairs": [], "star_calibration": {}}
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                    cal = json.load(f)
            except Exception:
                cal = {"pairs": [], "star_calibration": {}}

        cx, cy = candidate.get("center", (0.0, 0.0))
        cal.setdefault("pairs", []).append(
            {
                "timestamp": adsb_match.get("exposure_mid_ts") or time.time(),
                "callsign": adsb_match.get("callsign"),
                "hex": adsb_match.get("hex"),
                "det_x": round(float(cx), 1),
                "det_y": round(float(cy), 1),
                "adsb_alt_deg": adsb_match.get("exposure_el_mid") or adsb_match.get("exposure_el_end"),
                "adsb_az_deg": adsb_match.get("exposure_az_mid") or adsb_match.get("exposure_az_end"),
                "match_dist_px": adsb_match.get("match_dist_px"),
                "match_ang_deg": adsb_match.get("match_ang_deg"),
            }
        )
        cal["pairs"] = cal["pairs"][-100:]
        with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
            json.dump(cal, f, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Draw Live ADS-B Flight Callsigns & Vectors on Allsky Images
# ---------------------------------------------------------------------------
def draw_adsb_overlay_on_image(
    img: np.ndarray,
    cfg: Optional[dict] = None,
    highlight_hex: Optional[str] = None,
    exposure_sec: float = 30.0,
) -> np.ndarray:
    """
    Draws all currently visible ADS-B aircraft (`alt_deg >= 3.0`) onto `img`
    with their exposure-window trajectory arrow, callsign, aircraft type, altitude, speed, and distance.
    """
    if cfg is None:
        cfg = load_config()
    snap = get_live_adsb_snapshot()
    in_dome = snap.get("aircraft_in_dome", [])
    if not in_dome:
        return img

    out = img.copy()
    h, w = out.shape[:2]
    scale = max(0.45, w / 2200.0)
    thick = max(2, int(round(scale * 2.5)))
    exp_dur = max(10.0, min(90.0, float(exposure_sec or 30.0)))

    cam_lat = float(cfg.get("camera_lat", DEFAULT_LAT))
    cam_lon = float(cfg.get("camera_lon", DEFAULT_LON))
    cam_alt = float(cfg.get("camera_alt_ft", DEFAULT_ALT_FT))

    for ac in in_dome:
        el = float(ac.get("alt_deg") or 0.0)
        az = float(ac.get("az_deg") or 0.0)
        px, py = altaz_to_pixel(el, az, cfg, (h, w))
        if not (10 <= px < w - 10 and 10 <= py < h - 10):
            continue

        # Back-propagate across exposure window (-exp_dur -> now) and +10s ahead for flight vector
        lat0, lon0 = propagate_latlon(ac["lat"], ac["lon"], ac["gs_kt"], ac["track_deg"], -exp_dur)
        el0, az0, _, _ = geodetic_to_altaz(lat0, lon0, ac["alt_ft"], cam_lat, cam_lon, cam_alt)
        px0, py0 = altaz_to_pixel(el0, az0, cfg, (h, w))

        lat2, lon2 = propagate_latlon(ac["lat"], ac["lon"], ac["gs_kt"], ac["track_deg"], 12.0)
        el2, az2, _, _ = geodetic_to_altaz(lat2, lon2, ac["alt_ft"], cam_lat, cam_lon, cam_alt)
        px2, py2 = altaz_to_pixel(el2, az2, cfg, (h, w))

        is_hl = highlight_hex and str(ac.get("hex")).upper() == str(highlight_hex).upper()
        color = (0, 225, 255) if is_hl else (60, 195, 255)  # Gold/Amber for aircraft radar overlay

        if math.hypot(px - px0, py - py0) >= 8:
            cv2.circle(out, (px0, py0), max(3, int(5 * scale)), color, -1, cv2.LINE_AA)
            cv2.line(out, (px0, py0), (px, py), color, thick, cv2.LINE_AA)
        cv2.circle(out, (px, py), int(15 * scale), color, thick, cv2.LINE_AA)
        if math.hypot(px2 - px, py2 - py) >= 6:
            cv2.arrowedLine(out, (px, py), (px2, py2), color, thick, cv2.LINE_AA, tipLength=0.30)

        t_code = ac.get("type_code") or ac.get("model") or "ADS-B"
        dist_mi = ac.get("ground_dist_miles")
        dist_str = f" | {dist_mi:.1f}mi" if dist_mi is not None else ""
        label = f"FLIGHT: {ac.get('callsign')} ({t_code}) | {int(ac.get('alt_ft', 0)):,}ft | {int(ac.get('gs_kt', 0))}kt{dist_str}"
        font_scale = 0.62 * scale
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        lx = max(10, min(w - tw - 16, px + int(18 * scale)))
        ly = max(th + 14, min(h - 14, py - int(10 * scale)))
        cv2.rectangle(out, (lx - 6, ly - th - 6), (lx + tw + 6, ly + 6), (12, 18, 32), -1)
        cv2.rectangle(out, (lx - 6, ly - th - 6), (lx + tw + 6, ly + 6), color, max(1, thick - 1))
        cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Background Polling Daemon Thread
# ---------------------------------------------------------------------------
def _adsb_polling_loop() -> None:
    logger.info("Started background ADS-B sky tracker thread (ADBSTracker 192.168.1.50 + regional enrichment).")
    while True:
        try:
            poll_adsb_once()
        except Exception as e:
            logger.debug(f"ADS-B poll warning: {e}")
        time.sleep(2.5)


def start_background_adsb_thread() -> None:
    global _TRACKER_THREAD_STARTED
    if _TRACKER_THREAD_STARTED:
        return
    _TRACKER_THREAD_STARTED = True
    t = threading.Thread(target=_adsb_polling_loop, name="AllskyADSBTrackerThread", daemon=True)
    t.start()
