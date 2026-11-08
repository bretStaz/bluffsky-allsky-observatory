"""
Astrophysics & Solar System Discovery Patrol Engine for BluffSky Allsky (ASI676MC)
-----------------------------------------------------------------------------------
Transforms the calibrated 22-star 3D Spherical Fisheye Allsky camera into a continuous
wide-field astrophysical transient, meteor radiant, and recurrent-nova patrol instrument:

1. Exact J2000.0 Equatorial (RA, Dec) and Galactic (l, b) astrometry for any sensor pixel (x, y, t_utc).
2. Earth Umbral Shadow Cone geometry (height in km along line of sight) to distinguish LEO sunlit glints
   from deep-shadow cislunar / astrophysical optical transients (Novae, Fast Optical Transients, GRBs).
3. Live Recurrent Nova & Extragalactic Supernova Watchlist (including T Coronae Borealis 'Blaze Star',
   RS Ophiuchi, M31 Andromeda, M33 Triangulum, P Cygni, Gamma Cas, Mira, Algol, Betelgeuse).
4. Great-Circle Back-Projection of verified meteors/fireballs to active IAU Meteor Shower Radiants
   (Orionids, S/N Taurids, Leonids, Epsilon Geminids, Draconids) + Pairwise Great-Circle Intersection
   clustering to detect uncataloged meteor stream outbursts.
"""

import datetime
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import ephem
import numpy as np

import adsb_tracker
import detector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Curated Recurrent Nova, Symbiotic Binary & Extragalactic Supernova Watchlist
# ---------------------------------------------------------------------------
NOVA_AND_DEEP_SKY_WATCHLIST: List[Dict[str, Any]] = [
    {
        "id": "T_CrB",
        "name": "T Coronae Borealis (The Blaze Star)",
        "constellation": "Corona Borealis",
        "type": "Recurrent Nova (80-yr cycle)",
        "ra_str": "15:59:30.2",
        "dec_str": "+25:55:13",
        "quiescent_mag": 10.0,
        "eruption_mag": 2.0,
        "priority": "CRITICAL",
        "notes": "Last erupted in 1866 & 1946; pre-eruption dip observed — will jump 1,500x in brightness to Mag +2.0 (matching Polaris) within hours.",
    },
    {
        "id": "RS_Oph",
        "name": "RS Ophiuchi",
        "constellation": "Ophiuchus",
        "type": "Recurrent Symbiotic Nova",
        "ra_str": "17:50:13.2",
        "dec_str": "-06:42:28",
        "quiescent_mag": 11.0,
        "eruption_mag": 4.3,
        "priority": "HIGH",
        "notes": "White dwarf accreting from red giant; erupts to naked-eye Mag +4.3 with thermonuclear shockwaves.",
    },
    {
        "id": "M31",
        "name": "M31 — Andromeda Galaxy",
        "constellation": "Andromeda",
        "type": "Local Group Spiral (2.54 Mly)",
        "ra_str": "00:42:44.3",
        "dec_str": "+41:16:09",
        "quiescent_mag": 3.4,
        "eruption_mag": 5.5,
        "priority": "HIGH",
        "notes": "Primary extragalactic patrol target for Galactic-scale Core-Collapse / Type Ia Supernovae (like S Andromedae 1885) & Luminous Red Novae.",
    },
    {
        "id": "M33",
        "name": "M33 — Triangulum Galaxy",
        "constellation": "Triangulum",
        "type": "Local Group Spiral (2.73 Mly)",
        "ra_str": "01:33:50.9",
        "dec_str": "+30:39:37",
        "quiescent_mag": 5.7,
        "eruption_mag": 6.5,
        "priority": "MEDIUM",
        "notes": "High star-formation rate Local Group spiral; monitored for bright supernovae and LBV eruptions.",
    },
    {
        "id": "P_Cyg",
        "name": "P Cygni (34 Cygni)",
        "constellation": "Cygnus",
        "type": "Luminous Blue Variable Hypergiant",
        "ra_str": "20:17:47.2",
        "dec_str": "+38:01:59",
        "quiescent_mag": 4.8,
        "eruption_mag": 2.8,
        "priority": "HIGH",
        "notes": "Prototype LBV hypergiant (Great Eruption of 1600 AD); subject to sudden mass-ejection brightenings.",
    },
    {
        "id": "Gamma_Cas",
        "name": "Gamma Cassiopeiae (γ Cas)",
        "constellation": "Cassiopeia",
        "type": "Be X-Ray Binary / Shell Star",
        "ra_str": "00:56:42.5",
        "dec_str": "+60:43:00",
        "quiescent_mag": 2.2,
        "eruption_mag": 1.5,
        "priority": "MEDIUM",
        "notes": "Circumpolar from BluffSky (48.1°N); rapid rotator with decretion disk outbursts and hard X-ray/optical flares.",
    },
    {
        "id": "Betelgeuse",
        "name": "Betelgeuse (α Orionis)",
        "constellation": "Orion",
        "type": "Red Supergiant (Core-Collapse Prog.)",
        "ra_str": "05:55:10.3",
        "dec_str": "+07:24:25",
        "quiescent_mag": 0.5,
        "eruption_mag": -12.0,
        "priority": "HIGH",
        "notes": "Nearest red supergiant (~550 ly); monitored for episodic dust-ejection dimmings and shock breakout.",
    },
    {
        "id": "Mira",
        "name": "Mira (ο Ceti)",
        "constellation": "Cetus",
        "type": "AGB Pulsating Variable + Accreting WD",
        "ra_str": "02:19:20.8",
        "dec_str": "-02:58:40",
        "quiescent_mag": 6.5,
        "eruption_mag": 2.0,
        "priority": "MEDIUM",
        "notes": "332-day pulsating red giant with white dwarf companion; peaks between Mag +2.0 and +4.5.",
    },
    {
        "id": "Algol",
        "name": "Algol (β Persei)",
        "constellation": "Perseus",
        "type": "Prototype Eclipsing Binary",
        "ra_str": "03:08:10.1",
        "dec_str": "+40:57:20",
        "quiescent_mag": 2.1,
        "eruption_mag": 3.4,
        "priority": "MEDIUM",
        "notes": "Drops 1.3 magnitudes every 2.867 days during primary eclipse; benchmark photometric calibration target.",
    },
    {
        "id": "CH_Cyg",
        "name": "CH Cygni",
        "constellation": "Cygnus",
        "type": "Symbiotic White Dwarf Jet System",
        "ra_str": "19:24:33.1",
        "dec_str": "+50:14:29",
        "quiescent_mag": 7.8,
        "eruption_mag": 5.4,
        "priority": "MEDIUM",
        "notes": "M-giant + magnetic white dwarf launching relativistic bipolar jets during accretion state transitions.",
    },
]

# ---------------------------------------------------------------------------
# Active Autumn / Early-Winter IAU Meteor Showers (Oct - Nov Patrol)
# ---------------------------------------------------------------------------
IAU_METEOR_SHOWERS: List[Dict[str, Any]] = [
    {
        "code": "ORI",
        "name": "Orionids",
        "parent_body": "Comet 1P/Halley",
        "ra_deg": 95.0,
        "dec_deg": 16.0,
        "v_inf_kms": 66.0,
        "active_window": "Oct 2 – Nov 7 (Peak Oct 21)",
        "color": "#38bdf8",
    },
    {
        "code": "STA",
        "name": "Southern Taurids",
        "parent_body": "Comet 2P/Encke (Halloween Fireball Stream)",
        "ra_deg": 50.0,
        "dec_deg": 13.5,
        "v_inf_kms": 27.0,
        "active_window": "Sep 20 – Nov 20 (Swarm Peak Oct 25–Nov 5)",
        "color": "#f59e0b",
    },
    {
        "code": "NTA",
        "name": "Northern Taurids",
        "parent_body": "Comet 2P/Encke / Asteroid 2004 TG10",
        "ra_deg": 58.0,
        "dec_deg": 22.0,
        "v_inf_kms": 29.0,
        "active_window": "Oct 20 – Dec 10 (Peak Nov 12)",
        "color": "#fb923c",
    },
    {
        "code": "EGE",
        "name": "Epsilon Geminids",
        "parent_body": "Long-Period Comet C/1964 N1 (Ikeya)",
        "ra_deg": 102.0,
        "dec_deg": 27.0,
        "v_inf_kms": 70.0,
        "active_window": "Oct 14 – Oct 27",
        "color": "#a78bfa",
    },
    {
        "code": "LMI",
        "name": "Leo Minorids",
        "parent_body": "Comet C/1739 K1",
        "ra_deg": 162.0,
        "dec_deg": 37.0,
        "v_inf_kms": 62.0,
        "active_window": "Oct 19 – Oct 27",
        "color": "#34d399",
    },
    {
        "code": "LEO",
        "name": "Leonids",
        "parent_body": "Comet 55P/Tempel-Tuttle",
        "ra_deg": 152.0,
        "dec_deg": 22.0,
        "v_inf_kms": 71.0,
        "active_window": "Nov 6 – Nov 30 (Peak Nov 17)",
        "color": "#f472b6",
    },
    {
        "code": "DRA",
        "name": "Oct. Draconids",
        "parent_body": "Comet 21P/Giacobini-Zinner",
        "ra_deg": 262.0,
        "dec_deg": 54.0,
        "v_inf_kms": 20.0,
        "active_window": "Oct 6 – Oct 10 (Late stragglers possible)",
        "color": "#60a5fa",
    },
    {
        "code": "AND",
        "name": "Andromedids",
        "parent_body": "Disintegrated Comet 3D/Biela",
        "ra_deg": 25.0,
        "dec_deg": 44.0,
        "v_inf_kms": 19.0,
        "active_window": "Oct 25 – Dec 6",
        "color": "#c084fc",
    },
]


def _make_observer(cfg: dict, ts_utc: Optional[float] = None) -> ephem.Observer:
    obs = ephem.Observer()
    obs.lat = str(cfg.get("camera_lat", 45.0000))
    obs.lon = str(cfg.get("camera_lon", -100.0000))
    obs.elevation = float(cfg.get("camera_alt_ft", 130.0)) * 0.3048
    obs.pressure = 0  # Geometric astrometry matches SGP4 / star calibration
    if ts_utc is None:
        ts_utc = time.time()
    dt_utc = datetime.datetime.fromtimestamp(float(ts_utc), tz=datetime.timezone.utc)
    obs.date = dt_utc.strftime("%Y/%m/%d %H:%M:%S.%f")
    return obs


def format_ra_hms(ra_deg: float) -> str:
    ra_h = (ra_deg % 360.0) / 15.0
    h = int(ra_h)
    m_rem = (ra_h - h) * 60.0
    m = int(m_rem)
    s = (m_rem - m) * 60.0
    return f"{h:02d}h {m:02d}m {s:04.1f}s"


def format_dec_dms(dec_deg: float) -> str:
    sign = "+" if dec_deg >= 0 else "-"
    val = abs(dec_deg)
    d = int(val)
    m_rem = (val - d) * 60.0
    m = int(m_rem)
    s = (m_rem - m) * 60.0
    return f"{sign}{d:02d}° {m:02d}' {s:02.0f}\""


def compute_earth_shadow_height_km(alt_deg: float, az_deg: float, obs: ephem.Observer) -> float:
    """
    Computes the altitude (in km above Earth's surface) where the observer's line of sight
    (alt_deg, az_deg) exits Earth's cylindrical umbral shadow into sunlight.
    - Returns 0.0 if the Sun is above the horizon.
    - High values (e.g. > 1,500 km) mean Low-Earth-Orbit satellites along this sightline are in complete
      darkness (eclipsed), so any short optical flash at that position must be either self-luminous,
      high-altitude (MEO/GEO/Cislunar), or a true Deep-Space Optical Transient.
    """
    try:
        sun = ephem.Sun(obs)
        sun_alt = float(sun.alt)
        sun_az = float(sun.az)
        if sun_alt >= 0.0:
            return 0.0

        alt_r = math.radians(max(1.0, alt_deg))
        az_r = math.radians(az_deg)

        # Local Horizontal ENU unit vectors: [East, North, Up]
        los_enu = np.array([
            math.cos(alt_r) * math.sin(az_r),
            math.cos(alt_r) * math.cos(az_r),
            math.sin(alt_r),
        ], dtype=np.float64)

        sun_enu = np.array([
            math.cos(sun_alt) * math.sin(sun_az),
            math.cos(sun_alt) * math.cos(sun_az),
            math.sin(sun_alt),
        ], dtype=np.float64)

        R_e = 6371.0  # km
        obs_pos = np.array([0.0, 0.0, R_e], dtype=np.float64)

        # Ray P(s) = obs_pos + s * los_enu (s >= 0 in km).
        # A point P(s) is in Earth's shadow if dot(P(s), sun_enu) < 0 and ||P(s) - dot(P(s), sun_enu)*sun_enu|| < R_e.
        # Solve ||P(s) - (P(s).sun)*sun||^2 = R_e^2 along s > 0:
        u_perp = los_enu - np.dot(los_enu, sun_enu) * sun_enu
        p0_perp = obs_pos - np.dot(obs_pos, sun_enu) * sun_enu

        a = float(np.dot(u_perp, u_perp))
        b = 2.0 * float(np.dot(p0_perp, u_perp))
        c = float(np.dot(p0_perp, p0_perp)) - R_e * R_e

        if a < 1e-9:
            return 45000.0
        disc = b * b - 4.0 * a * c
        if disc <= 0.0:
            return 0.0
        s_exit = (-b + math.sqrt(disc)) / (2.0 * a)
        if s_exit <= 0.0:
            return 0.0
        p_exit = obs_pos + s_exit * los_enu
        # If p_exit is on the sunlit hemisphere (dot > 0), shadow height is 0
        if float(np.dot(p_exit, sun_enu)) > 0.0:
            return 0.0
        h_km = float(np.linalg.norm(p_exit)) - R_e
        return round(max(0.0, min(h_km, 60000.0)), 1)
    except Exception:
        return 0.0


def altaz_to_radec_j2000(
    alt_deg: float,
    az_deg: float,
    ts_utc: float,
    cfg: dict,
) -> Dict[str, Any]:
    """
    Converts topocentric (alt_deg, az_deg) and UTC timestamp into J2000.0 Equatorial (RA, Dec),
    Galactic (l, b), and Earth Umbral Shadow Height (km).
    """
    obs = _make_observer(cfg, ts_utc)
    az_rad = math.radians(az_deg % 360.0)
    alt_rad = math.radians(max(-89.9, min(89.9, alt_deg)))

    ra_of_date, dec_of_date = obs.radec_of(az_rad, alt_rad)
    eq_date = ephem.Equatorial(ra_of_date, dec_of_date, epoch=obs.date)
    eq_j2000 = ephem.Equatorial(eq_date, epoch=ephem.J2000)
    gal = ephem.Galactic(eq_j2000)

    ra_deg = math.degrees(float(eq_j2000.ra)) % 360.0
    dec_deg = math.degrees(float(eq_j2000.dec))
    gal_l_deg = math.degrees(float(gal.lon)) % 360.0
    gal_b_deg = math.degrees(float(gal.lat))
    shadow_km = compute_earth_shadow_height_km(alt_deg, az_deg, obs)

    return {
        "ra_deg": round(ra_deg, 4),
        "dec_deg": round(dec_deg, 4),
        "ra_hms": format_ra_hms(ra_deg),
        "dec_dms": format_dec_dms(dec_deg),
        "gal_l_deg": round(gal_l_deg, 2),
        "gal_b_deg": round(gal_b_deg, 2),
        "shadow_height_km": shadow_km,
    }


def pixel_to_radec_j2000(
    x_px: float,
    y_px: float,
    ts_utc: float,
    cfg: dict,
    shape: Tuple[int, int] = (3552, 3552),
) -> Dict[str, Any]:
    """
    Converts sensor pixel (x_px, y_px) at UTC timestamp into topocentric (Alt, Az) and J2000 (RA, Dec).
    """
    alt_deg, az_deg, cardinal = detector.pixel_to_alt_az(x_px, y_px, cfg, shape)
    astro = altaz_to_radec_j2000(alt_deg, az_deg, ts_utc, cfg)
    astro["alt_deg"] = alt_deg
    astro["az_deg"] = az_deg
    astro["cardinal"] = cardinal
    return astro


def radec_to_unit_vec(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra_r = math.radians(ra_deg)
    dec_r = math.radians(dec_deg)
    return np.array([
        math.cos(dec_r) * math.cos(ra_r),
        math.cos(dec_r) * math.sin(ra_r),
        math.sin(dec_r),
    ], dtype=np.float64)


def unit_vec_to_radec(vec: np.ndarray) -> Tuple[float, float]:
    n = float(np.linalg.norm(vec))
    if n < 1e-12:
        return 0.0, 0.0
    v = vec / n
    dec_deg = math.degrees(math.asin(max(-1.0, min(1.0, float(v[2])))))
    ra_deg = math.degrees(math.atan2(float(v[1]), float(v[0]))) % 360.0
    return ra_deg, dec_deg


def angular_sep_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    u1 = radec_to_unit_vec(ra1, dec1)
    u2 = radec_to_unit_vec(ra2, dec2)
    dot = max(-1.0, min(1.0, float(np.dot(u1, u2))))
    return math.degrees(math.acos(dot))


def _az_to_cardinal(az_deg: float) -> str:
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return dirs[int(round((az_deg % 360.0) / 22.5)) % 16]


def get_live_watchlist_status(
    cfg: dict,
    ts_utc: Optional[float] = None,
    shape: Tuple[int, int] = (3552, 3552),
) -> List[Dict[str, Any]]:
    """
    Computes the real-time altitude, azimuth, dome pixel coordinates, and observability
    status of all Recurrent Nova, Symbiotic Binary, and Extragalactic Supernova targets.
    """
    if ts_utc is None:
        ts_utc = time.time()
    obs = _make_observer(cfg, ts_utc)
    max_r = float(cfg.get("max_sky_radius_px", 1530.0)) - float(cfg.get("mask_erosion_px", 32.0))
    mcx, mcy = cfg.get("mask_center", [1718, 1796])

    results: List[Dict[str, Any]] = []
    for item in NOVA_AND_DEEP_SKY_WATCHLIST:
        body = ephem.FixedBody()
        body._ra = ephem.hours(item["ra_str"])
        body._dec = ephem.degrees(item["dec_str"])
        body._epoch = ephem.J2000
        body.compute(obs)

        alt_deg = round(math.degrees(float(body.alt)), 2)
        az_deg = round(math.degrees(float(body.az)) % 360.0, 2)
        ra_deg = round(math.degrees(float(ephem.hours(item["ra_str"]))), 3)
        dec_deg = round(math.degrees(float(ephem.degrees(item["dec_str"]))), 3)

        px = None
        in_dome = False
        if alt_deg > 0.0:
            pt = adsb_tracker.altaz_to_pixel(alt_deg, az_deg, cfg, shape)
            if pt is not None:
                px = [pt[0], pt[1]]
                dist_mask = math.hypot(pt[0] - mcx, pt[1] - mcy)
                in_dome = (alt_deg >= 12.0) and (dist_mask <= max_r)

        if in_dome:
            vis_status = "ACTIVE IN DOME"
        elif alt_deg > 0.0:
            vis_status = "LOW ON HORIZON"
        else:
            vis_status = "BELOW HORIZON"

        results.append({
            **item,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "alt_deg": alt_deg,
            "az_deg": az_deg,
            "cardinal": _az_to_cardinal(az_deg),
            "pixel": px,
            "in_dome": in_dome,
            "visibility_status": vis_status,
        })

    # Sort so targets currently inside the active dome appear first, ordered by priority & altitude
    prio_rank = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1}
    results.sort(
        key=lambda x: (
            1 if x["in_dome"] else (0 if x["alt_deg"] > 0 else -1),
            prio_rank.get(x["priority"], 1),
            x["alt_deg"],
        ),
        reverse=True,
    )
    return results


def _parse_event_features(ev: Dict[str, Any]) -> Dict[str, Any]:
    raw = ev.get("features_json")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def _event_epoch_ts(ev: Dict[str, Any]) -> float:
    fn = str(ev.get("frame_filename") or ev.get("filename") or ev.get("snapshot_filename") or "")
    fallback_ts = float(ev.get("detected_at") or ev.get("epoch") or time.time())
    s_ts, e_ts = detector.get_frame_exposure_window(fn, 20.0, fallback_ts)
    return 0.5 * (s_ts + e_ts)


def _has_adsb_or_sat_match(ev: Dict[str, Any], feat: Dict[str, Any]) -> Tuple[bool, bool]:
    has_adsb = bool(
        ev.get("adsb_match")
        or feat.get("adsb_flight")
        or "ADS-B" in str(ev.get("subject", ""))
        or "Flight " in str(ev.get("subject", ""))
    )
    orb = ev.get("orbital_match") or feat.get("orbital_object")
    has_sat = bool(orb or "NORAD #" in str(ev.get("subject", "")))
    # Reject legacy loose Mode-2 flash matches (dist > 28 px or wide cloud blob w > 9.0 px)
    if isinstance(orb, dict) and orb.get("match_mode") == "flash":
        if float(orb.get("match_dist_px", 0.0)) > 28.5 or float(feat.get("width_px", 0.0)) > 9.0:
            has_sat = False
    return has_adsb, has_sat


_EPHEM_BUCKET_CACHE: Dict[int, Dict[str, Any]] = {}
_PATROL_REPORT_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}


def invalidate_patrol_cache() -> None:
    _PATROL_REPORT_CACHE["ts"] = 0.0
    _PATROL_REPORT_CACHE["data"] = None


def _cached_ephemeris(ts_utc: float, cfg: dict) -> Dict[str, Any]:
    bucket = int(ts_utc // 60)
    res = _EPHEM_BUCKET_CACHE.get(bucket)
    if res is None:
        res = detector.compute_ephemeris_sun_moon(ts_utc, cfg, (3552, 3552))
        if len(_EPHEM_BUCKET_CACHE) > 2000:
            _EPHEM_BUCKET_CACHE.clear()
        _EPHEM_BUCKET_CACHE[bucket] = res
    return res


def _is_cloud_artifact_event(ev: Dict[str, Any], feat: Dict[str, Any], cfg: Optional[dict] = None) -> bool:
    """
    Filters out legacy pre-ephemeris cloud/roofline/lunar-halo false positives when building the astrophysical catalog.
    """
    user_cat = str(ev.get("user_corrected_category") or "")
    if user_cat in ("meteor", "fireball", "transient_flash", "sky_phenomenon"):
        return False
    if user_cat in ("false_positive", "aircraft", "day_aircraft", "day_bird"):
        return True
    has_adsb, _ = _has_adsb_or_sat_match(ev, feat)
    if has_adsb:
        return False
    cat = str(ev.get("category", ""))
    if cat in ("aircraft", "day_aircraft", "day_bird", "day_anomaly"):
        return True
    w_px = float(feat.get("width_px") or ev.get("width_px") or 0.0)
    if w_px > 9.5:
        return True
    bbox = ev.get("bbox") or feat.get("bbox") or [0, 0, 0, 0]
    if len(bbox) == 4:
        cx = 0.5 * (float(bbox[0]) + float(bbox[2]))
        cy = 0.5 * (float(bbox[1]) + float(bbox[3]))
        # North/ENE roofline & floodlight cloud zone
        if cy < 860 and 900 <= cx <= 2650:
            return True
        if cfg is not None:
            ts_utc = _event_epoch_ts(ev)
            ep = _cached_ephemeris(ts_utc, cfg)
            mx, my = ep.get("moon_px", (-9999, -9999))
            m_phase = float(ep.get("moon_phase", 0.0))
            if ep.get("moon_alt_deg", -90.0) > -6.0 and m_phase >= 20.0:
                r_lunar = 240.0 + 5.2 * m_phase
                if math.hypot(cx - mx, cy - my) <= r_lunar:
                    return True
    desc = (str(ev.get("vlm_description", "")) + " " + str(ev.get("description", "")) + " " + str(ev.get("subject", ""))).lower()
    if "cloud" in desc:
        return True
    return False


def analyze_meteor_radiants(
    events: List[Dict[str, Any]],
    cfg: dict,
    shape: Tuple[int, int] = (3552, 3552),
) -> Dict[str, Any]:
    """
    Performs great-circle back-projection of verified meteors and fireballs in J2000 (RA, Dec):
    1. Matches each meteor track against active IAU meteor showers (Orionids, S/N Taurids, Leonids, etc.).
    2. Computes pairwise great-circle intersections among sporadic/unmatched tracks to detect
       Candidate Uncataloged Meteor Stream Radiant Clusters.
    """
    meteor_tracks: List[Dict[str, Any]] = []
    shower_counts: Dict[str, int] = {s["code"]: 0 for s in IAU_METEOR_SHOWERS}
    shower_counts["SPORADIC"] = 0

    for ev in events:
        cat = str(ev.get("category", ""))
        if cat not in ("meteor", "fireball"):
            continue
        feat = _parse_event_features(ev)
        has_adsb, has_sat = _has_adsb_or_sat_match(ev, feat)
        if has_adsb or has_sat or _is_cloud_artifact_event(ev, feat, cfg):
            continue
        bbox = ev.get("bbox") or feat.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        bw, bh = max(8.0, x2 - x1), max(8.0, y2 - y1)
        length_px = float(ev.get("length_px") or feat.get("length_px") or math.hypot(bw, bh))
        if length_px < 25.0:
            continue

        ts_utc = _event_epoch_ts(ev)
        p1_f = ev.get("p1") or feat.get("p1")
        p2_f = ev.get("p2") or feat.get("p2")
        if p1_f and p2_f and len(p1_f) == 2 and len(p2_f) == 2:
            px1, py1 = float(p1_f[0]), float(p1_f[1])
            px2, py2 = float(p2_f[0]), float(p2_f[1])
        else:
            px1, py1 = x1, y1
            px2, py2 = x2, y2

        c_start = pixel_to_radec_j2000(px1, py1, ts_utc, cfg, shape)
        c_end = pixel_to_radec_j2000(px2, py2, ts_utc, cfg, shape)
        c_mid = pixel_to_radec_j2000(cx, cy, ts_utc, cfg, shape)

        u1 = radec_to_unit_vec(c_start["ra_deg"], c_start["dec_deg"])
        u2 = radec_to_unit_vec(c_end["ra_deg"], c_end["dec_deg"])
        pole = np.cross(u1, u2)
        p_norm = float(np.linalg.norm(pole))
        if p_norm < 1e-7:
            continue
        pole = pole / p_norm

        best_shower = None
        best_gc_err = 999.0
        u_mid = radec_to_unit_vec(c_mid["ra_deg"], c_mid["dec_deg"])

        for sh in IAU_METEOR_SHOWERS:
            u_rad = radec_to_unit_vec(sh["ra_deg"], sh["dec_deg"])
            gc_dist_deg = abs(math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(pole, u_rad)))))))
            sep_mid_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u_mid, u_rad))))))
            if gc_dist_deg <= 7.2 and 12.0 <= sep_mid_deg <= 125.0:
                if gc_dist_deg < best_gc_err:
                    best_gc_err = gc_dist_deg
                    best_shower = sh

        if best_shower is not None:
            shower_code = best_shower["code"]
            shower_name = best_shower["name"]
            parent_body = best_shower["parent_body"]
            shower_color = best_shower["color"]
            shower_counts[shower_code] = shower_counts.get(shower_code, 0) + 1
        else:
            shower_code = "SPORADIC"
            shower_name = "Sporadic / Uncataloged Stream"
            parent_body = "Uncataloged Solar System Debris"
            shower_color = "#e2e8f0"
            shower_counts["SPORADIC"] += 1

        ev_id = str(ev.get("frame_filename") or ev.get("id") or "")
        snap_fn = str(ev.get("snapshot_filename") or ev.get("snapshot") or "")
        clean_subj = str(ev.get("subject") or "Meteor Streak").split(" + 🛰️")[0]

        meteor_tracks.append({
            "id": ev_id,
            "timestamp": ev.get("frame_time_display") or ev.get("timestamp"),
            "date": ev.get("frame_date") or ev.get("date"),
            "category": cat,
            "subject": clean_subj,
            "snapshot": snap_fn,
            "snr": ev.get("snr", 0.0),
            "length_px": round(length_px, 1),
            "alt_deg": c_mid["alt_deg"],
            "az_deg": c_mid["az_deg"],
            "cardinal": c_mid["cardinal"],
            "ra_deg": c_mid["ra_deg"],
            "dec_deg": c_mid["dec_deg"],
            "ra_hms": c_mid["ra_hms"],
            "dec_dms": c_mid["dec_dms"],
            "start_radec": [c_start["ra_deg"], c_start["dec_deg"]],
            "end_radec": [c_end["ra_deg"], c_end["dec_deg"]],
            "pole_vec": pole.tolist(),
            "u_mid_vec": u_mid.tolist(),
            "shower_code": shower_code,
            "shower_name": shower_name,
            "parent_body": parent_body,
            "shower_gc_offset_deg": round(best_gc_err, 2) if best_shower else None,
            "color": shower_color,
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "user_corrected_category": str(ev.get("user_corrected_category") or ""),
        })

    # Pairwise Great-Circle Intersection Clustering for Candidate New/Minor Stream Outbursts
    intersections: List[Dict[str, Any]] = []
    tracks_sub = meteor_tracks[:65]
    n_tracks = len(tracks_sub)
    for i in range(n_tracks):
        for j in range(i + 1, n_tracks):
            t1 = tracks_sub[i]
            t2 = tracks_sub[j]
            p1 = np.array(t1["pole_vec"], dtype=np.float64)
            p2 = np.array(t2["pole_vec"], dtype=np.float64)
            cross = np.cross(p1, p2)
            cn = float(np.linalg.norm(cross))
            if cn < 0.08:  # Nearly parallel great circles
                continue
            cross = cross / cn
            u_m1 = np.array(t1["u_mid_vec"], dtype=np.float64)
            u_m2 = np.array(t2["u_mid_vec"], dtype=np.float64)

            # Test both antipodal intersections (+cross and -cross)
            for sign in (1.0, -1.0):
                cand_vec = sign * cross
                if cand_vec[2] < -0.35:  # Below Dec -20° (not observable radiant for 48°N)
                    continue
                sep1 = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(cand_vec, u_m1))))))
                sep2 = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(cand_vec, u_m2))))))
                if 10.0 <= sep1 <= 115.0 and 10.0 <= sep2 <= 115.0:
                    r_ra, r_dec = unit_vec_to_radec(cand_vec)
                    intersections.append({
                        "ra_deg": r_ra,
                        "dec_deg": r_dec,
                        "vec": cand_vec,
                        "tracks": {t1["id"], t2["id"]},
                    })

    # Cluster pairwise intersections within 6.0 degrees using vectorized NumPy dot products
    radiant_clusters: List[Dict[str, Any]] = []
    if intersections:
        all_vecs = np.array([pt["vec"] for pt in intersections], dtype=np.float64)
        cos_thresh = math.cos(math.radians(6.0))
        for idx_pt, pt in enumerate(intersections):
            dots = all_vecs @ pt["vec"]
            nbr_indices = np.where(dots >= cos_thresh)[0]
            if len(nbr_indices) < 2:
                continue
            member_tracks = set()
            for ni in nbr_indices:
                member_tracks.update(intersections[int(ni)]["tracks"])
            if len(member_tracks) >= 2:
                mean_v = np.mean(all_vecs[nbr_indices], axis=0)
                c_ra, c_dec = unit_vec_to_radec(mean_v)
                matched_iau = None
                for sh in IAU_METEOR_SHOWERS:
                    if angular_sep_deg(c_ra, c_dec, sh["ra_deg"], sh["dec_deg"]) <= 8.5:
                        matched_iau = sh
                        break
                radiant_clusters.append({
                    "ra_deg": round(c_ra, 2),
                    "dec_deg": round(c_dec, 2),
                    "ra_hms": format_ra_hms(c_ra),
                    "dec_dms": format_dec_dms(c_dec),
                    "meteor_count": len(member_tracks),
                    "track_ids": sorted(list(member_tracks)),
                    "matched_iau_code": matched_iau["code"] if matched_iau else None,
                    "matched_iau_name": matched_iau["name"] if matched_iau else "Uncataloged Radiant Convergence",
                    "parent_body": matched_iau["parent_body"] if matched_iau else "Candidate Minor Stream / Filament",
                })

    # Deduplicate clusters within 7.5 degrees, keeping largest meteor_count
    radiant_clusters.sort(key=lambda c: c["meteor_count"], reverse=True)
    dedup_clusters: List[Dict[str, Any]] = []
    for cl in radiant_clusters:
        if any(angular_sep_deg(cl["ra_deg"], cl["dec_deg"], ex["ra_deg"], ex["dec_deg"]) < 7.5 for ex in dedup_clusters):
            continue
        dedup_clusters.append(cl)

    # Clean internal numpy vectors before JSON serialization
    for t in meteor_tracks:
        t.pop("pole_vec", None)
        t.pop("u_mid_vec", None)

    return {
        "total_verified_meteors": len(meteor_tracks),
        "shower_counts": shower_counts,
        "iau_showers": IAU_METEOR_SHOWERS,
        "meteor_tracks": meteor_tracks[:45],
        "radiant_convergences": dedup_clusters[:8],
    }


def analyze_uncataloged_transients(
    events: List[Dict[str, Any]],
    cfg: dict,
    shape: Tuple[int, int] = (3552, 3552),
) -> List[Dict[str, Any]]:
    """
    Identifies High-Priority Uncataloged Optical Transients & Deep-Space Candidates from the archive:
    - Point-source optical flashes (`transient_flash`) that did NOT match any cataloged CelesTrak / McCants
      satellite or ADS-B aircraft, especially those occurring high inside Earth's umbral shadow cone.
    - Uncataloged orbital streaks (`space_object` without NORAD match) representing potential uncataloged
      cislunar/HEO objects, classified maneuvering payloads, or near-Earth asteroid / meteoroid grazers.
    - High-energy Fireballs (`fireball`) with full J2000 (RA, Dec) coordinates for trajectory triangulation.
    """
    candidates: List[Dict[str, Any]] = []

    for ev in events:
        feat = _parse_event_features(ev)
        has_adsb, has_sat = _has_adsb_or_sat_match(ev, feat)
        if has_adsb or _is_cloud_artifact_event(ev, feat, cfg):
            continue
        cat = str(ev.get("category", ""))
        if cat not in ("transient_flash", "space_object", "fireball"):
            continue
        # Only include unmatched space_object / transient_flash OR verified fireballs
        if (has_sat or str(ev.get("user_corrected_category") or "") == "space_object") and cat != "fireball":
            continue

        bbox = ev.get("bbox") or feat.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        cx = 0.5 * (float(bbox[0]) + float(bbox[2]))
        cy = 0.5 * (float(bbox[1]) + float(bbox[3]))
        alt_deg = float(ev.get("alt_deg") or 0.0)
        if alt_deg < 16.0:
            continue

        ts_utc = _event_epoch_ts(ev)
        astro = pixel_to_radec_j2000(cx, cy, ts_utc, cfg, shape)
        shadow_km = astro["shadow_height_km"]

        # Check proximity to any Watchlist target
        near_target = None
        for w in NOVA_AND_DEEP_SKY_WATCHLIST:
            w_ra = math.degrees(float(ephem.hours(w["ra_str"])))
            w_dec = math.degrees(float(ephem.degrees(w["dec_str"])))
            sep = angular_sep_deg(astro["ra_deg"], astro["dec_deg"], w_ra, w_dec)
            if sep <= 5.0:
                near_target = f"{w['name']} ({sep:.1f}° sep)"
                break

        if cat == "transient_flash":
            if shadow_km >= 1200.0:
                disc_class = "Deep-Shadow Uncataloged Optical Flash"
                sci_note = (
                    f"Line of sight exits Earth's umbral shadow at {shadow_km:,.0f} km altitude — "
                    f"LEO satellites (<1,000 km) are fully eclipsed! Candidate High-Altitude Cislunar/HEO Glint "
                    f"or Astrophysical Fast Optical Transient."
                )
                prio = "HIGH"
            else:
                disc_class = "Uncataloged Sub-Second Optical Flash"
                sci_note = (
                    f"Point-source flash absent from CelesTrak (14,800+ active) & McCants (2,900+ classified) catalogs "
                    f"(Earth shadow height {shadow_km:,.0f} km). Candidate uncataloged tumbling debris or head-on meteor."
                )
                prio = "MEDIUM"
        elif cat == "space_object":
            disc_class = "Uncataloged Orbital / Cislunar Streak"
            sci_note = (
                f"Linear sky track ({ev.get('length_px', 0):.0f} px) with zero match in public CelesTrak or classified "
                f"McCants TLE catalogs. Candidate freshly maneuvered payload, HEO object, or grazing meteoroid."
            )
            prio = "HIGH" if shadow_km >= 1000.0 else "MEDIUM"
        else:
            disc_class = "Bolide / Fireball Trajectory Anchor"
            sci_note = (
                f"High-flux atmospheric ablation event (SNR {ev.get('snr', 0):.1f}σ) anchored at J2000 "
                f"{astro['ra_hms']} {astro['dec_dms']} for multi-station orbit & meteorite strewn-field triangulation."
            )
            prio = "HIGH"

        ev_id = str(ev.get("frame_filename") or ev.get("id") or "")
        snap_fn = str(ev.get("snapshot_filename") or ev.get("snapshot") or "")
        clean_subj = str(ev.get("subject") or disc_class).split(" + 🛰️")[0]

        candidates.append({
            "id": ev_id,
            "timestamp": ev.get("frame_time_display") or ev.get("timestamp"),
            "date": ev.get("frame_date") or ev.get("date"),
            "category": cat,
            "discovery_class": disc_class,
            "priority": prio,
            "subject": clean_subj,
            "snapshot": snap_fn,
            "snr": ev.get("snr", 0.0),
            "length_px": ev.get("length_px", 0.0),
            "alt_deg": astro["alt_deg"],
            "az_deg": astro["az_deg"],
            "cardinal": astro["cardinal"],
            "ra_deg": astro["ra_deg"],
            "dec_deg": astro["dec_deg"],
            "ra_hms": astro["ra_hms"],
            "dec_dms": astro["dec_dms"],
            "gal_l_deg": astro["gal_l_deg"],
            "gal_b_deg": astro["gal_b_deg"],
            "shadow_height_km": shadow_km,
            "near_watchlist_target": near_target,
            "scientific_assessment": sci_note,
            "bbox": [round(float(v), 1) for v in bbox],
            "user_corrected_category": str(ev.get("user_corrected_category") or ""),
        })

    prio_map = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1}
    candidates.sort(key=lambda c: (prio_map.get(c["priority"], 1), c.get("shadow_height_km", 0.0), c.get("snr", 0.0)), reverse=True)
    return candidates[:35]


def build_discovery_patrol_report(cfg: Optional[dict] = None) -> Dict[str, Any]:
    """
    Master aggregation function for the `/discoveries` web portal and `/api/discoveries` JSON endpoint.
    """
    now_ts = time.time()
    if _PATROL_REPORT_CACHE["data"] is not None and (now_ts - _PATROL_REPORT_CACHE["ts"]) < 15.0:
        return _PATROL_REPORT_CACHE["data"]

    if cfg is None:
        cfg = detector.load_config()
    events: List[Dict[str, Any]] = []
    db_path = os.path.join(BASE_DIR, "allsky_detections.db")
    if os.path.exists(db_path):
        try:
            import sqlite3
            conn = sqlite3.connect(db_path, timeout=5)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM sky_events WHERE is_hidden = 0 AND (user_corrected_category IS NULL OR user_corrected_category != 'false_positive') ORDER BY id DESC LIMIT 600"
            )
            events = [dict(r) for r in cur.fetchall()]
            conn.close()
        except Exception:
            events = []
    if not events:
        history = detector.load_history()
        events = history.get("recent_events") or history.get("events") or []

    ephem_now = detector.compute_ephemeris_sun_moon(now_ts, cfg, (3552, 3552))
    watchlist = get_live_watchlist_status(cfg, now_ts, (3552, 3552))
    radiants = analyze_meteor_radiants(events, cfg, (3552, 3552))
    transients = analyze_uncataloged_transients(events, cfg, (3552, 3552))

    # Compute Local Sidereal Time (LST) at BluffSky
    obs = _make_observer(cfg, now_ts)
    lst_rad = float(obs.sidereal_time())
    lst_deg = math.degrees(lst_rad) % 360.0
    lst_hms = format_ra_hms(lst_deg)

    report = {
        "generated_at_utc": datetime.datetime.fromtimestamp(now_ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "observatory": {
            "name": cfg.get("camera_name", "BluffSky Allsky (ASI676MC)"),
            "lat": cfg.get("camera_lat", 45.0000),
            "lon": cfg.get("camera_lon", -100.0000),
            "lst_hms": lst_hms,
            "lst_deg": round(lst_deg, 2),
            "sun_alt_deg": ephem_now.get("sun_alt_deg"),
            "moon_alt_deg": ephem_now.get("moon_alt_deg"),
            "moon_az_deg": ephem_now.get("moon_az_deg"),
            "moon_phase_pct": ephem_now.get("moon_phase_pct"),
            "dark_sky_patrol_active": bool(ephem_now.get("sun_alt_deg", 0.0) <= -12.0 and ephem_now.get("moon_alt_deg", 0.0) <= 5.0),
        },
        "watchlist": watchlist,
        "meteor_patrol": radiants,
        "uncataloged_transients": transients,
    }
    _PATROL_REPORT_CACHE["ts"] = now_ts
    _PATROL_REPORT_CACHE["data"] = report
    return report


DISCOVERY_PATROL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>BluffSky Astrophysics & Solar System Discovery Patrol</title>
<style>
  :root {
    --bg: #070b14;
    --panel: #0f172a;
    --panel2: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --cyan: #38bdf8;
    --emerald: #34d399;
    --amber: #fbbf24;
    --purple: #c084fc;
    --rose: #fb7185;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.45;
  }
  header {
    background: rgba(15, 23, 42, 0.94); border-bottom: 1px solid var(--border);
    padding: 14px 24px; display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 12px; position: sticky; top: 0; z-index: 100; backdrop-filter: blur(10px);
  }
  .brand h1 { margin: 0; font-size: 1.25rem; font-weight: 800; letter-spacing: -0.02em; color: #f8fafc; }
  .brand span { font-size: 0.8rem; color: var(--muted); }
  .btn {
    background: #2563eb; color: #fff; border: 1px solid #3b82f6; padding: 7px 14px;
    border-radius: 8px; font-weight: 600; font-size: 0.82rem; cursor: pointer; text-decoration: none;
    display: inline-flex; align-items: center; gap: 6px; transition: all 0.15s;
  }
  .btn:hover { filter: brightness(1.12); }
  .btn.secondary { background: var(--panel2); border-color: var(--border); color: #e2e8f0; }
  .btn.emerald { background: #059669; border-color: #10b981; }
  .btn.purple { background: #7c3aed; border-color: #a855f7; }
  .pill {
    background: var(--panel2); border: 1px solid var(--border); padding: 5px 11px;
    border-radius: 999px; font-size: 0.78rem; font-weight: 600; color: #e2e8f0;
  }
  .container { max-width: 1580px; margin: 0 auto; padding: 20px 24px 60px; }
  .kpi-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-bottom: 18px;
  }
  .kpi-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px;
  }
  .kpi-card .lbl { font-size: 0.76rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .kpi-card .val { font-size: 1.45rem; font-weight: 800; margin-top: 4px; }
  .kpi-card .sub { font-size: 0.76rem; color: var(--muted); margin-top: 2px; }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
    padding: 18px 20px; margin-bottom: 20px;
  }
  .card h2 {
    margin: 0 0 12px; font-size: 1.05rem; font-weight: 700; display: flex;
    justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
  }
  .sky-canvas-wrap {
    position: relative; background: #04070e; border: 1px solid #1e293b; border-radius: 12px;
    overflow: hidden; width: 100%;
  }
  canvas#celestialCanvas { width: 100%; height: auto; display: block; cursor: crosshair; }
  .legend-bar {
    display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px; font-size: 0.8rem; color: #cbd5e1;
  }
  .two-col { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 18px; }
  @media (max-width: 1120px) { .two-col { grid-template-columns: 1fr; } }
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th, td { padding: 9px 10px; border-bottom: 1px solid rgba(51, 65, 85, 0.6); text-align: left; }
  th { color: var(--muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; }
  tr:hover td { background: rgba(30, 41, 59, 0.45); }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 700;
  }
  .badge.crit { background: rgba(244, 63, 94, 0.2); color: #fda4af; border: 1px solid rgba(244, 63, 94, 0.45); }
  .badge.high { background: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.45); }
  .badge.med { background: rgba(56, 189, 248, 0.18); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.4); }
  .badge.dome { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.45); }
  .badge.low { background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); }
  .transients-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 14px;
  }
  .t-card {
    background: var(--panel2); border: 1px solid var(--border); border-radius: 12px;
    overflow: hidden; display: flex; flex-direction: column; cursor: pointer; transition: transform 0.14s, border-color 0.14s;
  }
  .t-card:hover { transform: translateY(-2px); border-color: var(--cyan); }
  .t-card img { width: 100%; height: 220px; object-fit: cover; background: #000; }
  .t-body { padding: 12px 14px; display: flex; flex-direction: column; gap: 6px; flex: 1; }
  .mono { font-family: 'JetBrains Mono', Consolas, monospace; font-size: 0.78rem; color: #bae6fd; }
  /* Lightbox Modal */
  #snapModal {
    display: none; position: fixed; inset: 0; background: rgba(3, 6, 14, 0.94); z-index: 999;
    flex-direction: column;
  }
  #snapModal.open { display: flex; }
  .m-top {
    padding: 12px 20px; background: #0f172a; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center; gap: 12px;
  }
  .m-stage {
    flex: 1; position: relative; overflow: hidden; display: flex; justify-content: center; align-items: center;
    cursor: grab;
  }
  .m-stage.dragging { cursor: grabbing; }
  .m-stage img {
    max-width: 90vw; max-height: 84vh; transform-origin: center center; user-select: none;
  }
  .teach-btn {
    background: #1e293b; color: #e2e8f0; border: 1px solid #334155; padding: 4px 8px;
    border-radius: 6px; font-size: 0.75rem; font-weight: 600; cursor: pointer;
    display: inline-flex; align-items: center; gap: 4px; transition: all 0.14s;
  }
  .teach-btn:hover { background: #334155; border-color: #38bdf8; color: #fff; }
  .teach-btn.false-btn { color: #fda4af; border-color: rgba(244, 63, 94, 0.45); background: rgba(136, 19, 55, 0.28); }
  .teach-btn.false-btn:hover { background: rgba(190, 18, 60, 0.55); border-color: #fb7185; color: #fff; }
  .m-nav-arrow {
    position: absolute; top: 50%; transform: translateY(-50%); z-index: 20;
    background: rgba(15, 23, 42, 0.85); color: #f8fafc; border: 1px solid #334155;
    width: 44px; height: 44px; border-radius: 999px; font-size: 1.2rem; cursor: pointer;
    display: none; align-items: center; justify-content: center; transition: all 0.15s;
  }
  .m-nav-arrow:hover { background: #2563eb; border-color: #60a5fa; }
  .m-nav-arrow.prev { left: 18px; }
  .m-nav-arrow.next { right: 18px; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <h1>🔭 BluffSky Astrophysics & Solar System Discovery Patrol</h1>
    <span>22-Star 3D Spherical Fisheye J2000 Astrometry • Recurrent Nova Watch • Meteor Stream Radiant Triangulation • Earth Shadow Cone Transient Filter</span>
  </div>
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
    <span id="lstPill" class="pill">🌌 LST: --</span>
    <span id="moonInfoPill" class="pill">🌕 Moon: --</span>
    <a href="/" class="btn secondary">← Main Observatory Portal</a>
    <a href="/live" class="btn emerald">⛶ Full-Res Live Sky (3552×3552)</a>
    <a href="/astrophoto" class="btn" style="background:#d97706;">✨ Milky Way & Deep-Sky Portraits</a>
    <button class="btn purple" onclick="loadDiscoveryData()">🔄 Refresh Patrol</button>
  </div>
</header>

<div class="container">
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="lbl">Local Sidereal Time (BluffSky)</div>
      <div class="val" id="kpiLst" style="color:var(--cyan);">--</div>
      <div class="sub" id="kpiCoords">45.0000°N, 100.000°W • J2000.0</div>
    </div>
    <div class="kpi-card">
      <div class="lbl">Nova & Deep-Sky Watchlist in Dome</div>
      <div class="val" id="kpiWatchlist" style="color:var(--amber);">--</div>
      <div class="sub">Including T CrB (Blaze Star) & M31</div>
    </div>
    <div class="kpi-card">
      <div class="lbl">Verified Meteor Trajectories</div>
      <div class="val" id="kpiMeteors" style="color:var(--emerald);">--</div>
      <div class="sub" id="kpiShowersSub">Great-Circle Back-Projected</div>
    </div>
    <div class="kpi-card">
      <div class="lbl">Candidate Radiant Convergences</div>
      <div class="val" id="kpiClusters" style="color:var(--purple);">--</div>
      <div class="sub">Pairwise Great-Circle Intersections</div>
    </div>
    <div class="kpi-card">
      <div class="lbl">Uncataloged Deep-Shadow Transients</div>
      <div class="val" id="kpiTransients" style="color:var(--rose);">--</div>
      <div class="sub">Absent from CelesTrak & McCants TLEs</div>
    </div>
  </div>

  <!-- Multi-Epoch Deep-Sky Co-Addition Survey, Master J2000 Atlas & Blink Comparator -->
  <div class="card" style="border-color:rgba(56,189,248,0.45);background:linear-gradient(180deg,rgba(15,23,42,0.95),rgba(9,16,32,0.95));">
    <h2>
      <span>🔭 Multi-Epoch Sidereal Co-Addition Survey & Moving Space Object Tracklet Engine</span>
      <div style="display:flex;gap:10px;align-items:center;">
        <span id="deepSurveyMeta" class="mono" style="color:#6ee7b7;">Loading multi-night J2000 stack...</span>
        <button class="btn secondary" id="btnRescanSurvey" onclick="triggerDeepSurveyScan()" style="padding:5px 12px;font-size:0.78rem;">🔄 Re-Stack Archive</button>
      </div>
    </h2>
    <div id="deepSurveyKpis" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:10px;margin-bottom:14px;"></div>

    <div style="margin-bottom:16px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <div style="font-size:0.84rem;font-weight:700;color:#e0f2fe;">
          🌌 Master J2000.0 Multi-Night Co-Added Reference Sky Atlas (Click to Open Full-Resolution Infinite Zoom & Pan)
        </div>
        <span class="mono" style="color:#94a3b8;">Sidereally de-rotated & median co-added across clear nights — airplanes, satellites & meteors 100% erased</span>
      </div>
      <div id="masterAtlasWrap" style="border:1px solid var(--border);border-radius:10px;overflow:hidden;cursor:zoom-in;background:#050914;">
        <img id="masterAtlasImg" src="" style="width:100%;display:block;" onclick="openUrlModal('Master J2000.0 Multi-Epoch Co-Added Reference Sky Atlas', 'Sidereal De-Rotation + Multi-Night Median Stack', this.src)" />
      </div>
    </div>

    <div class="two-col" style="margin-bottom:14px;">
      <div>
        <div style="font-size:0.84rem;font-weight:700;color:#6ee7b7;margin-bottom:8px;">
          🪐 Multi-Night Recovered Solar System Bodies (Planets, Ice Giant Uranus & Minor Planets):
        </div>
        <div style="overflow-x:auto;">
          <table>
            <thead>
              <tr>
                <th>Body</th>
                <th>Nights Linked</th>
                <th>Observed Sky Motion</th>
                <th>J2000 Position</th>
                <th>Peak SNR</th>
                <th>Blink Cutout</th>
              </tr>
            </thead>
            <tbody id="solsysBody"></tbody>
          </table>
        </div>
      </div>

      <div>
        <div style="font-size:0.84rem;font-weight:700;color:#38bdf8;margin-bottom:8px;">
          🛸 Candidate Undiscovered Moving Tracklets (3-Night Collinear Orbital Motion):
        </div>
        <div style="overflow-x:auto;">
          <table>
            <thead>
              <tr>
                <th>Tracklet ID</th>
                <th>Nights Linked</th>
                <th>Angular Velocity</th>
                <th>Collinearity</th>
                <th>Mean SNR</th>
                <th>Blink Cutout</th>
              </tr>
            </thead>
            <tbody id="trackletsBody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <div>
      <div style="font-size:0.84rem;font-weight:700;color:#fbbf24;margin-bottom:8px;">
        🔬 Blink Comparator Triplet Cutouts (1. Master Reference Atlas vs. 2. Discovery Night Co-Add vs. 3. Difference Residual ΔI):
      </div>
      <div id="blinkCutoutsGrid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px;"></div>
    </div>
  </div>

  <!-- Interactive J2000 Celestial Sphere Map -->
  <div class="card">
    <h2>
      <span>🗺️ Interactive J2000.0 Celestial Sphere Map (Right Ascension 0h–24h vs. Declination -30° to +90°)</span>
      <span id="hoverReadout" class="mono" style="color:#38bdf8;">Hover over any Nova target, Meteor Radiant, or Uncataloged Transient for J2000 coordinates</span>
    </h2>
    <div class="sky-canvas-wrap">
      <canvas id="celestialCanvas" width="1440" height="540"></canvas>
    </div>
    <div class="legend-bar">
      <span>⭐ <strong style="color:#fbbf24;">Gold Star:</strong> Recurrent Nova / Supernova Watchlist Target</span>
      <span>◆ <strong style="color:#38bdf8;">Cyan Diamond:</strong> Active IAU Meteor Shower Radiant</span>
      <span>◎ <strong style="color:#e879f9;">Magenta Reticle:</strong> Candidate Uncataloged Radiant Convergence</span>
      <span>╱ <strong style="color:#34d399;">Green Great-Circle Arc:</strong> Verified Meteor & Back-Projected Radiant Path</span>
      <span>● <strong style="color:#fb7185;">Rose Circle:</strong> Uncataloged Deep-Shadow Optical Transient (Click to view frame)</span>
      <span>┆ <strong style="color:#6ee7b7;">Dashed Line:</strong> Current Local Sidereal Meridian (Overhead at BluffSky)</span>
    </div>
  </div>

  <div class="two-col">
    <!-- Left: Recurrent Nova & Extragalactic Supernova Watchlist -->
    <div class="card">
      <h2>
        <span>🌟 Live Recurrent Nova, Symbiotic Binary & Supernova Watchlist</span>
        <span style="font-size:0.78rem;color:var(--muted);">Real-time dome pixel & altitude monitoring</span>
      </h2>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>Target</th>
              <th>Type</th>
              <th>J2000 (RA / Dec)</th>
              <th>Quiescent → Peak</th>
              <th>Live Alt / Az</th>
              <th>Dome Status</th>
            </tr>
          </thead>
          <tbody id="watchlistBody"></tbody>
        </table>
      </div>
    </div>

    <!-- Right: Meteor Shower Radiants & Candidate Stream Convergences -->
    <div class="card">
      <h2>
        <span>☄️ Meteor Shower Radiants & Stream Convergence Patrol</span>
        <span style="font-size:0.78rem;color:var(--muted);">IAU Showers + Pairwise Great-Circle Clusters</span>
      </h2>
      <div id="showerPills" style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;"></div>
      <div style="font-size:0.82rem;font-weight:700;color:#e9d5ff;margin-bottom:8px;">
        ◎ Candidate Uncataloged Radiant Convergences (Multi-Night Great-Circle Intersections):
      </div>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>Radiant J2000 (RA, Dec)</th>
              <th>Intersecting Tracks</th>
              <th>Stream Assessment</th>
            </tr>
          </thead>
          <tbody id="convergenceBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- High-Priority Uncataloged Optical Transients & Deep-Shadow Candidates -->
  <div class="card">
    <h2>
      <span>🔬 High-Priority Uncataloged Optical Transients & Deep-Space Candidates</span>
      <span style="font-size:0.8rem;color:var(--muted);">Clear-sky optical flashes & streaks with zero match in CelesTrak/McCants, ranked by Earth Umbral Shadow Height</span>
    </h2>
    <div id="transientsGrid" class="transients-grid"></div>
  </div>
</div>

<!-- Zoomable Snapshot & Interactive Blink Comparator Modal -->
<div id="snapModal" onclick="if(event.target.id==='snapModal') closeSnapModal()">
  <div class="m-top" style="flex-wrap:wrap;gap:10px;">
    <div>
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
        <div id="mTitle" style="font-weight:700;font-size:1rem;"></div>
        <span id="mBlinkPhasePill" style="display:none;padding:3px 10px;border-radius:99px;font-size:0.74rem;font-weight:800;font-family:monospace;"></span>
      </div>
      <div id="mSub" class="mono" style="margin-top:2px;"></div>
    </div>
    <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
      <div id="mBlinkControls" style="display:none;gap:5px;align-items:center;background:#090f1f;padding:4px 8px;border-radius:8px;border:1px solid #334155;">
        <button class="btn" id="mBtnBlinkAuto" style="padding:4px 10px;font-size:0.75rem;" onclick="setModalBlinkMode('auto')">⚡ Live Blink (Ref ⇄ Night)</button>
        <button class="btn secondary" id="mBtnBlinkRef" style="padding:4px 9px;font-size:0.75rem;" onclick="setModalBlinkMode('ref')">1️⃣ Reference</button>
        <button class="btn secondary" id="mBtnBlinkSci" style="padding:4px 9px;font-size:0.75rem;" onclick="setModalBlinkMode('sci')">2️⃣ Discovery Night</button>
        <button class="btn secondary" id="mBtnBlinkTriplet" style="padding:4px 9px;font-size:0.75rem;" onclick="setModalBlinkMode('triplet')">🖼️ 3-Panel Triplet</button>
      </div>
      <div id="mTeachBar" style="display:none;gap:4px;align-items:center;flex-wrap:wrap;background:#090f1f;padding:4px 8px;border-radius:8px;border:1px solid #334155;">
        <span style="font-size:0.74rem;color:#94a3b8;margin-right:3px;font-weight:700;">🧠 Reclassify / Teach AI:</span>
        <button class="teach-btn" onclick="reclassifyActivePatrolModal('meteor')" title="Reclassify as Shooting Star / Meteor">☄️ Meteor</button>
        <button class="teach-btn" onclick="reclassifyActivePatrolModal('fireball')" title="Reclassify as Fireball / Bolide">🔥 Fireball</button>
        <button class="teach-btn" onclick="reclassifyActivePatrolModal('space_object')" title="Reclassify as Satellite / Space Object">🛰️ Sat</button>
        <button class="teach-btn" onclick="reclassifyActivePatrolModal('aircraft')" title="Reclassify as Aircraft (Removes from J2000 Transients)">✈️ Plane</button>
        <button class="teach-btn" onclick="reclassifyActivePatrolModal('transient_flash')" title="Confirm as Optical Space Flash">✨ Flash</button>
        <button class="teach-btn false-btn" onclick="reclassifyActivePatrolModal('false_positive')" title="Mark as False Image / Artifact (Trains AI & Removes from Map)">❌ False Image</button>
      </div>
      <button class="btn secondary" onclick="zoomSnap(0.75)" title="Zoom Out">➖</button>
      <button class="btn secondary" id="mZoomLbl" onclick="resetSnapZoom()" title="Reset Zoom to 100%">100%</button>
      <button class="btn secondary" onclick="zoomSnap(1.35)" title="Zoom In">➕</button>
      <button class="btn secondary" id="mFocusBtn" style="display:none;color:#34d399;border-color:rgba(52,211,153,0.45);" onclick="focusSnapBbox()" title="Zoom directly into the detected target box (Key: F)">🎯 Focus Target</button>
      <button class="btn secondary" onclick="closeSnapModal()">✕ Close</button>
    </div>
  </div>
  <div class="m-stage" id="mStage">
    <button id="mPrevBtn" class="m-nav-arrow prev" onclick="event.stopPropagation(); stepTransientModal(-1)" title="Previous Transient (Left Arrow)">❮</button>
    <img id="mImg" src="" draggable="false" />
    <button id="mNextBtn" class="m-nav-arrow next" onclick="event.stopPropagation(); stepTransientModal(1)" title="Next Transient (Right Arrow)">❯</button>
  </div>
</div>

<script>
let PATROL_DATA = null;
let INTERACTIVE_POINTS = [];
let DEEP_BLINK_ITEMS = [];
let TRANSIENT_SNAP_ITEMS = [];
let MODAL_BLINK_ITEM = null;
let MODAL_BLINK_TIMER = null;
let MODAL_BLINK_PHASE = false;
let MODAL_BLINK_MODE = 'triplet';
let MODAL_ACTIVE_FRAME_ID = '';
let MODAL_ACTIVE_BBOX = null;
let MODAL_TRANSIENT_IDX = -1;
let SNAP_ZOOM = 1.0, SNAP_X = 0, SNAP_Y = 0, SNAP_DRAG = false, SNAP_SX = 0, SNAP_SY = 0;

function stopModalBlinkTimer() {
  if (MODAL_BLINK_TIMER) {
    clearInterval(MODAL_BLINK_TIMER);
    MODAL_BLINK_TIMER = null;
  }
}

function applySnapTransform() {
  const img = document.getElementById('mImg');
  img.style.transform = `translate(${SNAP_X}px, ${SNAP_Y}px) scale(${SNAP_ZOOM})`;
  document.getElementById('mZoomLbl').textContent = `${Math.round(SNAP_ZOOM * 100)}%`;
}
function resetSnapZoom() { SNAP_ZOOM = 1.0; SNAP_X = 0; SNAP_Y = 0; applySnapTransform(); }
function zoomSnap(f) { SNAP_ZOOM = Math.max(0.1, Math.min(40.0, SNAP_ZOOM * f)); applySnapTransform(); }

function updateModalBlinkFrame() {
  if (!MODAL_BLINK_ITEM) return;
  const img = document.getElementById('mImg');
  const pill = document.getElementById('mBlinkPhasePill');
  const bust = MODAL_BLINK_ITEM.cacheBust || '';
  if (MODAL_BLINK_MODE === 'triplet') {
    img.onerror = null;
    img.src = MODAL_BLINK_ITEM.tripletUrl + bust;
    pill.style.display = 'inline-block';
    pill.style.background = 'rgba(56,189,248,0.18)';
    pill.style.color = '#7dd3fc';
    pill.style.border = '1px solid rgba(56,189,248,0.45)';
    pill.textContent = '🖼️ 3-PANEL TRIPLET (REF + DISCOVERY + ΔI)';
  } else {
    const showSci = (MODAL_BLINK_MODE === 'sci') || (MODAL_BLINK_MODE === 'auto' && MODAL_BLINK_PHASE);
    const targetUrl = (showSci ? MODAL_BLINK_ITEM.sciUrl : MODAL_BLINK_ITEM.refUrl) + bust;
    img.onerror = () => {
      stopModalBlinkTimer();
      MODAL_BLINK_MODE = 'triplet';
      img.onerror = null;
      img.src = MODAL_BLINK_ITEM.tripletUrl + bust;
    };
    img.src = targetUrl;
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
    const btn = document.getElementById('mBtnBlink' + m.charAt(0).toUpperCase() + m.slice(1));
    if (btn) btn.className = (MODAL_BLINK_MODE === m) ? 'btn' : 'btn secondary';
  });
}

function setModalBlinkMode(mode) {
  stopModalBlinkTimer();
  MODAL_BLINK_MODE = mode;
  if (mode === 'auto') {
    MODAL_BLINK_PHASE = true;
    updateModalBlinkFrame();
    MODAL_BLINK_TIMER = setInterval(() => {
      MODAL_BLINK_PHASE = !MODAL_BLINK_PHASE;
      updateModalBlinkFrame();
    }, 650);
  } else {
    updateModalBlinkFrame();
  }
}

function openBlinkItem(idx, startMode) {
  const item = DEEP_BLINK_ITEMS[idx];
  if (!item) return;
  stopModalBlinkTimer();
  MODAL_ACTIVE_FRAME_ID = '';
  MODAL_ACTIVE_BBOX = null;
  MODAL_TRANSIENT_IDX = -1;
  document.getElementById('mTeachBar').style.display = 'none';
  document.getElementById('mFocusBtn').style.display = 'none';
  document.getElementById('mPrevBtn').style.display = 'none';
  document.getElementById('mNextBtn').style.display = 'none';
  const bust = item.versionTag ? `?v=${encodeURIComponent(item.versionTag)}` : '';
  MODAL_BLINK_ITEM = Object.assign({}, item, { cacheBust: bust });
  const preRef = new Image(); preRef.src = item.refUrl + bust;
  const preSci = new Image(); preSci.src = item.sciUrl + bust;
  resetSnapZoom();
  document.getElementById('mTitle').textContent = item.title;
  document.getElementById('mSub').textContent = item.sub + ' • Press SPACE to flip Reference ⇄ Discovery Night';
  document.getElementById('mBlinkControls').style.display = 'inline-flex';
  document.getElementById('snapModal').classList.add('open');
  setModalBlinkMode(startMode || 'auto');
}

function focusSnapBbox() {
  const img = document.getElementById('mImg');
  if (!MODAL_ACTIVE_BBOX || MODAL_ACTIVE_BBOX.length !== 4 || !img) {
    zoomSnap(2.0);
    return;
  }
  const w = img.clientWidth || 720;
  const h = img.clientHeight || 720;
  const cxNorm = ((MODAL_ACTIVE_BBOX[0] + MODAL_ACTIVE_BBOX[2]) * 0.5) / 3552.0 - 0.5;
  const cyNorm = ((MODAL_ACTIVE_BBOX[1] + MODAL_ACTIVE_BBOX[3]) * 0.5) / 3552.0 - 0.5;
  const targetZoom = (SNAP_ZOOM >= 3.2 && SNAP_ZOOM < 18.0) ? Math.min(40.0, SNAP_ZOOM * 1.8) : 4.0;
  SNAP_ZOOM = targetZoom;
  SNAP_X = -(cxNorm * w) * targetZoom;
  SNAP_Y = -(cyNorm * h) * targetZoom;
  applySnapTransform();
}

function openTransientSnapIdx(idx) {
  if (idx < 0 || idx >= TRANSIENT_SNAP_ITEMS.length) return;
  MODAL_TRANSIENT_IDX = idx;
  const item = TRANSIENT_SNAP_ITEMS[idx];
  const verifiedTag = item.userCorrected ? ' • ✅ Human Verified' : '';
  const navTag = TRANSIENT_SNAP_ITEMS.length > 1 ? ` [${idx + 1} of ${TRANSIENT_SNAP_ITEMS.length}]` : '';
  openEventSnapModal(
    item.title + verifiedTag,
    `${item.sub} • Frame: ${item.frameId}${navTag}`,
    item.snapshot,
    item.frameId,
    item.bbox,
    TRANSIENT_SNAP_ITEMS.length > 1
  );
}

function stepTransientModal(delta) {
  if (!TRANSIENT_SNAP_ITEMS.length || MODAL_TRANSIENT_IDX < 0) return;
  const nextIdx = (MODAL_TRANSIENT_IDX + delta + TRANSIENT_SNAP_ITEMS.length) % TRANSIENT_SNAP_ITEMS.length;
  openTransientSnapIdx(nextIdx);
}

function openEventSnapModal(title, sub, snapFn, frameId, bbox, showTransientNav) {
  if (!snapFn) return;
  openUrlModal(title, sub, `/snapshots/${encodeURIComponent(snapFn)}`);
  MODAL_ACTIVE_FRAME_ID = frameId || '';
  MODAL_ACTIVE_BBOX = (Array.isArray(bbox) && bbox.length === 4) ? bbox : null;
  document.getElementById('mTeachBar').style.display = MODAL_ACTIVE_FRAME_ID ? 'inline-flex' : 'none';
  document.getElementById('mFocusBtn').style.display = MODAL_ACTIVE_BBOX ? 'inline-flex' : 'none';
  document.getElementById('mPrevBtn').style.display = showTransientNav ? 'flex' : 'none';
  document.getElementById('mNextBtn').style.display = showTransientNav ? 'flex' : 'none';
}

function openSnapModal(title, sub, snapFn) {
  if (!snapFn) return;
  MODAL_TRANSIENT_IDX = -1;
  openEventSnapModal(title, sub, snapFn, '', null, false);
}
function openUrlModal(title, sub, url) {
  if (!url) return;
  stopModalBlinkTimer();
  MODAL_BLINK_ITEM = null;
  MODAL_ACTIVE_FRAME_ID = '';
  MODAL_ACTIVE_BBOX = null;
  document.getElementById('mBlinkControls').style.display = 'none';
  document.getElementById('mBlinkPhasePill').style.display = 'none';
  document.getElementById('mTeachBar').style.display = 'none';
  document.getElementById('mFocusBtn').style.display = 'none';
  document.getElementById('mPrevBtn').style.display = 'none';
  document.getElementById('mNextBtn').style.display = 'none';
  resetSnapZoom();
  document.getElementById('mTitle').textContent = title;
  document.getElementById('mSub').textContent = sub;
  document.getElementById('mImg').onerror = null;
  document.getElementById('mImg').src = url;
  document.getElementById('snapModal').classList.add('open');
}
function closeSnapModal() {
  stopModalBlinkTimer();
  MODAL_TRANSIENT_IDX = -1;
  MODAL_ACTIVE_FRAME_ID = '';
  document.getElementById('snapModal').classList.remove('open');
}

function showPatrolToast(msg, isOk) {
  let t = document.getElementById('patrolTeachToast');
  if (!t) {
    t = document.createElement('div');
    t.id = 'patrolTeachToast';
    t.style.cssText = 'position:fixed;bottom:22px;right:22px;z-index:99999;padding:11px 16px;border-radius:10px;font-size:0.84rem;font-weight:700;box-shadow:0 12px 32px rgba(0,0,0,0.75);transition:opacity 0.25s ease;max-width:540px;';
    document.body.appendChild(t);
  }
  t.style.background = isOk ? 'rgba(6,78,59,0.96)' : 'rgba(127,29,29,0.96)';
  t.style.border = isOk ? '1px solid #34d399' : '1px solid #f87171';
  t.style.color = isOk ? '#a7f3d0' : '#fecaca';
  t.textContent = msg;
  t.style.opacity = '1';
  t.style.display = 'block';
  clearTimeout(window._patrolToastTimer);
  window._patrolToastTimer = setTimeout(() => {
    t.style.opacity = '0';
    setTimeout(() => { t.style.display = 'none'; }, 260);
  }, 4200);
}

async function reclassifyPatrolEvent(frameFilename, newCategory, domEvent) {
  if (domEvent) domEvent.stopPropagation();
  if (!frameFilename) return;
  const prevTransientIdx = MODAL_TRANSIENT_IDX;
  const wasModalOpen = document.getElementById('snapModal').classList.contains('open');
  showPatrolToast(`🧠 Teaching AI: Reclassifying ${frameFilename} → ${newCategory}...`, true);
  try {
    const res = await fetch('/api/reclassify', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        frame_filename: frameFilename,
        new_category: newCategory,
        note: `J2000 Celestial Sphere Map 1-click correction to ${newCategory}`
      })
    });
    const data = await res.json();
    showPatrolToast(data.detail || 'Learned correction!', res.ok);
    await loadDiscoveryData();
    if (wasModalOpen) {
      if (prevTransientIdx >= 0) {
        const sameIdx = TRANSIENT_SNAP_ITEMS.findIndex(x => x.frameId === frameFilename);
        if (sameIdx >= 0) {
          openTransientSnapIdx(sameIdx);
        } else if (TRANSIENT_SNAP_ITEMS.length > 0) {
          openTransientSnapIdx(Math.min(prevTransientIdx, TRANSIENT_SNAP_ITEMS.length - 1));
        } else {
          closeSnapModal();
        }
      } else if (newCategory === 'false_positive' || newCategory === 'aircraft') {
        closeSnapModal();
      }
    }
  } catch (err) {
    showPatrolToast(`Failed saving reclassification: ${err}`, false);
  }
}

async function reclassifyActivePatrolModal(newCategory) {
  if (!MODAL_ACTIVE_FRAME_ID) return;
  await reclassifyPatrolEvent(MODAL_ACTIVE_FRAME_ID, newCategory, null);
}

window.addEventListener('keydown', (e) => {
  const modalOpen = document.getElementById('snapModal').classList.contains('open');
  if (!modalOpen) return;
  if (e.key === 'Escape') {
    closeSnapModal();
  } else if (e.key === 'ArrowLeft' && MODAL_TRANSIENT_IDX >= 0) {
    stepTransientModal(-1);
  } else if (e.key === 'ArrowRight' && MODAL_TRANSIENT_IDX >= 0) {
    stepTransientModal(1);
  } else if ((e.key === 'f' || e.key === 'F') && MODAL_ACTIVE_BBOX) {
    focusSnapBbox();
  } else if (e.code === 'Space' && MODAL_BLINK_ITEM) {
    e.preventDefault();
    if (MODAL_BLINK_MODE === 'ref') {
      setModalBlinkMode('sci');
    } else {
      setModalBlinkMode('ref');
    }
  }
});

(function initSnapZoom() {
  const st = document.getElementById('mStage');
  st.addEventListener('wheel', (e) => {
    e.preventDefault();
    zoomSnap(e.deltaY < 0 ? 1.2 : 0.83);
  }, { passive: false });
  st.addEventListener('mousedown', (e) => {
    SNAP_DRAG = true; SNAP_SX = e.clientX - SNAP_X; SNAP_SY = e.clientY - SNAP_Y;
    st.classList.add('dragging');
  });
  window.addEventListener('mousemove', (e) => {
    if (!SNAP_DRAG) return;
    SNAP_X = e.clientX - SNAP_SX; SNAP_Y = e.clientY - SNAP_SY;
    applySnapTransform();
  });
  window.addEventListener('mouseup', () => { SNAP_DRAG = false; st.classList.remove('dragging'); });
})();

function raDecToCanvas(raDeg, decDeg, w, h) {
  const padL = 54, padR = 24, padT = 26, padB = 34;
  const pw = w - padL - padR;
  const ph = h - padT - padB;
  // Standard astronomical chart: RA 360 (24h) on left -> 0h on right; Dec -30° at bottom -> +90° at top
  const x = padL + ((360.0 - (raDeg % 360.0)) / 360.0) * pw;
  const y = padT + ((90.0 - Math.max(-30.0, Math.min(90.0, decDeg))) / 120.0) * ph;
  return [x, y];
}

function drawCelestialMap(data) {
  const canvas = document.getElementById('celestialCanvas');
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  INTERACTIVE_POINTS = [];

  // Background gradient
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, '#050a18');
  grad.addColorStop(1, '#091326');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  // RA / Dec Grid
  ctx.strokeStyle = 'rgba(51, 65, 85, 0.45)';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#64748b';
  ctx.font = '11px Inter, sans-serif';

  for (let raH = 0; raH <= 24; raH += 2) {
    const [x, y0] = raDecToCanvas(raH * 15, 90, w, h);
    const [, y1] = raDecToCanvas(raH * 15, -30, w, h);
    ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
    ctx.fillText(`${raH}h`, x - 8, y1 + 16);
  }
  for (let dec = -30; dec <= 90; dec += 30) {
    const [x0, y] = raDecToCanvas(360, dec, w, h);
    const [x1, ] = raDecToCanvas(0, dec, w, h);
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
    ctx.fillText(`${dec >= 0 ? '+' : ''}${dec}°`, 10, y + 4);
  }

  // Local Sidereal Time meridian
  const lstDeg = (data.observatory && data.observatory.lst_deg) || 0;
  const [lstX, ly0] = raDecToCanvas(lstDeg, 90, w, h);
  const [, ly1] = raDecToCanvas(lstDeg, -30, w, h);
  ctx.save();
  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = 'rgba(52, 211, 153, 0.65)';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(lstX, ly0); ctx.lineTo(lstX, ly1); ctx.stroke();
  ctx.fillStyle = '#6ee7b7';
  ctx.fillText(`LST Zenith (${data.observatory.lst_hms})`, Math.min(w - 160, Math.max(60, lstX + 6)), ly0 + 14);
  ctx.restore();

  // 1. Verified Meteor Tracks & Great-Circle Back-projection arcs
  const mp = data.meteor_patrol || {};
  (mp.meteor_tracks || []).forEach(t => {
    const [x1, y1] = raDecToCanvas(t.start_radec[0], t.start_radec[1], w, h);
    const [x2, y2] = raDecToCanvas(t.end_radec[0], t.end_radec[1], w, h);
    if (Math.abs(x1 - x2) < w * 0.45) {
      ctx.strokeStyle = t.color || '#34d399';
      ctx.lineWidth = 2.0;
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }
    const [mx, my] = raDecToCanvas(t.ra_deg, t.dec_deg, w, h);
    INTERACTIVE_POINTS.push({
      x: mx, y: my, r: 7,
      label: `☄️ ${t.subject} [${t.shower_name}] • J2000 ${t.ra_hms} ${t.dec_dms} • SNR ${t.snr}σ`,
      snapshot: t.snapshot, title: t.subject,
      frameId: t.id, bbox: t.bbox
    });
  });

  // 2. Active IAU Meteor Shower Radiants
  (mp.iau_showers || []).forEach(sh => {
    const [x, y] = raDecToCanvas(sh.ra_deg, sh.dec_deg, w, h);
    ctx.fillStyle = sh.color || '#38bdf8';
    ctx.beginPath();
    ctx.moveTo(x, y - 7); ctx.lineTo(x + 7, y); ctx.lineTo(x, y + 7); ctx.lineTo(x - 7, y);
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#e0f2fe';
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.fillText(sh.code, x + 9, y + 4);
    INTERACTIVE_POINTS.push({
      x, y, r: 10,
      label: `◆ IAU Radiant: ${sh.name} (${sh.code}) • Parent: ${sh.parent_body} • V∞=${sh.v_inf_kms} km/s`
    });
  });

  // 3. Candidate Uncataloged Radiant Convergences
  (mp.radiant_convergences || []).forEach(rc => {
    const [x, y] = raDecToCanvas(rc.ra_deg, rc.dec_deg, w, h);
    ctx.strokeStyle = '#e879f9';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fillStyle = '#f0abfc'; ctx.fill();
    INTERACTIVE_POINTS.push({
      x, y, r: 11,
      label: `◎ Radiant Convergence (${rc.meteor_count} tracks) • J2000 ${rc.ra_hms} ${rc.dec_dms} • ${rc.matched_iau_name}`
    });
  });

  // 4. Uncataloged Deep-Shadow Optical Transients
  (data.uncataloged_transients || []).forEach((tr, trIdx) => {
    const [x, y] = raDecToCanvas(tr.ra_deg, tr.dec_deg, w, h);
    ctx.fillStyle = tr.priority === 'HIGH' ? '#fb7185' : '#38bdf8';
    ctx.beginPath(); ctx.arc(x, y, 5.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke();
    INTERACTIVE_POINTS.push({
      x, y, r: 9,
      label: `🔬 ${tr.discovery_class} • J2000 ${tr.ra_hms} ${tr.dec_dms} • Earth Shadow ${Number(tr.shadow_height_km).toLocaleString()} km`,
      snapshot: tr.snapshot, title: tr.discovery_class,
      frameId: tr.id, bbox: tr.bbox, trIdx: trIdx
    });
  });

  // 5. Recurrent Nova & Supernova Watchlist Targets
  (data.watchlist || []).forEach(wItem => {
    const [x, y] = raDecToCanvas(wItem.ra_deg, wItem.dec_deg, w, h);
    ctx.fillStyle = wItem.in_dome ? '#fbbf24' : '#64748b';
    ctx.beginPath(); ctx.arc(x, y, wItem.priority === 'CRITICAL' ? 7 : 5.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#fef08a'; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.fillStyle = wItem.in_dome ? '#fde68a' : '#94a3b8';
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.fillText(wItem.id.replace('_', ' '), x + 8, y - 5);
    INTERACTIVE_POINTS.push({
      x, y, r: 10,
      label: `⭐ ${wItem.name} (${wItem.type}) • Mag ${wItem.quiescent_mag}→${wItem.eruption_mag} • Alt ${wItem.alt_deg}° (${wItem.visibility_status})`
    });
  });
}

(function bindCanvasEvents() {
  const canvas = document.getElementById('celestialCanvas');
  const readout = document.getElementById('hoverReadout');
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const sx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const sy = (e.clientY - rect.top) * (canvas.height / rect.height);
    let hit = null, bestD = 999;
    for (const pt of INTERACTIVE_POINTS) {
      const d = Math.hypot(sx - pt.x, sy - pt.y);
      if (d <= pt.r + 6 && d < bestD) { bestD = d; hit = pt; }
    }
    if (hit) {
      readout.textContent = hit.label + (hit.snapshot ? ' [Click to inspect & reclassify]' : '');
      canvas.style.cursor = hit.snapshot ? 'pointer' : 'crosshair';
    } else {
      canvas.style.cursor = 'crosshair';
    }
  });
  canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    const sx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const sy = (e.clientY - rect.top) * (canvas.height / rect.height);
    let hit = null, bestD = 999;
    for (const pt of INTERACTIVE_POINTS) {
      if (!pt.snapshot) continue;
      const d = Math.hypot(sx - pt.x, sy - pt.y);
      if (d <= pt.r + 6 && d < bestD) { bestD = d; hit = pt; }
    }
    if (hit) {
      if (typeof hit.trIdx === 'number' && hit.trIdx >= 0) {
        openTransientSnapIdx(hit.trIdx);
      } else {
        MODAL_TRANSIENT_IDX = -1;
        openEventSnapModal(hit.title || 'Sky Discovery Candidate', hit.label, hit.snapshot, hit.frameId || '', hit.bbox || null, false);
      }
    }
  });
})();

async function loadDiscoveryData() {
  const res = await fetch('/api/discoveries');
  const data = await res.json();
  PATROL_DATA = data;

  const obs = data.observatory || {};
  document.getElementById('lstPill').textContent = `🌌 LST: ${obs.lst_hms || '--'}`;
  document.getElementById('moonInfoPill').textContent = `🌕 Moon: ${obs.moon_alt_deg}° (${obs.moon_phase_pct || 0}%)`;
  document.getElementById('kpiLst').textContent = obs.lst_hms || '--';

  const wl = data.watchlist || [];
  const inDomeCnt = wl.filter(x => x.in_dome).length;
  document.getElementById('kpiWatchlist').textContent = `${inDomeCnt} / ${wl.length} Active`;

  const mp = data.meteor_patrol || {};
  document.getElementById('kpiMeteors').textContent = mp.total_verified_meteors || 0;
  document.getElementById('kpiClusters').textContent = (mp.radiant_convergences || []).length;

  const trs = data.uncataloged_transients || [];
  document.getElementById('kpiTransients').textContent = trs.length;

  TRANSIENT_SNAP_ITEMS = trs.map(t => ({
    title: t.discovery_class,
    sub: `J2000 ${t.ra_hms} ${t.dec_dms} • Gal (${t.gal_l_deg}°, ${t.gal_b_deg}°) • Shadow ${Number(t.shadow_height_km).toLocaleString()} km`,
    snapshot: t.snapshot,
    frameId: t.id,
    bbox: t.bbox,
    userCorrected: t.user_corrected_category || ''
  }));

  drawCelestialMap(data);

  // Populate Watchlist Table
  const wBody = document.getElementById('watchlistBody');
  wBody.innerHTML = wl.map(w => {
    const pBadge = w.priority === 'CRITICAL' ? 'crit' : (w.priority === 'HIGH' ? 'high' : 'med');
    const dBadge = w.in_dome ? 'dome' : 'low';
    const pxStr = w.pixel ? `(${w.pixel[0]}, ${w.pixel[1]})` : '--';
    return `<tr title="${w.notes}">
      <td><strong>${w.name}</strong><br><span style="font-size:0.74rem;color:var(--muted);">${w.constellation}</span></td>
      <td><span class="badge ${pBadge}">${w.priority}</span><br><span style="font-size:0.74rem;color:#cbd5e1;">${w.type}</span></td>
      <td class="mono">${w.ra_str}<br>${w.dec_str}</td>
      <td>Mag ${w.quiescent_mag} → <strong style="color:#fbbf24;">${w.eruption_mag}</strong></td>
      <td>${w.alt_deg}° ${w.cardinal}<br><span style="font-size:0.72rem;color:var(--muted);">px ${pxStr}</span></td>
      <td><span class="badge ${dBadge}">${w.visibility_status}</span></td>
    </tr>`;
  }).join('');

  // Populate Shower Pills & Convergences
  const sc = mp.shower_counts || {};
  const shPills = document.getElementById('showerPills');
  shPills.innerHTML = (mp.iau_showers || []).map(s =>
    `<span class="pill" style="border-color:${s.color};">◆ <strong>${s.name} (${s.code}):</strong> ${sc[s.code] || 0}</span>`
  ).join('') + `<span class="pill">✨ <strong>Sporadic / Uncataloged:</strong> ${sc.SPORADIC || 0}</span>`;

  const cBody = document.getElementById('convergenceBody');
  cBody.innerHTML = (mp.radiant_convergences || []).map(rc => `
    <tr>
      <td class="mono"><strong>${rc.ra_hms}</strong><br>${rc.dec_dms}</td>
      <td><span class="badge high">${rc.meteor_count} Intersecting Tracks</span></td>
      <td><strong>${rc.matched_iau_name}</strong><br><span style="font-size:0.74rem;color:var(--muted);">${rc.parent_body}</span></td>
    </tr>
  `).join('') || '<tr><td colspan="3" style="color:var(--muted);">Accumulating multi-night meteor tracks...</td></tr>';

  // Populate Uncataloged Transients Grid
  const tGrid = document.getElementById('transientsGrid');
  tGrid.innerHTML = trs.map((t, idx) => {
    const pBadge = t.priority === 'HIGH' ? 'high' : 'med';
    const verifiedBadge = t.user_corrected_category ? `<span class="badge dome">✅ Verified</span>` : '';
    return `<div class="t-card" onclick="openTransientSnapIdx(${idx})">
      <img src="/snapshots/${encodeURIComponent(t.snapshot || '')}" loading="lazy" />
      <div class="t-body">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap;">
          <div style="display:flex;gap:5px;align-items:center;">
            <span class="badge ${pBadge}">${t.priority} PRIORITY</span>
            ${verifiedBadge}
          </div>
          <span style="font-size:0.74rem;color:var(--muted);">${t.timestamp || ''}</span>
        </div>
        <div style="font-weight:700;font-size:0.92rem;color:#f8fafc;">${t.discovery_class}</div>
        <div class="mono">J2000: ${t.ra_hms} ${t.dec_dms}</div>
        <div class="mono" style="color:#6ee7b7;">🌍 Earth Shadow Exit: ${Number(t.shadow_height_km).toLocaleString()} km • SNR ${Number(t.snr).toFixed(1)}σ</div>
        <div style="font-size:0.76rem;color:#cbd5e1;margin-top:2px;">${t.scientific_assessment}</div>
        <div style="display:flex;gap:4px;flex-wrap:wrap;align-items:center;margin-top:6px;padding-top:7px;border-top:1px dashed #334155;">
          <span style="font-size:0.71rem;color:#94a3b8;margin-right:2px;">🧠 Teach:</span>
          <button class="teach-btn" style="padding:2px 6px;font-size:0.71rem;" onclick="reclassifyPatrolEvent('${t.id}', 'meteor', event)" title="Reclassify as Shooting Star / Meteor">☄️ Meteor</button>
          <button class="teach-btn" style="padding:2px 6px;font-size:0.71rem;" onclick="reclassifyPatrolEvent('${t.id}', 'fireball', event)" title="Reclassify as Fireball / Bolide">🔥 Fireball</button>
          <button class="teach-btn" style="padding:2px 6px;font-size:0.71rem;" onclick="reclassifyPatrolEvent('${t.id}', 'space_object', event)" title="Reclassify as Satellite / Space">🛰️ Sat</button>
          <button class="teach-btn" style="padding:2px 6px;font-size:0.71rem;" onclick="reclassifyPatrolEvent('${t.id}', 'aircraft', event)" title="Reclassify as Aircraft">✈️ Plane</button>
          <button class="teach-btn" style="padding:2px 6px;font-size:0.71rem;" onclick="reclassifyPatrolEvent('${t.id}', 'transient_flash', event)" title="Confirm as Optical Space Flash">✨ Flash</button>
          <button class="teach-btn false-btn" style="padding:2px 6px;font-size:0.71rem;" onclick="reclassifyPatrolEvent('${t.id}', 'false_positive', event)" title="Mark as False Image (Trains AI & Removes from Map)">❌ False</button>
        </div>
      </div>
    </div>`;
  }).join('');
}

async function loadDeepSurveyData() {
  try {
    const res = await fetch('/api/deep-survey');
    const data = await res.json();
    const ov = data.survey_overview;
    if (!ov) return;

    document.getElementById('deepSurveyMeta').textContent =
      `Stacked ${ov.clear_nights_stacked} Clear Nights (${ov.total_sidereal_frames_coadded} Frames) • Depth Gain ${ov.limiting_magnitude_gain_mag} (Est. Limit Mag +${ov.estimated_deep_stack_limit_mag})`;

    document.getElementById('deepSurveyKpis').innerHTML = `
      <div class="kpi-card" style="padding:10px 12px;">
        <div class="lbl">Clear Nights Stacked</div>
        <div class="val" style="font-size:1.25rem;color:#38bdf8;">${ov.clear_nights_stacked} / ${ov.total_local_nights_scanned}</div>
        <div class="sub">${ov.total_local_night_photos.toLocaleString()} local NVMe night photos</div>
      </div>
      <div class="kpi-card" style="padding:10px 12px;">
        <div class="lbl">Sidereal Co-Added Frames</div>
        <div class="val" style="font-size:1.25rem;color:#34d399;">${ov.total_sidereal_frames_coadded}</div>
        <div class="sub">${ov.grid_resolution}</div>
      </div>
      <div class="kpi-card" style="padding:10px 12px;">
        <div class="lbl">Limiting Magnitude Boost</div>
        <div class="val" style="font-size:1.25rem;color:#fbbf24;">${ov.limiting_magnitude_gain_mag}</div>
        <div class="sub">Reaches Mag +${ov.estimated_deep_stack_limit_mag} (Uranus Mag +5.61)</div>
      </div>
      <div class="kpi-card" style="padding:10px 12px;">
        <div class="lbl">Solar System Recoveries</div>
        <div class="val" style="font-size:1.25rem;color:#6ee7b7;">${(data.solar_system_recoveries || []).length} Bodies</div>
        <div class="sub">Multi-night moving-body calibration</div>
      </div>
      <div class="kpi-card" style="padding:10px 12px;">
        <div class="lbl">Candidate Moving Tracklets</div>
        <div class="val" style="font-size:1.25rem;color:#e879f9;">${(data.candidate_moving_tracklets || []).length} Tracklets</div>
        <div class="sub">3-Night collinear sky motion</div>
      </div>
      <div class="kpi-card" style="padding:10px 12px;">
        <div class="lbl">Multi-Frame Transients</div>
        <div class="val" style="font-size:1.25rem;color:#fb7185;">${(data.multi_frame_optical_transients || []).length} Candidates</div>
        <div class="sub">Verified across ≥4 frames/night</div>
      </div>
    `;

    const surveyVer = data.generated_at || '1';
    if (ov.master_atlas_url) {
      document.getElementById('masterAtlasImg').src = `${ov.master_atlas_url}?v=${encodeURIComponent(surveyVer)}`;
    }

    DEEP_BLINK_ITEMS = [];
    function registerBlinkItem(title, sub, cutoutUrl, badge, bClass) {
      const base = (cutoutUrl || '').replace(/\.jpg$/i, '');
      const idx = DEEP_BLINK_ITEMS.length;
      DEEP_BLINK_ITEMS.push({
        idx,
        title,
        sub,
        tripletUrl: cutoutUrl,
        refUrl: `${base}_ref.jpg`,
        sciUrl: `${base}_sci.jpg`,
        badge,
        bClass,
        versionTag: surveyVer
      });
      return idx;
    }

    const sol = data.solar_system_recoveries || [];
    document.getElementById('solsysBody').innerHTML = sol.map(s => {
      const bIdx = registerBlinkItem(
        `🪐 Solar System Recovery: ${s.name} (${s.body_type}, Mag ${s.mag >= 0 ? '+' : ''}${s.mag})`,
        `${s.nights_detected} nights (${s.first_night}→${s.last_night}) • Motion ${s.Angular_speed_deg_day}°/day • J2000 ${s.latest_ra_hms} ${s.latest_dec_dms} • Peak SNR ${s.peak_snr}σ`,
        s.cutout_url,
        'VERIFIED PLANET / ICE GIANT',
        'dome'
      );
      return `
      <tr>
        <td><strong>${s.name}</strong><br><span style="font-size:0.72rem;color:var(--muted);">${s.body_type} (Mag ${s.mag >= 0 ? '+' : ''}${s.mag})</span></td>
        <td><span class="badge dome">${s.nights_detected} Nights</span><br><span style="font-size:0.7rem;color:var(--muted);">${s.first_night}→${s.last_night}</span></td>
        <td class="mono">${s.total_arc_deg}° total<br><span style="color:#6ee7b7;">${s.Angular_speed_deg_day}°/day (${s.angular_speed_arcmin_day}'/d)</span></td>
        <td class="mono">${s.latest_ra_hms}<br>${s.latest_dec_dms}</td>
        <td><strong>${s.peak_snr}σ</strong></td>
        <td><button class="btn secondary" style="padding:5px 10px;font-size:0.76rem;border-color:rgba(56,189,248,0.45);color:#bae6fd;" onclick="openBlinkItem(${bIdx}, 'auto')">🔬 Blink</button></td>
      </tr>`;
    }).join('') || '<tr><td colspan="6" style="color:var(--muted);">No solar system bodies recovered yet.</td></tr>';

    const trks = data.candidate_moving_tracklets || [];
    document.getElementById('trackletsBody').innerHTML = trks.map(t => {
      const bIdx = registerBlinkItem(
        `🛸 Candidate Moving Tracklet: ${t.tracklet_id}`,
        `3-Night Collinear Motion (${(t.dates || []).join(' → ')}) • ${t.speed_deg_per_day}°/day • Collinearity ±${t.collinearity_err_deg}° • J2000 ${t.latest_ra_hms} ${t.latest_dec_dms} • SNR ${t.mean_snr}σ`,
        t.cutout_url,
        '3-NIGHT MOVING TRACKLET',
        'high'
      );
      return `
      <tr>
        <td><strong style="color:#38bdf8;">${t.tracklet_id}</strong><br><span style="font-size:0.7rem;color:var(--muted);">${t.classification_hint}</span></td>
        <td><span class="badge high">${t.nights_linked} Nights</span><br><span style="font-size:0.7rem;color:var(--muted);">${(t.dates || []).join('→')}</span></td>
        <td class="mono">${t.speed_deg_per_day}°/day<br><span style="color:#94a3b8;">Arc ${t.total_arc_deg}°</span></td>
        <td class="mono">±${t.collinearity_err_deg}°</td>
        <td><strong>${t.mean_snr}σ</strong></td>
        <td><button class="btn secondary" style="padding:5px 10px;font-size:0.76rem;border-color:rgba(56,189,248,0.45);color:#bae6fd;" onclick="openBlinkItem(${bIdx}, 'auto')">🔬 Blink</button></td>
      </tr>`;
    }).join('') || '<tr><td colspan="6" style="color:var(--muted);">No candidate moving tracklets linked.</td></tr>';

    const ots = data.multi_frame_optical_transients || [];
    const otIndices = ots.slice(0, 4).map(o => registerBlinkItem(
      `✨ ${o.transient_id} — Multi-Frame Optical Transient`,
      `Night ${o.date_str} (${o.frame_hits} sidereal frames) • J2000 ${o.ra_hms} ${o.dec_dms} • Diff SNR ${o.snr}σ`,
      o.cutout_url,
      'MULTI-FRAME TRANSIENT',
      'crit'
    ));

    const showcaseIndices = [
      ...sol.map((_, i) => i),
      ...trks.slice(0, 4).map((_, i) => sol.length + i),
      ...otIndices
    ];

    document.getElementById('blinkCutoutsGrid').innerHTML = showcaseIndices.map(bIdx => {
      const c = DEEP_BLINK_ITEMS[bIdx];
      return `
      <div class="t-card" onclick="openBlinkItem(${bIdx}, 'auto')">
        <img src="${c.tripletUrl}?v=${encodeURIComponent(surveyVer)}" style="height:175px;object-fit:contain;background:#090d16;" loading="lazy" />
        <div class="t-body">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span class="badge ${c.bClass}">${c.badge}</span>
            <span style="font-size:0.72rem;color:#38bdf8;font-weight:700;">🔬 Click to Live Blink →</span>
          </div>
          <div style="font-weight:700;font-size:0.9rem;color:#f8fafc;">${c.title}</div>
          <div class="mono" style="font-size:0.75rem;color:#cbd5e1;">${c.sub}</div>
        </div>
      </div>`;
    }).join('');
  } catch (err) {
    console.error('Deep survey load error:', err);
  }
}

async function triggerDeepSurveyScan() {
  const btn = document.getElementById('btnRescanSurvey');
  btn.disabled = true;
  btn.textContent = '⏳ Stacking Nights...';
  try {
    await fetch('/api/deep-survey/scan', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({force_rebuild: false})
    });
    await loadDeepSurveyData();
  } finally {
    btn.disabled = false;
    btn.textContent = '🔄 Re-Stack Archive';
  }
}

loadDiscoveryData();
loadDeepSurveyData();
setInterval(loadDiscoveryData, 30000);
</script>
</body>
</html>
"""


def get_discovery_patrol_html() -> str:
    return DISCOVERY_PATROL_HTML
