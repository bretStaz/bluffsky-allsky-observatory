"""
Solar System Ephemeris Shift-and-Stack & Earth-Orbital All-Sky Showcase Engine
==============================================================================
Additive companion module to `deep_sky_astrophoto.py` for BluffSky Observatory.

Adds two new showcase pillars without modifying any existing anomaly detection or survey pipelines:
  1. Solar System Ephemeris "Shift-and-Stack" & 16-Day Multi-Night Orbital Motion Portraits (960x720):
       - Ice Giant Neptune (mag 7.82, ~2.7 billion miles away in Pisces)
       - Ice Giant Uranus (mag 5.61, ~1.7 billion miles away in Taurus)
       - Terrestrial Planet Mars (mag 0.99, 2.73 deg prograde march across Cancer)
       - Ringed Giant Saturn (mag 0.41, 1.12 deg retrograde arc in Pisces/Cetus)
       - Gas Giant Jupiter (mag -1.81, pre-dawn prograde arc in Cancer)
  2. Earth-Orbital & All-Sky Composite Showcases (1200x900 4:3):
       - Geostationary "Clarke Belt" Satellite Necklace (35,786 km altitude, stationary GEO pinpoints + silky star trails)
       - Full-Arc SpaceMobile BlueBird (NORAD #69591) & Multi-Satellite LEO Overpass Composite
       - Multi-Night Orionid Fireball & Bolide All-Sky Celestial Dome Composite
       - True-Color Circumpolar Polaris Star-Trail Vortex (100% Airplane-Free 3-Frame Temporal Gate)
"""

import os
import re
import glob
import json
import math
import sqlite3
import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import ephem
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_CUDA = bool(torch.cuda.is_available())
except Exception:
    torch = None
    F = None
    HAS_CUDA = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASTRO_DIR = os.path.join(BASE_DIR, "astrophoto_data")
LINEAR_DIR = os.path.join(ASTRO_DIR, "linear_masters")
DB_PATH = os.path.join(BASE_DIR, "allsky_detections.db")
TLE_CACHE_DIR = os.path.join(BASE_DIR, "tle_cache")

ORB_W = 1200
ORB_H = 900
DOME_CX = 600.0
DOME_CY = 450.0
DOME_R = 414.0
DOME_MIN_ALT = 23.0


SOLAR_SYSTEM_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "sol_neptune",
        "planet_key": "Neptune",
        "name": "Ice Giant Neptune — Ephemeris Shift-and-Stack & 16-Day Orbit",
        "catalog_code": "Planet VIII • Neptune",
        "type": "solar_system",
        "type_label": "Outer Ice Giant • Ephemeris Stack",
        "constellation": "Pisces",
        "fov_w_deg": 7.6,
        "fov_h_deg": 5.7,
        "min_alt_deg": 32.0,
        "mask_px": 9,
        "ap_in_px": 8.0,
        "ap_out_px": 15.0,
        "min_epoch_sep_px": 6.5,
        "stretch_alpha": 31.0,
        "sat_boost": 1.20,
        "planet_boost": 1.38,
        "core_compress": False,
        "description": (
            "Over 10× fainter than the human naked-eye limit at magnitude 7.82 and 2.69 billion miles "
            "(29.0 AU) from Earth! Ephemeris shift-and-stack locks onto Neptune's sub-pixel orbital drift "
            "across clear October nights to recover the distant sapphire-blue ice giant from a wide-angle all-sky sensor."
        ),
    },
    {
        "id": "sol_uranus",
        "planet_key": "Uranus",
        "name": "Ice Giant Uranus — Ephemeris Shift-and-Stack & Retrograde Track",
        "catalog_code": "Planet VII • Uranus",
        "type": "solar_system",
        "type_label": "Ice Giant • Ephemeris Stack",
        "constellation": "Taurus",
        "fov_w_deg": 8.0,
        "fov_h_deg": 6.0,
        "min_alt_deg": 42.0,
        "mask_px": 11,
        "ap_in_px": 10.0,
        "ap_out_px": 18.0,
        "min_epoch_sep_px": 8.5,
        "stretch_alpha": 38.0,
        "sat_boost": 1.32,
        "planet_boost": 1.18,
        "core_compress": False,
        "description": (
            "Orbiting 1.74 billion miles (18.7 AU) away just south of the Pleiades in Taurus at magnitude 5.61. "
            "Co-adding over two hours of clear moonless exposures along Uranus's exact J2000 ephemeris reveals "
            "its signature pale cyan-aquamarine methane glow and multi-night retrograde arc."
        ),
    },
    {
        "id": "sol_mars",
        "planet_key": "Mars",
        "name": "Planet Mars — Multi-Night Orbital March Across Cancer",
        "catalog_code": "Planet IV • Mars",
        "type": "solar_system",
        "type_label": "Terrestrial Planet • 16-Day Orbit",
        "constellation": "Cancer",
        "fov_w_deg": 8.4,
        "fov_h_deg": 6.3,
        "min_alt_deg": 33.5,
        "mask_px": 24,
        "ap_in_px": 14.0,
        "ap_out_px": 24.0,
        "min_epoch_sep_px": 22.0,
        "stretch_alpha": 26.0,
        "sat_boost": 1.20,
        "planet_boost": 1.0,
        "core_compress": True,
        "description": (
            "Captured high in the pre-dawn sky as the ochre-amber terrestrial planet marches over "
            "5 full-moon diameters prograde across Cancer across clear October nights. Toggle '16-Day Orbit' "
            "to watch Mars step night-by-night across the fixed background starfield."
        ),
    },
    {
        "id": "sol_saturn",
        "planet_key": "Saturn",
        "name": "Ringed Giant Saturn — Multi-Night Retrograde Orbital Arc",
        "catalog_code": "Planet VI • Saturn",
        "type": "solar_system",
        "type_label": "Ringed Gas Giant • 16-Day Orbit",
        "constellation": "Pisces / Cetus",
        "fov_w_deg": 8.0,
        "fov_h_deg": 6.0,
        "min_alt_deg": 33.0,
        "mask_px": 25,
        "ap_in_px": 14.0,
        "ap_out_px": 24.0,
        "min_epoch_sep_px": 12.0,
        "stretch_alpha": 26.0,
        "sat_boost": 1.18,
        "planet_boost": 1.0,
        "core_compress": True,
        "description": (
            "Shining at golden-cream magnitude 0.41 nearly 790 million miles (8.5 AU) away along the "
            "Pisces/Cetus border, tracing a smooth westward retrograde orbital arc across two weeks of clear nights "
            "as Earth overtakes the ringed gas giant on the inside track."
        ),
    },
    {
        "id": "sol_jupiter",
        "planet_key": "Jupiter",
        "name": "Gas Giant Jupiter — Pre-Dawn Orbital Arc in Eastern Cancer",
        "catalog_code": "Planet V • Jupiter",
        "type": "solar_system",
        "type_label": "King of Planets • 16-Day Orbit",
        "constellation": "Cancer / Leo",
        "fov_w_deg": 8.4,
        "fov_h_deg": 6.3,
        "min_alt_deg": 33.5,
        "mask_px": 32,
        "ap_in_px": 16.0,
        "ap_out_px": 28.0,
        "min_epoch_sep_px": 11.0,
        "stretch_alpha": 24.0,
        "sat_boost": 1.16,
        "planet_boost": 1.0,
        "core_compress": True,
        "description": (
            "Blazing at magnitude -1.81 at 525 million miles (5.65 AU) high in the pre-dawn southeastern sky, "
            "advancing prograde across eastern Cancer while outshining every background star in the field."
        ),
    },
]


def _make_ephem_planet(key: str) -> Any:
    return getattr(ephem, key)()


def _observer_for_cfg(cfg: dict) -> ephem.Observer:
    obs = ephem.Observer()
    obs.lat = str(cfg.get("camera_lat", 45.0000))
    obs.lon = str(cfg.get("camera_lon", -100.0000))
    obs.elevation = 40.0
    obs.pressure = 0.0  # Geometric astrometric match with camera polynomial
    return obs


def _planet_j2000_via_inv_precession(
    planet_key: str,
    ts: float,
    obs: ephem.Observer,
    P_prec: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """
    Computes planet's apparent epoch-of-date (ra, dec) at UTC `ts` and transforms via `P_prec.T`
    into the exact J2000 coordinate frame of our gnomonic cutout grid.
    Returns (ra_j2000_deg, dec_j2000_deg, alt_deg, mag, earth_distance_au).
    """
    obs.date = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y/%m/%d %H:%M:%S")
    body = _make_ephem_planet(planet_key)
    body.compute(obs)
    ra_d = float(body.ra)
    dec_d = float(body.dec)
    cd = math.cos(dec_d)
    v_date = np.array([cd * math.cos(ra_d), cd * math.sin(ra_d), math.sin(dec_d)], dtype=np.float64)
    v_j = P_prec.T.astype(np.float64) @ v_date
    ra_j = math.degrees(math.atan2(float(v_j[1]), float(v_j[0]))) % 360.0
    dec_j = math.degrees(math.asin(max(-1.0, min(1.0, float(v_j[2])))))
    return ra_j, dec_j, math.degrees(float(body.alt)), float(body.mag), float(body.earth_distance)


def _format_au_distance(au: float) -> str:
    miles = au * 92955807.3
    if miles >= 1e9:
        return f"{miles / 1e9:.2f}B mi ({au:.2f} AU)"
    return f"{miles / 1e6:.0f}M mi ({au:.2f} AU)"


def _suppress_sensor_hot_pixels_rgb(rgb_lin: np.ndarray) -> np.ndarray:
    """
    Suppresses isolated 1-pixel Bayer hot-pixel impulses (where a single pixel spikes >3.5x above
    its 3x3 median + noise margin) while leaving optical star & GEO satellite PSFs (~2.4-4 px) intact.
    """
    med = cv2.medianBlur(rgb_lin, 3)
    diff = rgb_lin - med
    hot = diff > (0.018 + 2.6 * med)
    if np.any(hot):
        out = rgb_lin.copy()
        out[hot] = med[hot]
        return out
    return rgb_lin


def _enu_to_sensor_xy(
    east: np.ndarray,
    north: np.ndarray,
    up: np.ndarray,
    cfg: dict,
    poly_cx: Optional[np.ndarray] = None,
    poly_cy: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maps local Horizon unit vectors (East, North, Up) directly to 3552x3552 sensor pixels."""
    import deep_sky_astrophoto as dsa

    cx, cy = cfg.get("zenith_center", [1750.95, 1690.25])
    r_horiz = float(cfg.get("horizon_radius_px", 2007.74))
    phi = math.radians(float(cfg.get("bottom_azimuth_deg", 265.055)))
    cb, sb = math.cos(phi), math.sin(phi)
    vx0 = cb * east - sb * north
    vy0 = sb * east + cb * north
    vz0 = np.clip(up, -1.0, 1.0)
    alt_deg = np.degrees(np.arcsin(vz0)).astype(np.float32)

    tx = math.radians(float(cfg.get("tilt_x_deg", 6.292)))
    ty = math.radians(float(cfg.get("tilt_y_deg", -0.947)))
    vx1 = vx0
    vy1 = math.cos(tx) * vy0 - math.sin(tx) * vz0
    vz1 = math.sin(tx) * vy0 + math.cos(tx) * vz0
    vx_c = math.cos(ty) * vx1 + math.sin(ty) * vz1
    vy_c = vy1
    vz_c = np.clip(-math.sin(ty) * vx1 + math.cos(ty) * vz1, -1.0, 1.0)

    theta = np.arccos(vz_c)
    u = theta / (math.pi / 2.0)
    k2 = float(cfg.get("radial_k2", -0.14857))
    k3 = float(cfg.get("radial_k3", 0.03125))
    r = r_horiz * (u + k2 * (u ** 3) + k3 * (u ** 5))
    rho = np.maximum(np.hypot(vx_c, vy_c), 1e-9)
    px = (float(cx) + r * (vx_c / rho)).astype(np.float32)
    py = (float(cy) + r * (vy_c / rho)).astype(np.float32)

    if poly_cx is not None and poly_cy is not None:
        un = (px - 1776.0) / 1776.0
        vn = (py - 1776.0) / 1776.0
        valid_u = (np.abs(un) <= 1.05) & (np.abs(vn) <= 1.05) & (alt_deg >= 14.0)
        un_c = np.clip(un, -1.05, 1.05)
        vn_c = np.clip(vn, -1.05, 1.05)
        px = px + np.where(valid_u, dsa.eval_poly4_2d(un_c, vn_c, poly_cx), 0.0)
        py = py + np.where(valid_u, dsa.eval_poly4_2d(un_c, vn_c, poly_cy), 0.0)

    return px.astype(np.float32), py.astype(np.float32), alt_deg


def prepare_solar_system_grids(
    selected: List[Tuple[str, float, float]],
    cfg: dict,
    P_prec: np.ndarray,
    incremental: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Precomputes 960x720 tangential gnomonic grids centered on each planet's mid-archive J2000 position
    so cutouts can be warped on-GPU during the single sequential disk pass.
    """
    import deep_sky_astrophoto as dsa

    obs = _observer_for_cfg(cfg)
    sol_grids: Dict[str, Dict[str, Any]] = {}

    for t in SOLAR_SYSTEM_CATALOG:
        tid = t["id"]
        npz_p = os.path.join(LINEAR_DIR, f"{tid}_linear.npz")
        if incremental and os.path.exists(npz_p):
            continue

        pkey = t["planet_key"]
        # Anchor grid center at midpoint between earliest and latest archive timestamps so full 16-day orbital arc is centered!
        t_earliest = min(x[1] for x in selected)
        t_latest = max(x[1] for x in selected)
        ra_first, dec_first, _, _, _ = _planet_j2000_via_inv_precession(pkey, t_earliest, obs, P_prec)
        ra_last, dec_last, _, _, _ = _planet_j2000_via_inv_precession(pkey, t_latest, obs, P_prec)
        dra = ((ra_last - ra_first + 180.0) % 360.0) - 180.0
        ra0 = (ra_first + 0.5 * dra) % 360.0
        dec0 = 0.5 * (dec_first + dec_last)

        ra_grid_j, dec_grid_j = dsa.build_gnomonic_radec_grid(
            ra0, dec0, float(t["fov_w_deg"]), float(t["fov_h_deg"]), dsa.DSO_W, dsa.DSO_H
        )
        ra_d, dec_d = dsa.precess_radec_grid(ra_grid_j, dec_grid_j, P_prec)
        c_ra_d, c_dec_d = dsa.precess_radec_grid(
            np.array([math.radians(ra0)], dtype=np.float32),
            np.array([math.radians(dec0)], dtype=np.float32),
            P_prec,
        )
        dg: Dict[str, Any] = {
            "target": t,
            "ra0_deg": ra0,
            "dec0_deg": dec0,
            "ra_date": ra_d,
            "sin_dec": np.sin(dec_d).astype(np.float32),
            "cos_dec": np.cos(dec_d).astype(np.float32),
            "c_ra": c_ra_d,
            "c_sin_dec": np.sin(c_dec_d).astype(np.float32),
            "c_cos_dec": np.cos(c_dec_d).astype(np.float32),
            "raw_cutouts": [],  # List of (alt_deg, ts, float16_cutout)
        }
        if HAS_CUDA:
            dg["sin_g"] = torch.from_numpy(dg["sin_dec"]).cuda()
            dg["cos_g"] = torch.from_numpy(dg["cos_dec"]).cuda()
            dg["ra_g"] = torch.from_numpy(dg["ra_date"]).cuda()
        sol_grids[tid] = dg

    return sol_grids


def _masked_starfield_master_gpu(
    frames_list: List[np.ndarray],
    planet_xys: List[Tuple[float, float]],
    mask_radius_px: float,
) -> np.ndarray:
    """
    Computes the deep sigma-clipped starfield master while masking out a disk of `mask_radius_px`
    around the moving planet's instantaneous position `(px_t, py_t)` in each frame so zero planet
    ghost ever leaks into the starfield master!
    """
    n_f = len(frames_list)
    h, w = frames_list[0].shape[:2]
    arr = np.stack(frames_list, axis=0).astype(np.float32)  # (N, H, W, 3)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    valid_m = np.ones((n_f, h, w), dtype=bool)
    r2 = float(mask_radius_px * mask_radius_px)
    for i, (px, py) in enumerate(planet_xys):
        valid_m[i] = ((xx - px) ** 2 + (yy - py) ** 2) > r2

    # Ensure every pixel has at least 4 valid frames (fallback to unmasked if planet barely moved)
    cnt = valid_m.sum(axis=0)
    sparse = cnt < 4
    if np.any(sparse):
        valid_m[:, sparse] = True

    if HAS_CUDA:
        with torch.no_grad():
            t = torch.from_numpy(arr).cuda()
            vm = torch.from_numpy(valid_m).cuda().unsqueeze(-1)
            t_nan = torch.where(vm, t, torch.tensor(float("nan"), device="cuda"))
            med = torch.nan_to_num(torch.nanmedian(t_nan, dim=0).values, nan=0.0)
            mad = torch.nan_to_num(
                torch.nanmedian(
                    torch.where(vm, torch.abs(t - med.unsqueeze(0)), torch.tensor(float("nan"), device="cuda")),
                    dim=0,
                ).values,
                nan=0.0,
            )
            sigma = torch.clamp(1.4826 * mad, min=1e-4)
            inl = vm & (t >= med.unsqueeze(0) - 2.2 * sigma.unsqueeze(0)) & (t <= med.unsqueeze(0) + 2.3 * sigma.unsqueeze(0))
            inl_f = inl.to(torch.float32)
            out = (t * inl_f).sum(dim=0) / torch.clamp(inl_f.sum(dim=0), min=1.0)
            res = out.cpu().numpy().astype(np.float32)
            del t, vm, t_nan, med, mad, sigma, inl, inl_f, out
            torch.cuda.empty_cache()
            return res

    arr_nan = np.where(valid_m[..., None], arr, np.nan)
    med = np.nan_to_num(np.nanmedian(arr_nan, axis=0), nan=0.0)
    mad = np.nan_to_num(np.nanmedian(np.where(valid_m[..., None], np.abs(arr - med[None, ...]), np.nan), axis=0), nan=0.0)
    sigma = np.maximum(1.4826 * mad, 1e-4)
    inl = valid_m[..., None] & (arr >= med[None, ...] - 2.2 * sigma[None, ...]) & (arr <= med[None, ...] + 2.3 * sigma[None, ...])
    return ((arr * inl).sum(axis=0) / np.maximum(inl.sum(axis=0), 1)).astype(np.float32)


def _supplement_multi_night_planet_cutouts(
    dg: Dict[str, Any],
    cfg: dict,
    poly_cx: Optional[np.ndarray],
    poly_cy: Optional[np.ndarray],
    P_prec: np.ndarray,
    archive_root: str,
) -> None:
    """
    Ensures every Solar System target captures cutouts across ALL clear high-altitude nights in
    `archive_root` (even pre-dawn nights at `sun_alt <= -16.2 deg` or partial-moon nights that the
    deep-nebula moonless filter skipped), unlocking the full 16-day multi-night orbital motion arc!
    """
    import deep_sky_astrophoto as dsa

    t = dg["target"]
    pkey = t["planet_key"]
    min_alt = float(t["min_alt_deg"])
    existing_nights = {
        datetime.datetime.fromtimestamp(ts_v - 12.0 * 3600.0).strftime("%Y-%m-%d")
        for _, ts_v, _ in dg["raw_cutouts"]
    }
    if len(existing_nights) >= 8 or not archive_root or not os.path.isdir(archive_root):
        return

    day_dirs: List[str] = []
    base_n = os.path.basename(archive_root.rstrip("\\/"))
    if base_n.isdigit() and len(base_n) == 2:
        day_dirs = sorted(glob.glob(os.path.join(archive_root, "*")))
    else:
        for m_dir in sorted(glob.glob(os.path.join(archive_root, "*"))):
            if os.path.isdir(m_dir):
                mn = os.path.basename(m_dir)
                if mn.isdigit() and int(mn) < 10 and base_n == "2026":
                    continue
                day_dirs.extend(sorted(glob.glob(os.path.join(m_dir, "*"))))

    obs = _observer_for_cfg(cfg)
    sun = ephem.Sun()
    moon = ephem.Moon()
    is_faint_ice = float(t.get("planet_boost", 1.0)) > 1.15  # Neptune / Uranus prefer low-moon dark sky
    srgb_lut = np.power(np.arange(256, dtype=np.float32) / 255.0, 2.2).astype(np.float32)

    # Group candidate clear high-altitude files by observing night (90s step after fast filename pre-filter)
    cand_by_night: Dict[str, List[Tuple[float, float, str, float]]] = {}
    for d_dir in day_dirs:
        if not os.path.isdir(d_dir):
            continue
        files = sorted(glob.glob(os.path.join(d_dir, "image_*.jpg")))[::3]
        for fp in files:
            b = os.path.basename(fp)
            if len(b) >= 12 and b[6:8].isdigit():
                hh = int(b[6:8])
                if 6 <= hh <= 18:
                    continue
            ts = dsa.file_utc_midpoint(fp)
            nk = datetime.datetime.fromtimestamp(ts - 12.0 * 3600.0).strftime("%Y-%m-%d")
            if nk in existing_nights:
                continue
            obs.date = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y/%m/%d %H:%M:%S")
            sun.compute(obs)
            if math.degrees(float(sun.alt)) > (-17.5 if is_faint_ice else -15.0):
                continue
            if is_faint_ice:
                moon.compute(obs)
                if math.degrees(float(moon.alt)) > 5.0 and float(moon.phase) > 25.0:
                    continue
            body = _make_ephem_planet(pkey)
            body.compute(obs)
            alt_p = math.degrees(float(body.alt))
            if alt_p < min_alt:
                continue
            cand_by_night.setdefault(nk, []).append((alt_p, ts, fp, float(obs.sidereal_time())))

    for nk in sorted(cand_by_night.keys()):
        cands = sorted(cand_by_night[nk], key=lambda x: x[0], reverse=True)[:12]
        added_this_night = 0
        for alt_p, ts, fp, lst_rad in cands:
            im_q = cv2.imread(fp, cv2.IMREAD_REDUCED_GRAYSCALE_4)
            if im_q is None:
                continue
            qh, qw = im_q.shape[:2]
            c_q = im_q[qh // 4 : 3 * qh // 4, qw // 4 : 3 * qw // 4]
            if float(np.median(c_q)) > (49.0 if is_faint_ice else 64.0):
                continue
            hp_q = cv2.subtract(c_q, cv2.GaussianBlur(c_q, (7, 7), 0))
            if int(np.count_nonzero(hp_q > 16)) < (310 if is_faint_ice else 260):
                continue

            bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[0] != 3552:
                continue
            rgb_lin = _suppress_sensor_hot_pixels_rgb(srgb_lut[bgr[:, :, ::-1]])
            mx_d, my_d, _ = dsa.radec_date_to_sensor_xy(
                dg["sin_dec"], dg["cos_dec"], dg["ra_date"], lst_rad, cfg, poly_cx, poly_cy, img_size=3552
            )
            cut = np.clip(cv2.remap(rgb_lin, mx_d, my_d, interpolation=cv2.INTER_CUBIC), 0.0, 1.5)
            dg["raw_cutouts"].append((alt_p, ts, cut.astype(np.float16)))
            added_this_night += 1
            if added_this_night >= 6:
                break


def _shape_planetary_psf(psf: np.ndarray, box_r: np.ndarray, core_compress: bool) -> np.ndarray:
    """
    For bright naked-eye planets (`core_compress=True`: Mars, Saturn, Jupiter), tames wide-angle
    blue atmospheric dispersion fringes and compresses the outer saturation skirt so each planet
    renders as a crisp, round golden-cream / ochre-amber jewel rather than an over-stretched blob.
    """
    out = np.maximum(psf, 0.0).copy()
    if not core_compress:
        return out
    # 1. Tame wide-field fisheye blue chromatic dispersion skirt around saturated R/G core
    max_rg = np.maximum(out[:, :, 0], out[:, :, 1])
    out[:, :, 2] = np.minimum(out[:, :, 2], 0.94 * max_rg + 0.0015)
    # 2. Radial Gaussian wing suppression outside r > 4.2 px so atmospheric halo stays compact
    wing_env = np.where(box_r <= 4.2, 1.0, np.exp(-0.5 * ((box_r - 4.2) / 3.4) ** 2)).astype(np.float32)
    out *= wing_env[:, :, None]
    # 3. Soft HDR highlight compression preserving natural warm ochre/amber/cream hue
    lum = np.maximum(0.30 * out[:, :, 0] + 0.50 * out[:, :, 1] + 0.20 * out[:, :, 2], 1e-6)
    comp_lum = (0.11 * lum) / (0.045 + lum)
    out *= (comp_lum / lum)[:, :, None]
    return out.astype(np.float32)


def stack_solar_system_targets(
    sol_grids: Dict[str, Dict[str, Any]],
    cfg: dict,
    P_prec: np.ndarray,
    existing_objects_by_id: Dict[str, Dict[str, Any]],
    poly_cx: Optional[np.ndarray] = None,
    poly_cy: Optional[np.ndarray] = None,
    archive_root: str = "",
    max_dso_frames: int = 240,
    progress_cb: Optional[Callable[[str, int], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Builds or reuses all 5 Solar System Ephemeris Shift-and-Stack & 16-Day Orbital Motion portraits.
    """
    import deep_sky_astrophoto as dsa

    w, h = dsa.DSO_W, dsa.DSO_H
    obs = _observer_for_cfg(cfg)
    tukey_coarse = dsa._make_tukey_2d(w // 2, h // 2, alpha=0.18)
    yy_f, xx_f = np.mgrid[0:h, 0:w].astype(np.float32)

    results: List[Dict[str, Any]] = []
    for idx, t in enumerate(SOLAR_SYSTEM_CATALOG):
        tid = t["id"]
        npz_p = os.path.join(LINEAR_DIR, f"{tid}_linear.npz")
        if tid not in sol_grids and os.path.exists(npz_p) and tid in existing_objects_by_id:
            results.append(existing_objects_by_id[tid])
            continue
        if tid not in sol_grids:
            continue

        if progress_cb:
            progress_cb(f"Ephemeris Shift-and-Stack: {t['name']}...", 86 + idx)

        dg = sol_grids[tid]
        _supplement_multi_night_planet_cutouts(dg, cfg, poly_cx, poly_cy, P_prec, archive_root)
        ra0, dec0 = float(dg["ra0_deg"]), float(dg["dec0_deg"])
        fov_w, fov_h = float(t["fov_w_deg"]), float(t["fov_h_deg"])
        core_comp = bool(t.get("core_compress", False))
        raw_list = dg["raw_cutouts"]
        if len(raw_list) < 5:
            continue

        # Stratify across available nights while preferring highest altitude per night so all 16 days are represented!
        by_night: Dict[str, List[Tuple[float, float, np.ndarray]]] = {}
        for alt_v, ts_v, cut_f16 in raw_list:
            # Group by observing night (subtract 12h from local time so evening + pre-dawn share one night key)
            night_key = datetime.datetime.fromtimestamp(ts_v - 12.0 * 3600.0).strftime("%Y-%m-%d")
            by_night.setdefault(night_key, []).append((alt_v, ts_v, cut_f16))

        for nk in by_night:
            by_night[nk].sort(key=lambda x: x[0], reverse=True)

        selected_cuts: List[Tuple[float, float, np.ndarray, str]] = []
        per_night_quota = max(6, max_dso_frames // max(1, len(by_night)))
        overflow: List[Tuple[float, float, np.ndarray, str]] = []
        for nk in sorted(by_night.keys()):
            nlist = by_night[nk]
            for j, item_c in enumerate(nlist):
                if j < per_night_quota:
                    selected_cuts.append((item_c[0], item_c[1], item_c[2], nk))
                else:
                    overflow.append((item_c[0], item_c[1], item_c[2], nk))
        if len(selected_cuts) < max_dso_frames and overflow:
            overflow.sort(key=lambda x: x[0], reverse=True)
            selected_cuts.extend(overflow[: max_dso_frames - len(selected_cuts)])

        selected_cuts.sort(key=lambda x: x[0], reverse=True)
        max_alt_obs = float(selected_cuts[0][0])

        # Pick sharpest reference frame among top 10 highest-altitude cutouts
        best_ref_i = 0
        best_sc = -1.0
        for ci in range(min(10, len(selected_cuts))):
            hp_c = dsa._extract_star_hp(selected_cuts[ci][2].astype(np.float32))
            sc = float(np.percentile(hp_c, 99.6))
            if sc > best_sc:
                best_sc = sc
                best_ref_i = ci

        ref_raw = selected_cuts[best_ref_i][2].astype(np.float32)
        ref_star_hp = dsa._extract_star_hp(ref_raw)
        ref_coarse_hp = cv2.GaussianBlur(
            cv2.resize(ref_star_hp, (w // 2, h // 2), interpolation=cv2.INTER_AREA),
            (0, 0),
            sigmaX=1.05,
            sigmaY=1.05,
        )

        pkey = t["planet_key"]
        aligned_frames: List[np.ndarray] = []
        frame_ts_list: List[float] = []
        frame_nights: List[str] = []
        planet_xys: List[Tuple[float, float]] = []
        planet_radecs: List[Tuple[float, float]] = []
        mags: List[float] = []
        dists_au: List[float] = []
        resps: List[float] = []

        ref_flux_val: Optional[float] = None
        ref_clean_single: Optional[np.ndarray] = None

        # Process reference frame first so ref_flux_val is established
        order_idxs = [best_ref_i] + [i for i in range(len(selected_cuts)) if i != best_ref_i]
        for ci in order_idxs:
            alt_v, ts_v, cut_f16, nk = selected_cuts[ci]
            ra_p, dec_p, _, mag_p, au_p = _planet_j2000_via_inv_precession(pkey, ts_v, obs, P_prec)
            xy_p = dsa.radec_to_gnomonic_xy(ra_p, dec_p, ra0, dec0, fov_w, fov_h, w, h)
            if xy_p is None:
                continue
            if ci == best_ref_i:
                al_rgb = ref_raw
                resp = 1.0
            else:
                res = dsa.align_dso_cutout_subpixel(cut_f16.astype(np.float32), ref_star_hp, ref_coarse_hp, tukey_coarse)
                if res is None:
                    continue
                al_rgb, resp, _ = res

            # Mask out 38px around instantaneous planet position during sky-gradient & stellar transparency measurement
            bg_safe = ((xx_f - xy_p[0]) ** 2 + (yy_f - xy_p[1]) ** 2) >= (38.0 ** 2)
            clean_rgb, _ = dsa.remove_dso_sky_gradient_quadratic(al_rgb, bg_safe)
            lum_bg = 0.30 * clean_rgb[:, :, 0] + 0.50 * clean_rgb[:, :, 1] + 0.20 * clean_rgb[:, :, 2]
            hp_stars = (lum_bg - cv2.GaussianBlur(lum_bg, (0, 0), sigmaX=2.8, sigmaY=2.8)) * bg_safe.astype(np.float32)
            f_flux = max(float(np.percentile(np.maximum(hp_stars, 0.0), 99.3)), 1e-5)

            if ref_flux_val is None:
                ref_flux_val = f_flux
                ref_clean_single = clean_rgb.copy()
                scale_f = 1.0
            else:
                scale_f = float(np.clip(ref_flux_val / f_flux, 0.68, 1.50))

            aligned_frames.append((clean_rgb * scale_f).astype(np.float32))
            frame_ts_list.append(ts_v)
            frame_nights.append(nk)
            planet_xys.append(xy_p)
            planet_radecs.append((ra_p, dec_p))
            mags.append(mag_p)
            dists_au.append(au_p)
            if ci != best_ref_i:
                resps.append(resp)

        dg["raw_cutouts"] = []
        n_used = len(aligned_frames)
        if n_used < 5 or ref_clean_single is None:
            continue

        # 1. Build 100% planet-free deep starfield master
        starfield_master = _masked_starfield_master_gpu(aligned_frames, planet_xys, float(t["mask_px"]))

        # Background noise level of starfield_master for clean star-dipole gating
        sf_lum = 0.30 * starfield_master[:, :, 0] + 0.50 * starfield_master[:, :, 1] + 0.20 * starfield_master[:, :, 2]
        sf_med = float(np.median(sf_lum))
        sf_sigma = max(1.4826 * float(np.median(np.abs(sf_lum - sf_med))), 1e-5)

        # 2. Ephemeris Shift-and-Stack onto mid-archive J2000 reference position (center of portrait!)
        px_ref, py_ref = 0.5 * w - 0.5, 0.5 * h - 0.5
        ap_in = float(t["ap_in_px"])
        ap_out = float(t["ap_out_px"])
        planet_boost = float(t.get("planet_boost", 1.0))

        shifted_apertures: List[np.ndarray] = []
        R_box = int(math.ceil(ap_out)) + 6
        box_y, box_x = np.mgrid[-R_box : R_box + 1, -R_box : R_box + 1].astype(np.float32)
        box_r = np.hypot(box_x, box_y)
        ap_taper = np.clip((ap_out - box_r) / max(ap_out - ap_in, 1.0), 0.0, 1.0)
        ap_taper = (0.5 - 0.5 * np.cos(math.pi * ap_taper)).astype(np.float32)[:, :, None]

        for f_rgb, (px_t, py_t) in zip(aligned_frames, planet_xys):
            diff = f_rgb - starfield_master
            # Sample a (2*R_box+1, 2*R_box+1) patch centered on exact sub-pixel (px_t, py_t)
            sx_map = (px_t + box_x).astype(np.float32)
            sy_map = (py_t + box_y).astype(np.float32)
            patch = cv2.remap(diff, sx_map, sy_map, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            shifted_apertures.append(np.maximum(patch, -0.35 * sf_sigma) * ap_taper)

        sh_arr = np.stack(shifted_apertures, axis=0)  # (N, 2R+1, 2R+1, 3)
        # Trimmed mean (20%..88%) rejects any background star residual on a single night AND any airplane/satellite hit!
        sh_sorted = np.sort(sh_arr, axis=0)
        i_lo = int(0.18 * len(sh_sorted))
        i_hi = max(i_lo + 1, int(0.88 * len(sh_sorted)))
        planet_psf_raw = np.maximum(sh_sorted[i_lo:i_hi].mean(axis=0), 0.0) * planet_boost

        # Measure fine optical centroid inside central +/-7px of planet_psf_raw so reticle locks to <0.2px!
        p_lum = 0.30 * planet_psf_raw[:, :, 0] + 0.50 * planet_psf_raw[:, :, 1] + 0.20 * planet_psf_raw[:, :, 2]
        c_sub = np.maximum(p_lum[R_box - 6 : R_box + 7, R_box - 6 : R_box + 7] - float(np.percentile(p_lum, 60)), 0.0)
        fine_dx, fine_dy = 0.0, 0.0
        if float(c_sub.sum()) > 1e-5:
            cy_g, cx_g = np.mgrid[-6:7, -6:7].astype(np.float32)
            fine_dx = float(np.clip((c_sub * cx_g).sum() / c_sub.sum(), -4.5, 4.5))
            fine_dy = float(np.clip((c_sub * cy_g).sum() / c_sub.sum(), -4.5, 4.5))

        planet_psf = _shape_planetary_psf(planet_psf_raw, box_r, core_comp)

        stack_rgb = starfield_master.copy()
        ix_c, iy_c = int(round(px_ref)), int(round(py_ref))
        y0_b, y1_b = iy_c - R_box, iy_c + R_box + 1
        x0_b, x1_b = ix_c - R_box, ix_c + R_box + 1
        stack_rgb[y0_b:y1_b, x0_b:x1_b] += planet_psf

        # 3. Build 16-Day Multi-Night Orbital Motion Composite (`multi_epoch_rgb`)
        night_groups: Dict[str, List[int]] = {}
        for i_f, nk in enumerate(frame_nights):
            night_groups.setdefault(nk, []).append(i_f)

        night_infos: List[Dict[str, Any]] = []
        for nk in sorted(night_groups.keys()):
            idxs_n = night_groups[nk]
            mean_ts = float(np.mean([frame_ts_list[i] for i in idxs_n]))
            mean_x = float(np.mean([planet_xys[i][0] for i in idxs_n])) + fine_dx
            mean_y = float(np.mean([planet_xys[i][1] for i in idxs_n])) + fine_dy
            mean_ra = float(np.mean([planet_radecs[i][0] for i in idxs_n]))
            mean_dec = float(np.mean([planet_radecs[i][1] for i in idxs_n]))
            dt_lbl = datetime.datetime.fromtimestamp(mean_ts).strftime("Oct %d")
            night_infos.append({
                "night": nk,
                "label": dt_lbl,
                "ts": mean_ts,
                "x": mean_x,
                "y": mean_y,
                "ra": mean_ra,
                "dec": mean_dec,
                "idxs": idxs_n,
            })

        # Select well-spaced epochs along the orbital arc (always anchoring first and last clear night!)
        min_sep = float(t["min_epoch_sep_px"])
        spaced_epochs: List[Dict[str, Any]] = []
        if len(night_infos) <= 2:
            spaced_epochs = night_infos
        else:
            spaced_epochs.append(night_infos[0])
            last_ep = night_infos[-1]
            for cand in night_infos[1:-1]:
                d_prev = math.hypot(cand["x"] - spaced_epochs[-1]["x"], cand["y"] - spaced_epochs[-1]["y"])
                d_last = math.hypot(cand["x"] - last_ep["x"], cand["y"] - last_ep["y"])
                if d_prev >= min_sep and d_last >= 0.72 * min_sep:
                    spaced_epochs.append(cand)
            if math.hypot(last_ep["x"] - spaced_epochs[-1]["x"], last_ep["y"] - spaced_epochs[-1]["y"]) >= 3.0 or len(spaced_epochs) == 1:
                spaced_epochs.append(last_ep)

        multi_epoch_rgb = starfield_master.copy()
        epoch_meta_clean: List[Dict[str, Any]] = []
        for ep in spaced_epochs:
            idxs_n = ep["idxs"]
            ep_patches = []
            for i_f in idxs_n:
                px_t, py_t = planet_xys[i_f]
                diff = aligned_frames[i_f] - starfield_master
                patch = cv2.remap(
                    diff,
                    (px_t + box_x).astype(np.float32),
                    (py_t + box_y).astype(np.float32),
                    interpolation=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                ep_patches.append(np.maximum(patch, 0.0) * ap_taper)
            if len(ep_patches) >= 3:
                ep_s = np.sort(np.stack(ep_patches, axis=0), axis=0)
                ep_psf = ep_s[: max(1, int(0.85 * len(ep_s)))].mean(axis=0) * planet_boost
            else:
                ep_psf = np.mean(ep_patches, axis=0) * planet_boost
            ep_psf = _shape_planetary_psf(ep_psf, box_r, core_comp)

            # Stamp epoch PSF at exact sub-pixel `(ep['x'] - fine_dx, ep['y'] - fine_dy)` via bicubic shift
            ex_raw = ep["x"] - fine_dx
            ey_raw = ep["y"] - fine_dy
            ix_e, iy_e = int(round(ex_raw)), int(round(ey_raw))
            frac_x, frac_y = ex_raw - ix_e, ey_raw - iy_e
            ep_psf_shifted = cv2.remap(
                ep_psf.astype(np.float32),
                (box_x + R_box - frac_x).astype(np.float32),
                (box_y + R_box - frac_y).astype(np.float32),
                interpolation=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
            )
            ye0, ye1 = max(0, iy_e - R_box), min(h, iy_e + R_box + 1)
            xe0, xe1 = max(0, ix_e - R_box), min(w, ix_e + R_box + 1)
            py0, py1 = ye0 - (iy_e - R_box), ye1 - (iy_e - R_box)
            px0, px1 = xe0 - (ix_e - R_box), xe1 - (ix_e - R_box)
            if ye1 > ye0 and xe1 > xe0:
                multi_epoch_rgb[ye0:ye1, xe0:xe1] += np.maximum(ep_psf_shifted[py0:py1, px0:px1], 0.0)

            epoch_meta_clean.append({
                "night": ep["night"],
                "label": ep["label"],
                "x": round(float(ep["x"]), 2),
                "y": round(float(ep["y"]), 2),
                "ra": round(float(ep["ra"]), 4),
                "dec": round(float(ep["dec"]), 4),
                "frames": len(idxs_n),
            })

        # Compute total angular orbital displacement in degrees across all observed nights
        if len(night_infos) >= 2:
            r1, d1 = math.radians(night_infos[0]["ra"]), math.radians(night_infos[0]["dec"])
            r2, d2 = math.radians(night_infos[-1]["ra"]), math.radians(night_infos[-1]["dec"])
            cos_sep = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r2 - r1)
            arc_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_sep))))
        else:
            arc_deg = 0.0

        mean_mag = round(float(np.mean(mags)), 2)
        mean_au = float(np.mean(dists_au))
        dist_str = _format_au_distance(mean_au)

        meta_payload = {
            "ra0_deg": ra0,
            "dec0_deg": dec0,
            "mid_x": round(float(px_ref + fine_dx), 2),
            "mid_y": round(float(py_ref + fine_dy), 2),
            "arc_deg": round(arc_deg, 2),
            "mean_mag": mean_mag,
            "mean_au": round(mean_au, 3),
            "dist_str": dist_str,
            "total_nights": len(night_infos),
            "epochs": epoch_meta_clean,
        }

        np.savez_compressed(
            npz_p,
            stack_rgb=stack_rgb.astype(np.float32),
            multi_epoch_rgb=multi_epoch_rgb.astype(np.float32),
            single_rgb=ref_clean_single.astype(np.float32),
            meta_json=np.array(json.dumps(meta_payload)),
        )

        exp_min = round(n_used * 0.5, 1)
        exp_hrs = round(exp_min / 60.0, 2)
        snr_gain = round(math.sqrt(n_used), 2)
        mag_gain = round(1.25 * math.log10(max(n_used, 1)), 2)
        arcmin_px = round((fov_w * 60.0) / w, 2)

        results.append({
            "id": tid,
            "name": t["name"],
            "catalog_code": f"{t['catalog_code']} (mag {mean_mag:+.2f})",
            "type": "solar_system",
            "type_label": t["type_label"],
            "constellation": t["constellation"],
            "distance": dist_str,
            "apparent_mag": f"{mean_mag:.2f}",
            "ra_deg": round(ra0, 4),
            "dec_deg": round(dec0, 4),
            "ra_hms": dsa.format_ra_hms(ra0),
            "dec_dms": dsa.format_dec_dms(dec0),
            "fov_str": f"{fov_w:.1f}° × {fov_h:.1f}° ({arc_deg:.2f}° Orbit Arc)",
            "pixel_scale_arcmin": arcmin_px,
            "resolution": f"{w} × {h} (Ephemeris Shift-and-Stack)",
            "frames_stacked": n_used,
            "integration_minutes": exp_min,
            "integration_hours": exp_hrs,
            "snr_gain_x": snr_gain,
            "limiting_mag_gain": f"+{mag_gain:.2f} mag",
            "max_altitude_deg": round(max_alt_obs, 1),
            "median_registration_corr": round(float(np.median(resps)) if resps else 1.0, 3),
            "description": t["description"],
            "stack_url": f"/astrophoto_data/{tid}_stack.jpg",
            "annotated_url": f"/astrophoto_data/{tid}_annotated.jpg",
            "single_url": f"/astrophoto_data/{tid}_single.jpg",
        })

    return results


def _annotate_solar_system_plate(
    multi_bgr: np.ndarray,
    target: Dict[str, Any],
    meta: Dict[str, Any],
    n_frames: int,
    exp_minutes: float,
) -> np.ndarray:
    """
    Annotates the 16-Day Multi-Night Orbital Motion composite with dated epoch ticks,
    orbital direction vector, astrometric compass rose, and a crisp magnified track inset
    for compact outer-planet orbital arcs (Neptune, Uranus, Saturn, Jupiter).
    """
    import deep_sky_astrophoto as dsa

    base_clean = multi_bgr.copy()
    vis = multi_bgr.copy()
    h, w = vis.shape[:2]
    epochs = meta.get("epochs", [])

    if epochs:
        pts = [(float(e["x"]), float(e["y"])) for e in epochs]
        # Direction vector along track and perpendicular normal for non-overlapping date ticks
        if len(pts) >= 2:
            vx = pts[-1][0] - pts[0][0]
            vy = pts[-1][1] - pts[0][1]
            vlen = max(math.hypot(vx, vy), 1e-3)
            ux, uy = vx / vlen, vy / vlen
            nx, ny = -uy, ux
            if ny > 0:
                nx, ny = -nx, -ny  # Prefer upward tick offset
        else:
            ux, uy, nx, ny = 1.0, 0.0, 0.0, -1.0
            vlen = 0.0

        # Draw subtle orbital path guide offset slightly below/behind the planet pearls
        if len(pts) >= 2:
            off_px = -14.0
            p_start = (int(round(pts[0][0] - nx * (-off_px))), int(round(pts[0][1] - ny * (-off_px))))
            p_end = (int(round(pts[-1][0] - nx * (-off_px) + ux * 14.0)), int(round(pts[-1][1] - ny * (-off_px) + uy * 14.0)))
            cv2.arrowedLine(vis, p_start, p_end, (240, 210, 75), 1, cv2.LINE_AA, tipLength=max(0.06, min(0.24, 11.0 / (vlen + 18.0))))

        # Draw crisp gap-tick pointers and staggered date labels on spaced epochs
        for i, ep in enumerate(epochs):
            ex, ey = float(ep["x"]), float(ep["y"])
            ix, iy = int(round(ex)), int(round(ey))
            is_end = (i == 0 or i == len(epochs) - 1)
            col = (140, 250, 175) if is_end else (240, 215, 85)
            # Short perpendicular pointer tick outside the planet pearl (never covering the planet!)
            sgn = 1.0 if (i % 2 == 0 or vlen > 140.0) else -1.0
            t0 = (int(round(ex + sgn * nx * 11.0)), int(round(ey + sgn * ny * 11.0)))
            t1 = (int(round(ex + sgn * nx * 21.0)), int(round(ey + sgn * ny * 21.0)))
            cv2.line(vis, t0, t1, col, 1, cv2.LINE_AA)
            if is_end or vlen >= 120.0 or (len(epochs) <= 4 and i == len(epochs) // 2):
                lbl = ep["label"]
                tx = int(np.clip(t1[0] - 18, 12, w - 95))
                ty = int(np.clip(t1[1] + (-6 if sgn * ny < 0 else 14), 22, h - 46))
                cv2.putText(vis, lbl, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(vis, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.43, col, 1, cv2.LINE_AA)

        # Magnified Multi-Night Orbital Track Inset for compact outer-planet arcs (span <= 165 px)
        xs_ep = [p[0] for p in pts]
        ys_ep = [p[1] for p in pts]
        span_px = math.hypot(max(xs_ep) - min(xs_ep), max(ys_ep) - min(ys_ep))
        if span_px <= 165.0:
            iw_box, ih_box = 276, 204
            zoom_f = float(np.clip(156.0 / max(span_px, 32.0), 1.55, 3.6))
            crop_w = int(round(iw_box / zoom_f))
            crop_h = int(round(ih_box / zoom_f))
            cx_tr = 0.5 * (min(xs_ep) + max(xs_ep))
            cy_tr = 0.5 * (min(ys_ep) + max(ys_ep))
            x0_c = int(np.clip(round(cx_tr - 0.5 * crop_w), 0, w - crop_w))
            y0_c = int(np.clip(round(cy_tr - 0.5 * crop_h), 0, h - crop_h))

            patch_zoom = cv2.resize(
                base_clean[y0_c : y0_c + crop_h, x0_c : x0_c + crop_w],
                (iw_box, ih_box),
                interpolation=cv2.INTER_CUBIC,
            )
            # Annotate each pearl cleanly inside the magnified inset
            for i, ep in enumerate(epochs):
                zx = (float(ep["x"]) - x0_c) * zoom_f
                zy = (float(ep["y"]) - y0_c) * zoom_f
                izx, izy = int(round(zx)), int(round(zy))
                col_z = (140, 250, 175) if (i == 0 or i == len(epochs) - 1) else (240, 215, 85)
                cv2.circle(patch_zoom, (izx, izy), int(round(7 + 0.8 * zoom_f)), col_z, 1, cv2.LINE_AA)
                sgn_z = 1.0 if (i % 2 == 0) else -1.0
                ly_off = int(round(izy + sgn_z * (16 + zoom_f)))
                ly_off = int(np.clip(ly_off, 28, ih_box - 10))
                lx_off = int(np.clip(izx - 20, 6, iw_box - 54))
                cv2.putText(patch_zoom, ep["label"], (lx_off + 1, ly_off + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(patch_zoom, ep["label"], (lx_off, ly_off), cv2.FONT_HERSHEY_SIMPLEX, 0.39, col_z, 1, cv2.LINE_AA)

            # Title header inside inset
            hdr = patch_zoom[:22, :].copy()
            patch_zoom[:22, :] = (hdr.astype(np.float32) * 0.35).astype(np.uint8)
            in_title = f"{zoom_f:.1f}x Zoom: {target['planet_key']} ({meta.get('arc_deg', 0):.2f} deg arc)"
            cv2.putText(patch_zoom, in_title, (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 242, 255), 1, cv2.LINE_AA)

            # Place inset in top-left (or bottom-left if planet track is in top-left)
            bx0 = 16
            by0 = 16 if (cy_tr > 235 or cx_tr > 330) else (h - ih_box - 46)
            vis[by0 : by0 + ih_box, bx0 : bx0 + iw_box] = patch_zoom
            cv2.rectangle(vis, (bx0, by0), (bx0 + iw_box, by0 + ih_box), (235, 210, 75), 1, cv2.LINE_AA)

    # Compass rose (North UP, East LEFT)
    cx_r, cy_r = w - 42, 46
    cv2.arrowedLine(vis, (cx_r, cy_r), (cx_r, cy_r - 24), (235, 215, 90), 1, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(vis, (cx_r, cy_r), (cx_r - 24, cy_r), (235, 215, 90), 1, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(vis, "N", (cx_r - 4, cy_r - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 215, 90), 1, cv2.LINE_AA)
    cv2.putText(vis, "E", (cx_r - 36, cy_r + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 215, 90), 1, cv2.LINE_AA)

    bar = vis[h - 32 : h, :].copy()
    vis[h - 32 : h, :] = (bar.astype(np.float32) * 0.40).astype(np.uint8)
    plate = (
        f"{target['planet_key']} (mag {meta.get('mean_mag', 0):+.2f}, {meta.get('dist_str', '')})  |  "
        f"{meta.get('total_nights', len(epochs))} Clear Nights ({meta.get('arc_deg', 0):.2f} deg Drift)  |  "
        f"{n_frames}x30s ({exp_minutes:.0f} min)  |  J2000 {dsa.format_ra_hms(meta['ra0_deg'])} {dsa.format_dec_dms(meta['dec0_deg'])}"
    )
    cv2.putText(vis, plate, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (226, 236, 248), 1, cv2.LINE_AA)
    return vis


def render_solar_system_item(item: Dict[str, Any]) -> None:
    """Re-renders `*_stack.jpg`, `*_annotated.jpg`, and `*_single.jpg` from saved `<tid>_linear.npz`."""
    import deep_sky_astrophoto as dsa

    tid = item["id"]
    cat_map = {t["id"]: t for t in SOLAR_SYSTEM_CATALOG}
    target = cat_map.get(tid)
    npz_p = os.path.join(LINEAR_DIR, f"{tid}_linear.npz")
    if not target or not os.path.exists(npz_p):
        return

    d = np.load(npz_p, allow_pickle=False)
    st_raw = dsa.align_chromatic_channels_rgb(d["stack_rgb"].astype(np.float32))
    me_raw = dsa.align_chromatic_channels_rgb(d["multi_epoch_rgb"].astype(np.float32))
    sg_raw = dsa.align_chromatic_channels_rgb(d["single_rgb"].astype(np.float32))
    meta = json.loads(str(d["meta_json"]))

    h, w = st_raw.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg_mask = np.hypot(xx - float(meta.get("mid_x", w * 0.5)), yy - float(meta.get("mid_y", h * 0.5))) >= 42.0

    alpha = float(target.get("stretch_alpha", 32.0))
    sat = float(target.get("sat_boost", 1.22))

    st_bgr = dsa.stretch_astrophoto_rgb(st_raw, bg_mask=bg_mask, alpha=alpha, sat_boost=sat, sky_floor=0.044)
    me_bgr = dsa.stretch_astrophoto_rgb(me_raw, bg_mask=bg_mask, alpha=alpha, sat_boost=sat, sky_floor=0.044)
    sg_bgr = dsa.stretch_astrophoto_rgb(sg_raw, bg_mask=bg_mask, alpha=alpha, sat_boost=1.18, sky_floor=0.044, is_single_frame=True)

    ann_bgr = _annotate_solar_system_plate(
        me_bgr, target, meta, int(item["frames_stacked"]), float(item["integration_minutes"])
    )

    cv2.imwrite(os.path.join(ASTRO_DIR, f"{tid}_stack.jpg"), st_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(os.path.join(ASTRO_DIR, f"{tid}_annotated.jpg"), ann_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    cv2.imwrite(os.path.join(ASTRO_DIR, f"{tid}_single.jpg"), sg_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


# ==============================================================================
# PILLAR C: EARTH-ORBITAL & ALL-SKY COMPOSITE SHOWCASES (1200 x 900)
# ==============================================================================

def _find_best_continuous_clear_run(
    cfg: dict,
    archive_root: str,
    target_len: int = 156,
) -> List[Tuple[str, float]]:
    """
    Finds the cleanest continuous moonless astronomical-dark sequence of `target_len` consecutive
    30-second 3552x3552 exposures in `archive_root` (scanning Oct 21 / Oct 20 / latest clear nights first).
    """
    import deep_sky_astrophoto as dsa

    obs = _observer_for_cfg(cfg)
    sun = ephem.Sun()
    moon = ephem.Moon()

    day_dirs: List[str] = []
    if os.path.isdir(archive_root):
        base_n = os.path.basename(archive_root.rstrip("\\/"))
        if base_n.isdigit() and len(base_n) == 2:
            day_dirs = sorted(glob.glob(os.path.join(archive_root, "*")), reverse=True)
        else:
            for m_dir in sorted(glob.glob(os.path.join(archive_root, "*")), reverse=True):
                if os.path.isdir(m_dir):
                    mn = os.path.basename(m_dir)
                    if mn.isdigit() and int(mn) < 10 and base_n == "2026":
                        continue
                    day_dirs.extend(sorted(glob.glob(os.path.join(m_dir, "*")), reverse=True))

    best_seq: List[Tuple[str, float]] = []
    for d_dir in day_dirs:
        if not os.path.isdir(d_dir):
            continue
        files = sorted(glob.glob(os.path.join(d_dir, "image_*.jpg")))
        if len(files) < target_len:
            continue

        dark_seq: List[Tuple[str, float]] = []
        for fp in files:
            b = os.path.basename(fp)
            if len(b) >= 12 and b[6:8].isdigit():
                hh = int(b[6:8])
                if 6 <= hh <= 19:
                    continue
            ts = dsa.file_utc_midpoint(fp)
            obs.date = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y/%m/%d %H:%M:%S")
            sun.compute(obs)
            moon.compute(obs)
            if math.degrees(float(sun.alt)) <= -18.0 and (math.degrees(float(moon.alt)) <= -1.0 or float(moon.phase) < 8.0):
                dark_seq.append((fp, ts))

        if len(dark_seq) < target_len:
            continue

        # Slide window in steps of 24 frames, verifying continuity (<65s gap) and 100% clear sky every 12th frame
        for s_i in range(0, len(dark_seq) - target_len + 1, 24):
            cand = dark_seq[s_i : s_i + target_len]
            if (cand[-1][1] - cand[0][1]) > target_len * 44.0:
                continue
            ok = True
            star_scores = []
            for p_i in range(0, target_len, 14):
                im = cv2.imread(cand[p_i][0], cv2.IMREAD_REDUCED_GRAYSCALE_2)
                if im is None:
                    ok = False
                    break
                qh, qw = im.shape[:2]
                med = float(np.median(im[qh // 4 : 3 * qh // 4, qw // 4 : 3 * qw // 4]))
                if med > 45.5:
                    ok = False
                    break
                hp = cv2.subtract(im, cv2.GaussianBlur(im, (9, 9), 0))
                st = int(np.count_nonzero(hp[qh // 4 : 3 * qh // 4, qw // 4 : 3 * qw // 4] > 18))
                if st < 1950:
                    ok = False
                    break
                star_scores.append(st)
            if ok and star_scores:
                if len(cand) >= target_len:
                    return cand
                if len(cand) > len(best_seq):
                    best_seq = cand

    return best_seq


def _remove_smooth_sky_background_rgb(rgb: np.ndarray, gw: int = 64, gh: int = 48) -> np.ndarray:
    """Fast morphological-median + Gaussian smooth atmospheric sky-glow subtraction."""
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (gw, gh), interpolation=cv2.INTER_AREA)
    eroded = cv2.erode(small, np.ones((3, 3), dtype=np.uint8))
    med = cv2.medianBlur(eroded, 5)
    smooth = cv2.GaussianBlur(med, (0, 0), sigmaX=4.5, sigmaY=4.5)
    bg = cv2.resize(smooth, (w, h), interpolation=cv2.INTER_CUBIC)
    return (rgb - bg).astype(np.float32)


def _match_cached_geo_tles(
    mid_ts: float,
    cfg: dict,
    az_min: float,
    az_max: float,
    alt_min: float,
    alt_max: float,
    w: int,
    h: int,
) -> List[Dict[str, Any]]:
    """
    Propagates cached CelesTrak / McCants TLEs (`active.tle`, `visual.tle`, `inttles.tle`, `classfd.tle`)
    at `mid_ts` to locate true Geostationary satellites (`0.95 <= mean_motion <= 1.05 rev/day`) inside
    our Southern Clarke Belt window.
    """
    obs = _observer_for_cfg(cfg)
    obs.date = datetime.datetime.fromtimestamp(mid_ts, tz=datetime.timezone.utc).strftime("%Y/%m/%d %H:%M:%S")
    seen_norad = set()
    matches: List[Dict[str, Any]] = []

    for fname in ("visual.tle", "active.tle", "inttles.tle", "classfd.tle"):
        fp = os.path.join(TLE_CACHE_DIR, fname)
        if not os.path.exists(fp):
            continue
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            continue

        i = 0
        while i + 2 < len(lines):
            l0, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
            if not (l1.startswith("1 ") and l2.startswith("2 ") and len(l2) >= 63):
                i += 1
                continue
            i += 3
            try:
                mm = float(l2[52:63])
                inc = float(l2[8:16])
                norad = l2[2:7].strip()
            except Exception:
                continue
            if not (0.96 <= mm <= 1.04 and inc <= 8.5):
                continue
            if norad in seen_norad:
                continue
            seen_norad.add(norad)
            try:
                sat = ephem.readtle(l0, l1, l2)
                sat.compute(obs)
                if sat.eclipsed:
                    continue
                az_d = math.degrees(float(sat.az))
                alt_d = math.degrees(float(sat.alt))
                if (az_min + 1.5) <= az_d <= (az_max - 1.5) and (alt_min + 1.0) <= alt_d <= (alt_max - 1.5):
                    col = ((az_d - az_min) / (az_max - az_min)) * w
                    row = ((alt_max - alt_d) / (alt_max - alt_min)) * h
                    clean_nm = re.sub(r"\s*\[[^\]]*\]\s*$", "", l0).strip()
                    matches.append({
                        "name": clean_nm,
                        "norad": norad,
                        "az": round(az_d, 2),
                        "alt": round(alt_d, 2),
                        "x": round(col, 2),
                        "y": round(row, 2),
                        "range_km": int(round(float(sat.range) / 1000.0)),
                        "inc": round(inc, 2),
                    })
            except Exception:
                continue

    return matches


def _build_geo_clarke_belt_plate(
    seq: List[Tuple[str, float]],
    cfg: dict,
    poly_cx: np.ndarray,
    poly_cy: np.ndarray,
    srgb_lut: np.ndarray,
) -> Dict[str, Any]:
    """
    Builds `orb_geo_clarke_belt`: Earth-fixed Southern Clarke Belt view (Az = 156..204 deg, Alt = 30.8..66.8 deg,
    1200x900) combining 3-frame temporal-gated airplane-free equatorial star trails with persistent stationary
    Geostationary satellites at 35,786 km altitude!
    """
    tid = "orb_geo_clarke_belt"
    use_seq = seq[: min(130, len(seq))]
    az_min, az_max = 156.0, 204.0
    alt_min, alt_max = 30.8, 66.8

    az_1d = np.linspace(az_min, az_max, ORB_W, dtype=np.float32)
    alt_1d = np.linspace(alt_max, alt_min, ORB_H, dtype=np.float32)
    az_g, alt_g = np.meshgrid(np.radians(az_1d), np.radians(alt_1d))
    cos_a = np.cos(alt_g)
    east = (cos_a * np.sin(az_g)).astype(np.float32)
    north = (cos_a * np.cos(az_g)).astype(np.float32)
    up = np.sin(alt_g).astype(np.float32)

    mx_3552, my_3552, _ = _enu_to_sensor_xy(east, north, up, cfg, poly_cx, poly_cy)

    if HAS_CUDA:
        gx_g = 2.0 * (torch.from_numpy(mx_3552).cuda() / 3551.0) - 1.0
        gy_g = 2.0 * (torch.from_numpy(my_3552).cuda() / 3551.0) - 1.0
        grid_g = torch.stack([gx_g, gy_g], dim=-1).unsqueeze(0)

    trail_max = np.zeros((ORB_H, ORB_W, 3), dtype=np.float32)
    trail_sum = np.zeros((ORB_H, ORB_W, 3), dtype=np.float32)
    geo_samples: List[np.ndarray] = []
    single_ref: Optional[np.ndarray] = None

    k_gate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    prev_clean: Optional[np.ndarray] = None
    curr_clean: Optional[np.ndarray] = None
    prev_mask: Optional[np.ndarray] = None
    curr_mask: Optional[np.ndarray] = None
    valid_count = 0

    for i, (fp, _) in enumerate(use_seq):
        bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb_lin = _suppress_sensor_hot_pixels_rgb(srgb_lut[bgr[:, :, ::-1]])
        if HAS_CUDA:
            with torch.no_grad():
                t_im = torch.from_numpy(rgb_lin).cuda().permute(2, 0, 1).unsqueeze(0)
                warped = torch.clamp(
                    F.grid_sample(t_im, grid_g, mode="bicubic", padding_mode="zeros", align_corners=True),
                    0.0,
                    1.5,
                )[0].permute(1, 2, 0).cpu().numpy()
                del t_im
        else:
            warped = cv2.remap(rgb_lin, mx_3552, my_3552, interpolation=cv2.INTER_CUBIC)

        clean = _remove_smooth_sky_background_rgb(warped, gw=64, gh=48)
        if single_ref is None and i >= len(use_seq) // 3:
            single_ref = clean.copy()

        # Sample every 4th frame (~2.3 min apart -> stars drift >13 px so median eliminates 100% of stars!)
        if valid_count % 4 == 0:
            geo_samples.append(clean.astype(np.float32))
        valid_count += 1

        lum = 0.30 * clean[:, :, 0] + 0.50 * clean[:, :, 1] + 0.20 * clean[:, :, 2]
        mad = float(np.median(np.abs(lum - np.median(lum))))
        sig = max(1.4826 * mad, 1e-5)
        m_now = ((lum > 2.4 * sig).astype(np.uint8)) * 255

        if prev_clean is not None and curr_clean is not None and prev_mask is not None and curr_mask is not None:
            gate = (cv2.dilate(prev_mask, k_gate) > 0) & (cv2.dilate(m_now, k_gate) > 0)
            gate_f = cv2.GaussianBlur(gate.astype(np.float32), (0, 0), sigmaX=1.1, sigmaY=1.1)[:, :, None]
            gated_curr = np.maximum(curr_clean, 0.0) * gate_f
            trail_max = np.maximum(trail_max, gated_curr)
            trail_sum += gated_curr

        prev_clean, curr_clean = curr_clean, clean
        prev_mask, curr_mask = curr_mask, m_now

    if single_ref is None:
        single_ref = curr_clean if curr_clean is not None else np.zeros((ORB_H, ORB_W, 3), dtype=np.float32)

    # Stationary Earth-fixed temporal median / trimmed-mean isolates true 35,786 km Geostationary satellites!
    if len(geo_samples) >= 5:
        g_arr = np.sort(np.stack(geo_samples, axis=0), axis=0)
        n_g = len(g_arr)
        geo_stat = g_arr[int(0.25 * n_g) : max(int(0.25 * n_g) + 1, int(0.80 * n_g))].mean(axis=0).astype(np.float32)
    else:
        geo_stat = np.zeros_like(trail_max)

    # Extract clean high-pass stationary point sources from `geo_stat` so background remains velvet smooth
    geo_hp = geo_stat - cv2.GaussianBlur(geo_stat, (0, 0), sigmaX=4.2, sigmaY=4.2)
    g_lum = 0.30 * geo_hp[:, :, 0] + 0.50 * geo_hp[:, :, 1] + 0.20 * geo_hp[:, :, 2]
    g_sig = max(1.4826 * float(np.median(np.abs(g_lum - np.median(g_lum)))), 1e-5)
    geo_pinpoints = np.maximum(geo_hp - 2.3 * g_sig, 0.0)

    # Combine deep stationary dark-sky floor + silky equatorial star trails + boosted stationary GEO pinpoints
    stack_rgb = (
        geo_stat
        + 0.85 * trail_max
        + 0.14 * (trail_sum / max(1.0, valid_count * 0.11))
        + 3.4 * geo_pinpoints
    ).astype(np.float32)

    mid_ts = use_seq[len(use_seq) // 2][1]
    tle_geos = _match_cached_geo_tles(mid_ts, cfg, az_min, az_max, alt_min, alt_max, ORB_W, ORB_H)

    # Match TLE GEOs against real optical peaks in `g_lum` (within 11 px) or top optical peaks along Clarke Belt
    verified_geos: List[Dict[str, Any]] = []
    for tg in tle_geos:
        ix, iy = int(round(tg["x"])), int(round(tg["y"]))
        r_s = 11
        y0, y1 = max(0, iy - r_s), min(ORB_H, iy + r_s + 1)
        x0, x1 = max(0, ix - r_s), min(ORB_W, ix + r_s + 1)
        patch = g_lum[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        _, maxv, _, maxl = cv2.minMaxLoc(patch)
        snr_v = float(maxv / g_sig)
        if snr_v >= 3.2:
            tg_c = dict(tg)
            tg_c["x"] = round(float(x0 + maxl[0]), 1)
            tg_c["y"] = round(float(y0 + maxl[1]), 1)
            tg_c["snr"] = round(snr_v, 1)
            verified_geos.append(tg_c)

    verified_geos.sort(key=lambda d: d["snr"], reverse=True)
    # Keep well-spaced top labels (>= 62 px horizontal separation) for crisp, clutter-free plate typography
    spaced_geos: List[Dict[str, Any]] = []
    # Prioritize iconic named spacecraft (GOES, ANIK, ECHOSTAR, DIRECTV, GALAXY, SES, VIASAT, TDRS)
    def _geo_prio(g: Dict[str, Any]) -> float:
        nm = g["name"].upper()
        bonus = 4.0 if any(k in nm for k in ("GOES", "ANIK", "DIRECTV", "ECHOSTAR", "GALAXY", "SES", "VIASAT", "TDRS", "INTELSAT")) else 0.0
        return float(g["snr"]) + bonus

    for cand in sorted(verified_geos, key=_geo_prio, reverse=True):
        if all(abs(cand["x"] - s["x"]) >= 68.0 or math.hypot(cand["x"] - s["x"], cand["y"] - s["y"]) >= 72.0 for s in spaced_geos):
            spaced_geos.append(cand)
        if len(spaced_geos) >= 9:
            break
    spaced_geos.sort(key=lambda d: d["x"])

    # Compute exact theoretical Clarke Belt curve (geocentric equatorial ring r = 42,164.2 km) in (col, row)
    lat_r = math.radians(float(cfg.get("camera_lat", 45.0000)))
    re_km = 6378.137
    r_geo_km = 42164.17
    obs_z = re_km * math.sin(lat_r)
    obs_x = re_km * math.cos(lat_r)
    belt_pts: List[List[float]] = []
    for ha_deg in np.linspace(-38.0, 38.0, 77):
        ha_r = math.radians(float(ha_deg))
        # Satellite in meridian-aligned geocentric equatorial frame: (r*cos(ha), -r*sin(ha), 0)
        dx_m = r_geo_km * math.cos(ha_r) - obs_x
        dy_e = -r_geo_km * math.sin(ha_r)
        dz_p = -obs_z
        # Convert to local Horizon (East, North, Up)
        east_v = dy_e
        north_v = -math.sin(lat_r) * dx_m + math.cos(lat_r) * dz_p
        up_v = math.cos(lat_r) * dx_m + math.sin(lat_r) * dz_p
        rng = math.sqrt(east_v * east_v + north_v * north_v + up_v * up_v)
        alt_b = math.degrees(math.asin(up_v / rng))
        az_b = math.degrees(math.atan2(east_v, north_v)) % 360.0
        if az_min <= az_b <= az_max and alt_min <= alt_b <= alt_max:
            bx = ((az_b - az_min) / (az_max - az_min)) * ORB_W
            by = ((alt_max - alt_b) / (alt_max - alt_min)) * ORB_H
            belt_pts.append([round(bx, 1), round(by, 1)])

    dur_min = round((use_seq[-1][1] - use_seq[0][1]) / 60.0, 1)
    night_str = datetime.datetime.fromtimestamp(mid_ts).strftime("%Y-%m-%d")
    meta = {
        "kind": "geo_clarke_belt",
        "night": night_str,
        "frames": valid_count,
        "duration_min": dur_min,
        "az_range": [az_min, az_max],
        "alt_range": [alt_min, alt_max],
        "belt_pts": belt_pts,
        "verified_total": len(verified_geos),
        "geos": spaced_geos,
    }
    np.savez_compressed(
        os.path.join(LINEAR_DIR, f"{tid}_linear.npz"),
        stack_rgb=stack_rgb,
        single_rgb=single_ref.astype(np.float32),
        meta_json=np.array(json.dumps(meta)),
    )

    exp_min = round(valid_count * 0.5, 1)
    exp_hrs = round(exp_min / 60.0, 2)
    return {
        "id": tid,
        "name": "Geostationary 'Clarke Belt' Satellite Necklace (35,786 km Orbit)",
        "catalog_code": f"Clarke Belt GEO Ring ({len(verified_geos)} Stationary Sats)",
        "type": "orbital",
        "type_label": "Geostationary Orbit • 35,786 km",
        "constellation": "Southern Equatorial Sky",
        "distance": "35,786 km (22,236 miles)",
        "apparent_mag": "10.5 – 13.0",
        "ra_deg": 0.0,
        "dec_deg": -7.15,
        "ra_hms": "Az 156°→204° S",
        "dec_dms": "Dec -07° 09' (Parallax)",
        "fov_str": "48.0° × 36.0° Earth-Fixed",
        "pixel_scale_arcmin": 2.4,
        "resolution": f"{ORB_W} × {ORB_H} (Earth-Fixed Temporal Median + Trail Gate)",
        "frames_stacked": valid_count,
        "integration_minutes": exp_min,
        "integration_hours": exp_hrs,
        "snr_gain_x": round(math.sqrt(max(len(geo_samples), 1)), 2),
        "limiting_mag_gain": "+2.10 mag (GEO Pinpoints)",
        "max_altitude_deg": 34.7,
        "median_registration_corr": 1.0,
        "description": (
            f"While equatorial stars sweep across 18° in silky airplane-free trails over {dur_min:.0f} minutes, "
            f"communications and weather satellites parked in the 35,786 km (22,236 mi) Clarke Geostationary Belt "
            f"match Earth's rotation 1:1 — appearing as stationary pinpoints of light along Dec -7.15° due South!"
        ),
        "stack_url": f"/astrophoto_data/{tid}_stack.jpg",
        "annotated_url": f"/astrophoto_data/{tid}_annotated.jpg",
        "single_url": f"/astrophoto_data/{tid}_single.jpg",
    }


def _build_polaris_startrail_vortex_plate(
    seq: List[Tuple[str, float]],
    cfg: dict,
    poly_cx: np.ndarray,
    poly_cy: np.ndarray,
    srgb_lut: np.ndarray,
) -> Dict[str, Any]:
    """
    Builds `orb_polaris_startrail_vortex`: 80-minute True-Color Circumpolar Star-Trail Vortex centered above
    Polaris (Az = 0 deg North, Alt = 35.5..77.5 deg, 1200x900) using a 3-frame temporal-coherence gate that
    100% erases airplanes, strobes, meteors, and LEO satellites while preserving pure starlight colors!
    """
    import deep_sky_astrophoto as dsa

    tid = "orb_polaris_startrail_vortex"
    # Tangential gnomonic Earth-fixed camera aimed North at (Az = 0 deg, Alt = 56.5 deg), FOV 56 x 42 deg (100% tree-free!)
    fov_w, fov_h = 56.0, 42.0
    alt0 = math.radians(56.5)
    xi_deg = ((np.arange(ORB_W, dtype=np.float64) + 0.5) / ORB_W - 0.5) * fov_w  # Left = NW (-Az), Right = NE (+Az)
    eta_deg = (0.5 - (np.arange(ORB_H, dtype=np.float64) + 0.5) / ORB_H) * fov_h  # Top = high Alt, Bottom = low Alt
    xi, eta = np.meshgrid(np.tan(np.radians(xi_deg)), np.tan(np.radians(eta_deg)))
    norm = np.sqrt(1.0 + xi * xi + eta * eta)
    # Camera basis at Az=0, Alt=alt0: forward = (0, cos(alt0), sin(alt0)), right = (1, 0, 0), up_cam = (0, -sin(alt0), cos(alt0))
    east = (xi / norm).astype(np.float32)
    north = ((math.cos(alt0) - eta * math.sin(alt0)) / norm).astype(np.float32)
    up = ((math.sin(alt0) + eta * math.cos(alt0)) / norm).astype(np.float32)

    mx_3552, my_3552, _ = _enu_to_sensor_xy(east, north, up, cfg, poly_cx, poly_cy)
    if HAS_CUDA:
        gx_g = 2.0 * (torch.from_numpy(mx_3552).cuda() / 3551.0) - 1.0
        gy_g = 2.0 * (torch.from_numpy(my_3552).cuda() / 3551.0) - 1.0
        grid_g = torch.stack([gx_g, gy_g], dim=-1).unsqueeze(0)

    trail_max = np.zeros((ORB_H, ORB_W, 3), dtype=np.float32)
    trail_sum = np.zeros((ORB_H, ORB_W, 3), dtype=np.float32)
    bg_samples: List[np.ndarray] = []
    single_ref: Optional[np.ndarray] = None

    k_gate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    prev_clean: Optional[np.ndarray] = None
    curr_clean: Optional[np.ndarray] = None
    prev_mask: Optional[np.ndarray] = None
    curr_mask: Optional[np.ndarray] = None
    valid_count = 0

    for i, (fp, _) in enumerate(seq):
        bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb_lin = _suppress_sensor_hot_pixels_rgb(srgb_lut[bgr[:, :, ::-1]])
        if HAS_CUDA:
            with torch.no_grad():
                t_im = torch.from_numpy(rgb_lin).cuda().permute(2, 0, 1).unsqueeze(0)
                warped = torch.clamp(
                    F.grid_sample(t_im, grid_g, mode="bicubic", padding_mode="zeros", align_corners=True),
                    0.0,
                    1.5,
                )[0].permute(1, 2, 0).cpu().numpy()
                del t_im
        else:
            warped = cv2.remap(rgb_lin, mx_3552, my_3552, interpolation=cv2.INTER_CUBIC)

        clean = _remove_smooth_sky_background_rgb(warped, gw=64, gh=48)
        if single_ref is None and i >= len(seq) // 2:
            single_ref = clean.copy()
        if valid_count % 8 == 0:
            bg_samples.append(clean.astype(np.float32))
        valid_count += 1

        lum = 0.30 * clean[:, :, 0] + 0.50 * clean[:, :, 1] + 0.20 * clean[:, :, 2]
        mad = float(np.median(np.abs(lum - np.median(lum))))
        sig = max(1.4826 * mad, 1e-5)
        m_now = ((lum > 2.35 * sig).astype(np.uint8)) * 255

        if prev_clean is not None and curr_clean is not None and prev_mask is not None and curr_mask is not None:
            gate = (cv2.dilate(prev_mask, k_gate) > 0) & (cv2.dilate(m_now, k_gate) > 0)
            gate_f = cv2.GaussianBlur(gate.astype(np.float32), (0, 0), sigmaX=1.05, sigmaY=1.05)[:, :, None]
            gated_curr = np.maximum(curr_clean, 0.0) * gate_f
            trail_max = np.maximum(trail_max, gated_curr)
            trail_sum += gated_curr

        prev_clean, curr_clean = curr_clean, clean
        prev_mask, curr_mask = curr_mask, m_now

    if single_ref is None:
        single_ref = np.zeros((ORB_H, ORB_W, 3), dtype=np.float32)

    bg_base = np.median(np.stack(bg_samples, axis=0), axis=0).astype(np.float32) if bg_samples else np.zeros_like(trail_max)
    stack_rgb = (bg_base + 0.86 * trail_max + 0.14 * (trail_sum / max(1.0, valid_count * 0.13))).astype(np.float32)

    # Project North Celestial Pole & key circumpolar trail stars at midpoint timestamp for annotation
    def _enu_to_vortex_xy(e_v: float, n_v: float, u_v: float) -> Optional[Tuple[float, float]]:
        z_cam = n_v * math.cos(alt0) + u_v * math.sin(alt0)
        if z_cam <= 0.1:
            return None
        xi_v = e_v / z_cam
        eta_v = (-n_v * math.sin(alt0) + u_v * math.cos(alt0)) / z_cam
        col = (math.degrees(math.atan(xi_v)) / fov_w + 0.5) * ORB_W
        row = (0.5 - math.degrees(math.atan(eta_v)) / fov_h) * ORB_H
        if 8 <= col <= ORB_W - 8 and 8 <= row <= ORB_H - 36:
            return round(col, 1), round(row, 1)
        return None

    lat_r = math.radians(float(cfg.get("camera_lat", 45.0000)))
    ncp_xy = _enu_to_vortex_xy(0.0, math.cos(lat_r), math.sin(lat_r))
    mid_ts = seq[len(seq) // 2][1]
    obs = _observer_for_cfg(cfg)
    obs.date = datetime.datetime.fromtimestamp(mid_ts, tz=datetime.timezone.utc).strftime("%Y/%m/%d %H:%M:%S")
    star_labels = []
    for s_nm, disp_nm in (
        ("Polaris", "Polaris (alpha UMi)"),
        ("Kochab", "Kochab (Ursa Minor)"),
        ("Alderamin", "Alderamin (Cepheus)"),
        ("Caph", "Caph (Cassiopeia)"),
        ("Schedar", "Schedar (Cassiopeia)"),
        ("Dubhe", "Dubhe (Ursa Major)"),
        ("Eltanin", "Eltanin (Draco)"),
    ):
        try:
            st = ephem.star(s_nm)
            st.compute(obs)
            az_s, alt_s = float(st.az), float(st.alt)
            xy_s = _enu_to_vortex_xy(math.cos(alt_s) * math.sin(az_s), math.cos(alt_s) * math.cos(az_s), math.sin(alt_s))
            if xy_s is not None:
                star_labels.append({"label": disp_nm, "x": xy_s[0], "y": xy_s[1]})
        except Exception:
            continue

    dur_min = round((seq[-1][1] - seq[0][1]) / 60.0, 1)
    rot_deg = round(dur_min * 0.25068, 1)
    meta = {
        "kind": "polaris_vortex",
        "night": datetime.datetime.fromtimestamp(mid_ts).strftime("%Y-%m-%d"),
        "frames": valid_count,
        "duration_min": dur_min,
        "rotation_deg": rot_deg,
        "ncp_xy": list(ncp_xy) if ncp_xy else [600.0, 636.0],
        "stars": star_labels,
    }
    np.savez_compressed(
        os.path.join(LINEAR_DIR, f"{tid}_linear.npz"),
        stack_rgb=stack_rgb,
        single_rgb=single_ref.astype(np.float32),
        meta_json=np.array(json.dumps(meta)),
    )

    exp_min = round(valid_count * 0.5, 1)
    exp_hrs = round(exp_min / 60.0, 2)
    return {
        "id": tid,
        "name": f"True-Color Circumpolar Polaris Star-Trail Vortex ({rot_deg}° Earth Rotation)",
        "catalog_code": "North Celestial Pole • 100% Airplane-Free",
        "type": "orbital",
        "type_label": "Circumpolar Star Vortex • Temporal Gate",
        "constellation": "Ursa Minor / Cepheus / Cassiopeia",
        "distance": "433 ly (Polaris)",
        "apparent_mag": "1.98 (Polaris)",
        "ra_deg": 37.9545,
        "dec_deg": 89.2641,
        "ra_hms": dsa.format_ra_hms(37.9545),
        "dec_dms": dsa.format_dec_dms(89.2641),
        "fov_str": f"56.0° × 42.0° North ({rot_deg}° Arcs)",
        "pixel_scale_arcmin": 2.8,
        "resolution": f"{ORB_W} × {ORB_H} (3-Frame Temporal Airplane Erasure)",
        "frames_stacked": valid_count,
        "integration_minutes": exp_min,
        "integration_hours": exp_hrs,
        "snr_gain_x": round(math.sqrt(max(len(bg_samples), 1)), 2),
        "limiting_mag_gain": "0 Airplanes / 100% Pure Starlight",
        "max_altitude_deg": 48.1,
        "median_registration_corr": 1.0,
        "description": (
            f"Continuous {dur_min:.0f}-minute ({valid_count}×30s) northern exposure centered above Polaris at 48.1° "
            f"altitude. A 3-frame temporal-coherence gate completely erases every passing airliner strobe and satellite "
            f"streak while preserving pure sapphire, pearl-white, golden, and garnet stellar spectral arcs."
        ),
        "stack_url": f"/astrophoto_data/{tid}_stack.jpg",
        "annotated_url": f"/astrophoto_data/{tid}_annotated.jpg",
        "single_url": f"/astrophoto_data/{tid}_single.jpg",
    }


class _ArchiveFrameLookup:
    """Fast cached resolver mapping DB `image-YYYYMMDDHHMMSS.jpg` to untouched 3552x3552 files on `F:\\AllSky_Archive`."""

    def __init__(self, archive_root: str) -> None:
        self.year_root = archive_root
        if os.path.basename(archive_root.rstrip("\\/")).isdigit() and len(os.path.basename(archive_root.rstrip("\\/"))) == 2:
            self.year_root = os.path.dirname(archive_root.rstrip("\\/"))
        self._cache: Dict[str, List[Tuple[int, str]]] = {}

    @staticmethod
    def _sec_of_day(hh: int, mm: int, ss: int) -> int:
        return hh * 3600 + mm * 60 + ss

    def _day_entries(self, yyyy: str, mm: str, dd: str) -> List[Tuple[int, str]]:
        key = f"{yyyy}-{mm}-{dd}"
        if key not in self._cache:
            d_dir = os.path.join(self.year_root, mm, dd) if os.path.basename(self.year_root) == yyyy else os.path.join(self.year_root, yyyy, mm, dd)
            entries: List[Tuple[int, str]] = []
            if os.path.isdir(d_dir):
                for fp in sorted(glob.glob(os.path.join(d_dir, "image_*.jpg"))):
                    b = os.path.basename(fp)
                    if len(b) >= 12 and b[6:12].isdigit():
                        sec = self._sec_of_day(int(b[6:8]), int(b[8:10]), int(b[10:12]))
                        entries.append((sec, fp))
            self._cache[key] = entries
        return self._cache[key]

    def resolve(self, db_frame_filename: str, max_tol_sec: int = 20) -> Optional[Tuple[str, List[str]]]:
        """
        Resolves `'image-YYYYMMDDHHMMSS.jpg'` to `(matched_fp, neighbor_fps)` on `F:\\AllSky_Archive`.
        """
        m = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", db_frame_filename or "")
        if not m:
            return None
        yyyy, mm, dd, hh, mi, ss = m.groups()
        target_sec = self._sec_of_day(int(hh), int(mi), int(ss))
        entries = self._day_entries(yyyy, mm, dd)
        if not entries:
            return None

        best_idx = -1
        best_dt = 999999
        for idx, (sec, _) in enumerate(entries):
            dt = abs(sec - target_sec)
            if dt < best_dt:
                best_dt = dt
                best_idx = idx
        if best_idx < 0 or best_dt > max_tol_sec:
            return None

        matched_fp = entries[best_idx][1]
        neighbors: List[str] = []
        for off in (-2, -1, 1, 2):
            ni = best_idx + off
            if 0 <= ni < len(entries) and abs(entries[ni][0] - entries[best_idx][0]) <= 110:
                neighbors.append(entries[ni][1])
        return matched_fp, neighbors


def _build_allsky_dome_projection(
    cfg: dict,
    poly_cx: np.ndarray,
    poly_cy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Precomputes the symmetric 360-deg 1200x900 North-Up / East-Left All-Sky Celestial Dome projection
    (`Alt = DOME_MIN_ALT..90 deg`) without panoramic rectangular zero-blocks, while smoothly masking out
    the floodlit East house roof gable (`Az = 72..126 deg, Alt < 39 deg`).
    Returns `(mx_3552, my_3552, dome_mask_weight, r_out)`.
    """
    yy, xx = np.mgrid[0:ORB_H, 0:ORB_W].astype(np.float32)
    dx = xx - DOME_CX
    dy = yy - DOME_CY
    r_out = np.hypot(dx, dy)
    alt_deg = np.clip(90.0 - (r_out / DOME_R) * (90.0 - DOME_MIN_ALT), 5.0, 90.0)
    az_rad = np.arctan2(-dx, -dy)  # North=UP, East=LEFT, South=DOWN, West=RIGHT
    az_deg = np.mod(np.degrees(az_rad), 360.0)
    alt_rad = np.radians(alt_deg)

    east = (np.cos(alt_rad) * np.sin(az_rad)).astype(np.float32)
    north = (np.cos(alt_rad) * np.cos(az_rad)).astype(np.float32)
    up = np.sin(alt_rad).astype(np.float32)

    mx_3552, my_3552, _ = _enu_to_sensor_xy(east, north, up, cfg, poly_cx, poly_cy)
    in_sensor = (
        (mx_3552 >= 24.0) & (mx_3552 <= 3528.0) & (my_3552 >= 24.0) & (my_3552 <= 3528.0)
    ).astype(np.float32)

    rim_taper = np.clip((DOME_R - r_out) / 14.0, 0.0, 1.0)
    w_tot = (in_sensor * rim_taper).astype(np.float32)
    return mx_3552, my_3552, w_tot, r_out


def _nyquist_broaden_streaks_3552(canvas_3552: np.ndarray, gain: float = 4.4) -> np.ndarray:
    """
    Anti-aliased Nyquist PSF broadening on 3552x3552 before 4.1x downsampling (`cv2.remap` -> 1200x900)
    with gentle highlight compression so both 25-sigma LEO rocket bodies and 136-sigma bolides gleam
    as unbroken, razor-sharp luminous arcs across the All-Sky Celestial Dome!
    """
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dil = cv2.dilate(canvas_3552, k7)
    smooth = cv2.GaussianBlur(dil, (0, 0), sigmaX=2.2, sigmaY=2.2)
    # Soft toe boost lifts faint high-orbit / slender tracks without clipping bright bolide cores
    lum = np.maximum(0.30 * smooth[:, :, 0] + 0.50 * smooth[:, :, 1] + 0.20 * smooth[:, :, 2], 1e-7)
    boosted_lum = gain * (lum / (1.0 + 1.6 * lum))
    return (smooth * (boosted_lum / lum)[:, :, None]).astype(np.float32)


def _az_alt_to_dome_xy(az_deg: float, alt_deg: float) -> Tuple[float, float]:
    r = ((90.0 - max(DOME_MIN_ALT, min(90.0, alt_deg))) / (90.0 - DOME_MIN_ALT)) * DOME_R
    az_r = math.radians(az_deg)
    return float(DOME_CX - r * math.sin(az_r)), float(DOME_CY - r * math.cos(az_r))


def _build_deep_allsky_starry_base(
    base_files: List[str],
    mx_3552: np.ndarray,
    my_3552: np.ndarray,
    w_dome: np.ndarray,
    srgb_lut: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Co-adds ~16-20 clear moonless 3552x3552 frames into a smooth, noise-free, airplane-free
    1200x900 All-Sky Celestial Dome starry sky base (plus returns a single 30s frame).
    """
    import deep_sky_astrophoto as dsa

    stack_list: List[np.ndarray] = []
    single_dome: Optional[np.ndarray] = None
    active_m = (w_dome > 0.03)[:, :, None].astype(np.float32)

    for fp in base_files:
        bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb_lin = _suppress_sensor_hot_pixels_rgb(srgb_lut[bgr[:, :, ::-1]])
        warped = cv2.remap(rgb_lin, mx_3552, my_3552, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        # Subtract only gentle broad horizon airglow (sigma ~ 35 px on 1200x900) so Milky Way band is preserved!
        sh, sw = 45, 60
        sm = cv2.resize(warped, (sw, sh), interpolation=cv2.INTER_AREA)
        sm_w = cv2.resize((w_dome > 0.35).astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
        er = cv2.erode(sm, np.ones((3, 3), dtype=np.uint8))
        bg_s = cv2.GaussianBlur(cv2.medianBlur(er, 5) * sm_w[:, :, None], (0, 0), sigmaX=6.5, sigmaY=6.5)
        w_b = cv2.GaussianBlur(sm_w, (0, 0), sigmaX=6.5, sigmaY=6.5)
        dome_bg = cv2.resize(bg_s / np.maximum(w_b[:, :, None], 1e-4), (ORB_W, ORB_H), interpolation=cv2.INTER_CUBIC)
        clean = (warped - dome_bg) * active_m * (w_dome[:, :, None] ** 0.5)
        if single_dome is None:
            single_dome = clean.copy()
        stack_list.append(clean.astype(np.float32))

    if not stack_list:
        z = np.zeros((ORB_H, ORB_W, 3), dtype=np.float32)
        return z, z
    base_master = dsa.gpu_sigma_clipped_mean(stack_list, low_sigma=2.2, high_sigma=2.1)

    # Smoothly silhouette ONLY the exact floodlit East house gable pixels (x < DOME_CX - 150 where static wall glows > 0.024)
    yy_d, xx_d = np.mgrid[0:ORB_H, 0:ORB_W].astype(np.float32)
    east_sector = (xx_d < (DOME_CX - 150.0)) & (np.abs(yy_d - DOME_CY) < 235.0)
    wall_raw = ((np.max(base_master, axis=-1) > 0.024) & east_sector).astype(np.uint8)
    if np.any(wall_raw):
        wall_dil = cv2.dilate(wall_raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))
        wall_feather = cv2.GaussianBlur(wall_dil.astype(np.float32), (0, 0), sigmaX=5.5, sigmaY=5.5)[:, :, None]
        sil_scale = 1.0 - 0.994 * np.clip(wall_feather * 1.25, 0.0, 1.0)
        base_master = (base_master * sil_scale).astype(np.float32)
        if single_dome is not None:
            single_dome = (single_dome * sil_scale).astype(np.float32)

    return base_master, (single_dome if single_dome is not None else base_master)


def _capsule_mask_3552(
    fj: dict,
    pad_px: float = 42.0,
) -> Optional[Tuple[int, int, int, int, np.ndarray]]:
    """
    Builds a localized `(y0, y1, x0, x1, mask_patch)` feathered corridor on 3552x3552 strictly around
    the detected streak segment `(p1, p2)` or `bbox` so zero unrelated aircraft elsewhere in the frame can leak in!
    """
    bbox = fj.get("bbox")
    p1 = fj.get("p1")
    p2 = fj.get("p2")
    if not bbox or len(bbox) != 4:
        return None

    bx1, by1, bx2, by2 = [float(v) for v in bbox]
    margin = int(math.ceil(pad_px)) + 16
    x0 = max(0, int(math.floor(min(bx1, bx2))) - margin)
    x1 = min(3552, int(math.ceil(max(bx1, bx2))) + margin)
    y0 = max(0, int(math.floor(min(by1, by2))) - margin)
    y1 = min(3552, int(math.ceil(max(by1, by2))) + margin)
    if x1 <= x0 + 4 or y1 <= y0 + 4:
        return None

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    eff_pad = min(pad_px, 16.0) if (p1 and p2 and len(p1) == 2 and len(p2) == 2) else min(pad_px, 24.0)
    if p1 and p2 and len(p1) == 2 and len(p2) == 2:
        ax, ay = float(p1[0]), float(p1[1])
        bx, by = float(p2[0]), float(p2[1])
        vx, vy = bx - ax, by - ay
        l2 = max(vx * vx + vy * vy, 1.0)
        t_proj = np.clip(((xx - ax) * vx + (yy - ay) * vy) / l2, 0.0, 1.0)
        dist = np.hypot(xx - (ax + t_proj * vx), yy - (ay + t_proj * vy))
        m = np.clip((eff_pad - dist) / 6.0, 0.0, 1.0)
    else:
        dx = np.maximum(np.maximum(bx1 - xx, 0.0), xx - bx2)
        dy = np.maximum(np.maximum(by1 - yy, 0.0), yy - by2)
        dist = np.hypot(dx, dy)
        m = np.clip((eff_pad - dist) / 7.0, 0.0, 1.0)

    return y0, y1, x0, x1, (0.5 - 0.5 * np.cos(math.pi * m)).astype(np.float32)


def _extract_transient_excess_3552(
    event_fp: str,
    neighbor_fps: List[str],
    fj: dict,
    srgb_lut: np.ndarray,
    pad_px: float = 42.0,
    max_center_med: float = 52.0,
) -> Optional[Tuple[int, int, int, int, np.ndarray, np.ndarray]]:
    """
    Extracts ONLY the genuine optical streak/bolide excess `(rgb_event - ref_neighbors)` inside
    the localized event corridor on 3552x3552, completely suppressing drifted background stars.
    Returns `(y0, y1, x0, x1, excess_patch_rgb, full_event_rgb_or_none)`.
    """
    cap = _capsule_mask_3552(fj, pad_px=pad_px)
    if cap is None or not neighbor_fps:
        return None
    y0, y1, x0, x1, mask_p = cap

    bgr_ev = cv2.imread(event_fp, cv2.IMREAD_COLOR)
    if bgr_ev is None or bgr_ev.shape[0] != 3552:
        return None
    c_med = float(np.median(bgr_ev[888:2664, 888:2664, 1]))
    if c_med > max_center_med:
        return None

    ev_raw = srgb_lut[bgr_ev[y0:y1, x0:x1, ::-1]]
    # Remove each frame's own smooth sky-glow pedestal (sigma=12 px) FIRST so dawn/dusk twilight changes between t-1 and t+1 cancel 100%!
    ev_hp = np.maximum(ev_raw - cv2.GaussianBlur(ev_raw, (0, 0), sigmaX=12.0, sigmaY=12.0), 0.0)

    nb_hps: List[np.ndarray] = []
    for nfp in neighbor_fps[:3]:
        nb_bgr = cv2.imread(nfp, cv2.IMREAD_COLOR)
        if nb_bgr is not None and nb_bgr.shape[0] == 3552:
            nb_r = srgb_lut[nb_bgr[y0:y1, x0:x1, ::-1]]
            nb_hps.append(np.maximum(nb_r - cv2.GaussianBlur(nb_r, (0, 0), sigmaX=12.0, sigmaY=12.0), 0.0))
    if not nb_hps:
        return None

    # Pixelwise minimum of immediate high-pass neighbors preserves 100% of background stars while excluding adjacent pass segments!
    if len(nb_hps) == 1:
        ref_hp = nb_hps[0]
    elif len(nb_hps) == 2:
        ref_hp = np.minimum(nb_hps[0], nb_hps[1])
    else:
        ref_hp = np.sort(np.stack(nb_hps, axis=0), axis=0)[1].astype(np.float32)

    # 9x9 (+/-4 px) sidereal-drift star envelope erases drifted background stars on a 0.0 baseline without touching blank sky!
    k_star = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ref_star_env = cv2.GaussianBlur(cv2.dilate(ref_hp, k_star), (0, 0), sigmaX=1.1, sigmaY=1.1)
    diff = ev_hp - 1.15 * ref_star_env

    lum_d = 0.30 * diff[:, :, 0] + 0.50 * diff[:, :, 1] + 0.20 * diff[:, :, 2]
    lum_ev = 0.30 * ev_hp[:, :, 0] + 0.50 * ev_hp[:, :, 1] + 0.20 * ev_hp[:, :, 2]
    mad = float(np.median(np.abs(lum_ev - np.median(lum_ev))))
    sig = max(1.4826 * mad, 1.5e-4)

    # Keep only contiguous streak/bolide connected components (area >= 12 px, span >= 6 px) -> zero isolated star specks!
    binary = ((lum_d > 2.4 * sig) & (mask_p > 0.10)).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n_lbl, lbl_im, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_bin = np.zeros_like(binary, dtype=np.float32)
    for k_i in range(1, n_lbl):
        w_cc = int(stats[k_i, cv2.CC_STAT_WIDTH])
        h_cc = int(stats[k_i, cv2.CC_STAT_HEIGHT])
        area_cc = int(stats[k_i, cv2.CC_STAT_AREA])
        if area_cc >= 12 and max(w_cc, h_cc) >= 6:
            clean_bin[lbl_im == k_i] = 1.0

    if float(clean_bin.max()) < 0.5:
        return None
    snr_gate = cv2.GaussianBlur(clean_bin, (0, 0), sigmaX=0.9, sigmaY=0.9) * np.clip((lum_d - 2.0 * sig) / (1.8 * sig), 0.0, 1.0)
    excess = np.maximum(diff, 0.0) * (mask_p * snr_gate)[:, :, None]
    if float(excess.max()) < 3.0 * sig:
        return None
    return y0, y1, x0, x1, excess.astype(np.float32), bgr_ev


def _build_leo_spacemobile_plate(
    cfg: dict,
    poly_cx: np.ndarray,
    poly_cy: np.ndarray,
    archive_root: str,
    srgb_lut: np.ndarray,
    mx_3552: np.ndarray,
    my_3552: np.ndarray,
    w_dome: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Builds `orb_leo_spacemobile_pass`: Full-Arc SpaceMobile BlueBird (NORAD #69591) 5-frame 1,846-px zenith
    overpass & simultaneous pre-dawn LEO constellation/rocket-body passes on 2026-10-21 (05:48-06:11).
    """
    if not os.path.exists(DB_PATH):
        return None

    tid = "orb_leo_spacemobile_pass"
    lookup = _ArchiveFrameLookup(archive_root)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, frame_filename, subject, snr, length_px, alt_deg, az_deg, features_json "
        "FROM sky_events WHERE is_hidden=0 AND mode='NIGHT' AND category='space_object' "
        "AND subject LIKE '%NORAD%' "
        "AND frame_filename >= 'image-20261021054800.jpg' AND frame_filename <= 'image-20261021061230.jpg' "
        "ORDER BY frame_filename ASC"
    ).fetchall()
    if len(rows) < 4:
        # Fallback to top NORAD space_object streaks across whole DB if Oct 21 window isn't present
        rows = conn.execute(
            "SELECT id, frame_filename, subject, snr, length_px, alt_deg, az_deg, features_json "
            "FROM sky_events WHERE is_hidden=0 AND mode='NIGHT' AND category='space_object' "
            "AND subject LIKE '%NORAD%' ORDER BY snr DESC LIMIT 18"
        ).fetchall()
    conn.close()
    if not rows:
        return None

    # Build deep starry sky dome from 16 clear dark frames right in the 2026-10-21 05:35-06:04 window
    oct21_files = sorted(glob.glob(os.path.join(lookup.year_root, "10", "21", "image_05[345]*.jpg")))
    if len(oct21_files) < 10:
        oct21_files = sorted(glob.glob(os.path.join(lookup.year_root, "10", "*", "image_04*.jpg")))[:18]
    base_pick = [oct21_files[i] for i in np.linspace(0, len(oct21_files) - 1, min(16, len(oct21_files)), dtype=int)] if oct21_files else []

    base_dome, single_dome = _build_deep_allsky_starry_base(base_pick, mx_3552, my_3552, w_dome, srgb_lut)
    streak_canvas_3552 = np.zeros((3552, 3552, 3), dtype=np.float32)
    single_hero_bgr: Optional[np.ndarray] = None

    sat_tracks: Dict[str, Dict[str, Any]] = {}
    segments_composited = 0

    for eid, f_name, subj, snr_v, len_px, alt_d, az_d, fj_str in rows:
        fj = json.loads(fj_str) if (fj_str and fj_str.strip()) else {}
        res_f = lookup.resolve(f_name)
        if res_f is None:
            continue
        ev_fp, nb_fps = res_f
        ext = _extract_transient_excess_3552(ev_fp, nb_fps, fj, srgb_lut, pad_px=40.0, max_center_med=72.0)
        if ext is None:
            continue
        y0, y1, x0, x1, excess_p, ev_bgr = ext
        # Boost slightly so slender high-orbit SL-16 R/B & Gaofen tracks gleam cleanly beside SpaceMobile-010
        boost = 1.25 if float(snr_v or 0) >= 85.0 else 1.75
        streak_canvas_3552[y0:y1, x0:x1] = np.maximum(streak_canvas_3552[y0:y1, x0:x1], excess_p * boost)
        segments_composited += 1

        if "SPACEMOBILE-010" in (subj or "") and "060941" in (f_name or ""):
            single_hero_bgr = ev_bgr

        # Parse clean spacecraft name + NORAD + orbit altitude
        m_sat = re.search(r"([A-Z0-9\-\s/]+?)\s*\[NORAD\s*#(\d+)\]\s*\(([^)]+)\)", subj or "")
        if m_sat:
            s_name = m_sat.group(1).strip()
            norad = m_sat.group(2).strip()
            orb_info = m_sat.group(3).split("/")[0].strip()
            key_s = f"{s_name} #{norad}"
        else:
            s_name = re.sub(r"[^\x20-\x7E]+", "", subj or "Satellite").strip()[:28]
            norad = ""
            orb_info = "LEO"
            key_s = s_name

        dx_out, dy_out = _az_alt_to_dome_xy(float(az_d or 180.0), float(alt_d or 45.0))
        tr = sat_tracks.setdefault(key_s, {
            "name": s_name,
            "norad": norad,
            "orbit": orb_info,
            "max_snr": 0.0,
            "segments": 0,
            "total_px": 0.0,
            "pts": [],
        })
        tr["max_snr"] = max(tr["max_snr"], float(snr_v or 0.0))
        tr["segments"] += 1
        tr["total_px"] += float(len_px or 0.0)
        tr["pts"].append([round(dx_out, 1), round(dy_out, 1)])

    if single_hero_bgr is not None:
        hero_lin = _suppress_sensor_hot_pixels_rgb(srgb_lut[single_hero_bgr[:, :, ::-1]])
        single_warped = cv2.remap(hero_lin, mx_3552, my_3552, interpolation=cv2.INTER_CUBIC)
        single_dome = single_warped * (w_dome > 0.03)[:, :, None] * (w_dome[:, :, None] ** 0.4)

    streak_broad = _nyquist_broaden_streaks_3552(streak_canvas_3552, gain=4.6)
    warped_streaks = cv2.remap(streak_broad, mx_3552, my_3552, interpolation=cv2.INTER_LINEAR) * (w_dome > 0.04)[:, :, None]
    stack_rgb = (base_dome + warped_streaks).astype(np.float32)

    tracks_list = sorted(sat_tracks.values(), key=lambda d: (d["segments"] * 40.0 + d["max_snr"]), reverse=True)
    meta = {
        "kind": "allsky_leo_overpass",
        "window": "2026-10-21 05:48 - 06:11 PDT",
        "base_frames": len(base_pick),
        "segments_composited": segments_composited,
        "tracks": tracks_list[:8],
    }
    np.savez_compressed(
        os.path.join(LINEAR_DIR, f"{tid}_linear.npz"),
        stack_rgb=stack_rgb,
        single_rgb=single_dome.astype(np.float32),
        meta_json=np.array(json.dumps(meta)),
    )

    tot_frames = len(base_pick) + segments_composited
    exp_min = round(tot_frames * 0.5, 1)
    return {
        "id": tid,
        "name": "SpaceMobile BlueBird (NORAD #69591) & Pre-Dawn LEO Overpass Dome",
        "catalog_code": f"NORAD #69591 / #22220 / #62825 ({len(tracks_list)} Spacecraft)",
        "type": "orbital",
        "type_label": "Multi-Satellite LEO Overpass • TLE Verified",
        "constellation": "360° All-Sky Celestial Dome",
        "distance": "468 – 842 km (LEO Altitude)",
        "apparent_mag": "0.8 – 4.2",
        "ra_deg": 0.0,
        "dec_deg": 45.0000,
        "ra_hms": "North-Up 360° Dome",
        "dec_dms": "Zenith +48° 08' N",
        "fov_str": "147° All-Sky Dome (N-Up / E-Left)",
        "pixel_scale_arcmin": 10.6,
        "resolution": f"{ORB_W} × {ORB_H} (Multi-Exposure Track Composite)",
        "frames_stacked": tot_frames,
        "integration_minutes": exp_min,
        "integration_hours": round(exp_min / 60.0, 2),
        "snr_gain_x": round(math.sqrt(max(len(base_pick), 1)), 2),
        "limiting_mag_gain": "127.3σ Peak Track",
        "max_altitude_deg": 78.5,
        "median_registration_corr": 1.0,
        "description": (
            f"Captured during a crystal-clear 22-minute pre-dawn window on Oct 21, 2026: AST SpaceMobile's "
            f"giant 64 m² BlueBird phased-array spacecraft (SPACEMOBILE-010, NORAD #69591 at 532 km) carves an "
            f"unbroken 5-frame, 1,846-pixel horizon-to-horizon arc across zenith alongside Soviet/Ukrainian "
            f"SL-16 Zenit-2 rocket bodies (839 km), Gaofen-5, and Starlink passes."
        ),
        "stack_url": f"/astrophoto_data/{tid}_stack.jpg",
        "annotated_url": f"/astrophoto_data/{tid}_annotated.jpg",
        "single_url": f"/astrophoto_data/{tid}_single.jpg",
    }


def _build_orionid_fireball_storm_plate(
    cfg: dict,
    poly_cx: np.ndarray,
    poly_cy: np.ndarray,
    archive_root: str,
    srgb_lut: np.ndarray,
    mx_3552: np.ndarray,
    my_3552: np.ndarray,
    w_dome: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Builds `orb_orionid_fireball_storm`: Multi-Night October Orionid Fireball, Bolide & Shooting-Star
    All-Sky Celestial Dome Composite extracted from untouched 3552x3552 archive exposures!
    """
    if not os.path.exists(DB_PATH):
        return None

    tid = "orb_orionid_fireball_storm"
    lookup = _ArchiveFrameLookup(archive_root)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, frame_filename, category, subject, snr, length_px, alt_deg, az_deg, features_json "
        "FROM sky_events WHERE is_hidden=0 AND mode='NIGHT' "
        "AND category IN ('fireball', 'meteor') "
        "AND (user_corrected_category IS NULL OR user_corrected_category = '' OR user_corrected_category IN ('fireball', 'meteor')) "
        "AND features_json IS NOT NULL AND features_json != '' AND features_json != '{}' "
        "ORDER BY interest_score DESC, snr DESC LIMIT 95"
    ).fetchall()
    conn.close()
    if not rows:
        return None

    # Deep starry sky base from 18 clear moonless midnight-to-03:30 exposures (Orion & Milky Way overhead!)
    oct_clear = sorted(glob.glob(os.path.join(lookup.year_root, "10", "21", "image_03[1234]*.jpg")))
    if len(oct_clear) < 12:
        oct_clear = sorted(glob.glob(os.path.join(lookup.year_root, "10", "20", "image_03*.jpg")))
    base_pick = [oct_clear[i] for i in np.linspace(0, len(oct_clear) - 1, min(18, len(oct_clear)), dtype=int)] if oct_clear else []
    base_dome, single_dome = _build_deep_allsky_starry_base(base_pick, mx_3552, my_3552, w_dome, srgb_lut)

    meteor_canvas_3552 = np.zeros((3552, 3552, 3), dtype=np.float32)
    single_hero_bgr: Optional[np.ndarray] = None
    events_meta: List[Dict[str, Any]] = []

    for eid, f_name, cat, subj, snr_v, len_px, alt_d, az_d, fj_str in rows:
        if float(alt_d or 0.0) < 23.5:
            continue
        fj = json.loads(fj_str)
        # Skip multi-label rows that were actually satellite passes
        if "NORAD" in (subj or ""):
            continue
        res_f = lookup.resolve(f_name)
        if res_f is None:
            continue
        ev_fp, nb_fps = res_f
        ext = _extract_transient_excess_3552(ev_fp, nb_fps, fj, srgb_lut, pad_px=44.0, max_center_med=82.0)
        if ext is None:
            continue
        y0, y1, x0, x1, excess_p, ev_bgr = ext

        # Ensure meteor lies inside clean tree-free dome before accepting
        dx_out, dy_out = _az_alt_to_dome_xy(float(az_d or 180.0), float(alt_d or 45.0))
        ix_o, iy_o = int(np.clip(round(dx_out), 0, ORB_W - 1)), int(np.clip(round(dy_out), 0, ORB_H - 1))
        if float(w_dome[iy_o, ix_o]) < 0.25:
            continue

        boost = 1.65 if cat == "fireball" else 2.15
        meteor_canvas_3552[y0:y1, x0:x1] = np.maximum(meteor_canvas_3552[y0:y1, x0:x1], excess_p * boost)

        if single_hero_bgr is None and cat == "fireball" and float(len_px or 0) >= 160.0:
            single_hero_bgr = ev_bgr

        m_dt = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", f_name or "")
        ts_lbl = f"Oct {m_dt.group(3)} {m_dt.group(4)}:{m_dt.group(5)}" if m_dt else f_name
        events_meta.append({
            "id": int(eid),
            "category": cat,
            "snr": round(float(snr_v or 0.0), 1),
            "length_px": int(round(float(len_px or 0.0))),
            "alt": round(float(alt_d or 0.0), 1),
            "az": round(float(az_d or 0.0), 1),
            "x": round(dx_out, 1),
            "y": round(dy_out, 1),
            "time_lbl": ts_lbl,
        })
        if len(events_meta) >= 22:
            break

    if single_hero_bgr is not None:
        hero_lin = _suppress_sensor_hot_pixels_rgb(srgb_lut[single_hero_bgr[:, :, ::-1]])
        single_warped = cv2.remap(hero_lin, mx_3552, my_3552, interpolation=cv2.INTER_CUBIC)
        single_dome = single_warped * (w_dome > 0.03)[:, :, None] * (w_dome[:, :, None] ** 0.4)

    meteor_broad = _nyquist_broaden_streaks_3552(meteor_canvas_3552, gain=4.4)
    warped_meteors = cv2.remap(meteor_broad, mx_3552, my_3552, interpolation=cv2.INTER_LINEAR) * (w_dome > 0.04)[:, :, None]
    stack_rgb = (base_dome + warped_meteors).astype(np.float32)

    n_fb = sum(1 for e in events_meta if e["category"] == "fireball")
    n_mt = len(events_meta) - n_fb
    meta = {
        "kind": "allsky_fireball_storm",
        "base_frames": len(base_pick),
        "total_events": len(events_meta),
        "fireballs": n_fb,
        "meteors": n_mt,
        "events": events_meta,
    }
    np.savez_compressed(
        os.path.join(LINEAR_DIR, f"{tid}_linear.npz"),
        stack_rgb=stack_rgb,
        single_rgb=single_dome.astype(np.float32),
        meta_json=np.array(json.dumps(meta)),
    )

    tot_frames = len(base_pick) + len(events_meta)
    exp_min = round(tot_frames * 0.5, 1)
    return {
        "id": tid,
        "name": f"Multi-Night October Orionid Fireball & Bolide All-Sky Composite ({len(events_meta)} Meteors)",
        "catalog_code": f"{n_fb} Bolide Fireballs + {n_mt} Bright Meteors",
        "type": "orbital",
        "type_label": "All-Sky Meteor & Bolide Storm Composite",
        "constellation": "Orionid / Taurid All-Sky Dome",
        "distance": "85 – 115 km (Mesosphere Ablation)",
        "apparent_mag": "-3.5 to +1.5",
        "ra_deg": 95.0,
        "dec_deg": 16.0,
        "ra_hms": "North-Up 360° Dome",
        "dec_dms": "Mesosphere 95 km Alt",
        "fov_str": "147° All-Sky Dome (N-Up / E-Left)",
        "pixel_scale_arcmin": 10.6,
        "resolution": f"{ORB_W} × {ORB_H} (3552×3552 Raw Bolide Isolation)",
        "frames_stacked": tot_frames,
        "integration_minutes": exp_min,
        "integration_hours": round(exp_min / 60.0, 2),
        "snr_gain_x": round(math.sqrt(max(len(base_pick), 1)), 2),
        "limiting_mag_gain": "138.0σ Peak Bolide",
        "max_altitude_deg": 82.2,
        "median_registration_corr": 1.0,
        "description": (
            f"Compositing {n_fb} explosive bolide fireballs (up to 138σ and 372 pixels long!) and {n_mt} bright "
            f"October Orionid/Taurid meteors directly from untouched 3552×3552 raw archive exposures onto a "
            f"deep-stacked, airplane-free starry Milky Way dome."
        ),
        "stack_url": f"/astrophoto_data/{tid}_stack.jpg",
        "annotated_url": f"/astrophoto_data/{tid}_annotated.jpg",
        "single_url": f"/astrophoto_data/{tid}_single.jpg",
    }


def _draw_allsky_dome_compass_and_rings(vis: np.ndarray) -> None:
    """Draws subtle observatory horizon rim, 30/60 deg altitude circles, and N/E/S/W cardinal markers."""
    cx, cy, r = int(round(DOME_CX)), int(round(DOME_CY)), int(round(DOME_R))
    cv2.circle(vis, (cx, cy), r, (140, 120, 65), 1, cv2.LINE_AA)
    for alt_ring in (30.0, 60.0):
        rr = int(round(((90.0 - alt_ring) / (90.0 - DOME_MIN_ALT)) * DOME_R))
        cv2.circle(vis, (cx, cy), rr, (90, 80, 52), 1, cv2.LINE_AA)
        cv2.putText(vis, f"{int(alt_ring)}d", (cx + 5, cy - rr + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 165, 185), 1, cv2.LINE_AA)

    for lbl, az_d, dx_t, dy_t in (
        ("N", 0.0, -5, -10),
        ("E", 90.0, -20, 5),
        ("S", 180.0, -5, 20),
        ("W", 270.0, 10, 5),
    ):
        px = int(round(DOME_CX - (DOME_R + 4) * math.sin(math.radians(az_d))))
        py = int(round(DOME_CY - (DOME_R + 4) * math.cos(math.radians(az_d))))
        cv2.putText(vis, lbl, (px + dx_t, py + dy_t), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 215, 90), 2, cv2.LINE_AA)


def _annotate_orbital_plate(bgr_img: np.ndarray, item: Dict[str, Any], meta: Dict[str, Any]) -> np.ndarray:
    """Renders crisp spacecraft/Clarke-Belt/fireball/circumpolar telemetry overlays onto 1200x900 Orbital plates."""
    vis = bgr_img.copy()
    h, w = vis.shape[:2]
    kind = meta.get("kind", "")

    if kind == "geo_clarke_belt":
        belt_pts = meta.get("belt_pts", [])
        if len(belt_pts) >= 4:
            for k in range(0, len(belt_pts) - 1, 2):
                p0 = (int(round(belt_pts[k][0])), int(round(belt_pts[k][1])))
                p1 = (int(round(belt_pts[k + 1][0])), int(round(belt_pts[k + 1][1])))
                cv2.line(vis, p0, p1, (220, 190, 65), 1, cv2.LINE_AA)
            lp = belt_pts[len(belt_pts) // 5]
            by_lbl = min(h - 46, int(lp[1]) - 54)
            bx_lbl = max(16, int(lp[0]) - 20)
            b_txt = "Clarke Geostationary Belt (35,786 km / Dec -7.15 deg)"
            cv2.putText(vis, b_txt, (bx_lbl + 1, by_lbl + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, b_txt, (bx_lbl, by_lbl), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (235, 215, 85), 1, cv2.LINE_AA)

        for idx, g in enumerate(meta.get("geos", [])):
            ix, iy = int(round(g["x"])), int(round(g["y"]))
            col = (145, 250, 175)
            cv2.circle(vis, (ix, iy), 11, col, 1, cv2.LINE_AA)
            sgn = -1 if (idx % 2 == 0) else 1
            ty = int(np.clip(iy + sgn * (28 + 14 * (idx % 3)), 28, h - 48))
            cv2.line(vis, (ix, iy + sgn * 11), (ix, ty + (-4 if sgn > 0 else 4)), col, 1, cv2.LINE_AA)
            short_nm = g["name"][:20]
            lbl = f"{short_nm} (#{g['norad']})"
            tx = int(np.clip(ix - 48, 10, w - 210))
            cv2.putText(vis, lbl, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.41, col, 1, cv2.LINE_AA)

        for az_txt, ax_p in (("SE (Az 156d)", 14), ("DUE SOUTH (Az 180d)", w // 2 - 76), ("SW (Az 204d)", w - 115)):
            cv2.putText(vis, az_txt, (ax_p + 1, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, az_txt, (ax_p, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (225, 215, 95), 1, cv2.LINE_AA)

    elif kind == "polaris_vortex":
        ncp = meta.get("ncp_xy", [600.0, 636.0])
        nx_i, ny_i = int(round(ncp[0])), int(round(ncp[1]))
        cv2.line(vis, (nx_i - 16, ny_i), (nx_i - 5, ny_i), (240, 215, 80), 1, cv2.LINE_AA)
        cv2.line(vis, (nx_i + 5, ny_i), (nx_i + 16, ny_i), (240, 215, 80), 1, cv2.LINE_AA)
        cv2.line(vis, (nx_i, ny_i - 16), (nx_i, ny_i - 5), (240, 215, 80), 1, cv2.LINE_AA)
        cv2.line(vis, (nx_i, ny_i + 5), (nx_i, ny_i + 16), (240, 215, 80), 1, cv2.LINE_AA)
        cv2.circle(vis, (nx_i, ny_i), 22, (240, 215, 80), 1, cv2.LINE_AA)
        cv2.putText(vis, "North Celestial Pole (Alt 48.1d)", (nx_i + 26, ny_i + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, "North Celestial Pole (Alt 48.1d)", (nx_i + 25, ny_i + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 215, 80), 1, cv2.LINE_AA)

        for st in meta.get("stars", []):
            sx, sy = int(round(st["x"])), int(round(st["y"]))
            if math.hypot(sx - nx_i, sy - ny_i) < 28.0 and "Polaris" not in st["label"]:
                continue
            col = (165, 230, 255)
            cv2.circle(vis, (sx, sy), 6, col, 1, cv2.LINE_AA)
            tx = int(np.clip(sx + 10, 12, w - 195))
            ty = int(np.clip(sy - 8 if "Polaris" not in st["label"] else sy - 22, 24, h - 46))
            cv2.putText(vis, st["label"], (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, st["label"], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.43, col, 1, cv2.LINE_AA)

    elif kind == "allsky_leo_overpass":
        _draw_allsky_dome_compass_and_rings(vis)
        placed_ys: List[Tuple[int, int]] = []
        for idx, tr in enumerate(meta.get("tracks", [])):
            pts = tr.get("pts", [])
            if not pts:
                continue
            mid_p = pts[len(pts) // 2]
            ix, iy = int(round(mid_p[0])), int(round(mid_p[1]))
            is_hero = "SPACEMOBILE" in tr["name"].upper()
            col = (140, 252, 180) if is_hero else (240, 210, 85)
            cv2.circle(vis, (ix, iy), 11 if is_hero else 8, col, 1, cv2.LINE_AA)

            # Place clean telemetry callout in left or right margin outside the 360-deg dome!
            on_left = ix < DOME_CX
            tx = 16 if on_left else (w - 252)
            ty = int(np.clip(iy, 52, h - 62))
            for ox, oy in placed_ys:
                if abs(tx - ox) < 100 and abs(ty - oy) < 36:
                    ty = min(h - 54, oy + 38)
            placed_ys.append((tx, ty))
            anchor_x = (tx + 225) if on_left else (tx - 6)
            cv2.line(vis, (ix, iy), (anchor_x, ty - 4), col, 1, cv2.LINE_AA)
            l1 = f"{tr['name']} #{tr['norad']}" if tr.get("norad") else tr["name"]
            l2 = f"{tr['orbit']} • {tr['max_snr']:.0f} sigma ({tr['segments']} frames)"
            cv2.putText(vis, l1, (tx + 1, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, l1, (tx, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
            cv2.putText(vis, l2, (tx, ty + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (185, 205, 225), 1, cv2.LINE_AA)

    elif kind == "allsky_fireball_storm":
        _draw_allsky_dome_compass_and_rings(vis)
        # Annotate top 9 highest-SNR fireballs & bolides cleanly in side margins
        top_ev = sorted(meta.get("events", []), key=lambda e: (e["category"] == "fireball", e["snr"]), reverse=True)[:9]
        left_y, right_y = 66, 66
        for ev in sorted(top_ev, key=lambda e: e["y"]):
            ix, iy = int(round(ev["x"])), int(round(ev["y"]))
            is_fb = ev["category"] == "fireball"
            col = (120, 195, 255) if is_fb else (235, 215, 95)  # Warm amber-gold / cyan in BGR
            cv2.circle(vis, (ix, iy), 11 if is_fb else 8, col, 1, cv2.LINE_AA)
            if ix < DOME_CX:
                tx, ty = 16, max(left_y, min(h - 60, iy))
                left_y = ty + 46
                ax = tx + 212
            else:
                tx, ty = w - 242, max(right_y, min(h - 60, iy))
                right_y = ty + 46
                ax = tx - 6
            cv2.line(vis, (ix, iy), (ax, ty - 4), col, 1, cv2.LINE_AA)
            tag = "BOLIDE FIREBALL" if is_fb else "ORIONID METEOR"
            l1 = f"{tag} ({ev['snr']:.0f} sigma, {ev['length_px']}px)"
            l2 = f"{ev['time_lbl']} • Alt {ev['alt']:.0f}d Az {ev['az']:.0f}d"
            cv2.putText(vis, l1, (tx + 1, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, l1, (tx, ty - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
            cv2.putText(vis, l2, (tx, ty + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 200, 222), 1, cv2.LINE_AA)

    bar = vis[h - 34 : h, :].copy()
    vis[h - 34 : h, :] = (bar.astype(np.float32) * 0.40).astype(np.uint8)
    caption = f"{item['name']}  |  {item['catalog_code']}  |  {item['frames_stacked']}x30s  |  BluffSky Observatory"
    cv2.putText(vis, caption, (14, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 238, 250), 1, cv2.LINE_AA)
    return vis


def render_orbital_item(item: Dict[str, Any]) -> None:
    """Re-renders `*_stack.jpg`, `*_annotated.jpg`, and `*_single.jpg` for an Orbital/Composite plate."""
    import deep_sky_astrophoto as dsa

    tid = item["id"]
    npz_p = os.path.join(LINEAR_DIR, f"{tid}_linear.npz")
    if not os.path.exists(npz_p):
        return
    d = np.load(npz_p, allow_pickle=False)
    st_raw = d["stack_rgb"].astype(np.float32)
    sg_raw = d["single_rgb"].astype(np.float32)
    meta = json.loads(str(d["meta_json"]))
    kind = meta.get("kind", "")

    h, w = st_raw.shape[:2]
    if kind in ("allsky_leo_overpass", "allsky_fireball_storm"):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r_out = np.hypot(xx - DOME_CX, yy - DOME_CY)
        bg_mask = (r_out <= DOME_R - 36.0) & np.any(np.abs(st_raw) > 1e-6, axis=-1)
        alpha = 40.0
        sat = 1.38
    else:
        bg_mask = np.ones((h, w), dtype=bool)
        alpha = 38.0 if kind == "polaris_vortex" else 42.0
        sat = 1.44 if kind == "polaris_vortex" else 1.36

    st_bgr = dsa.stretch_astrophoto_rgb(st_raw, bg_mask=bg_mask, alpha=alpha, sat_boost=sat, sky_floor=0.040)
    sg_bgr = dsa.stretch_astrophoto_rgb(sg_raw, bg_mask=bg_mask, alpha=alpha, sat_boost=1.22, sky_floor=0.040, is_single_frame=True)

    # Keep circular All-Sky Dome exterior bezel crisp deep observatory midnight (#050811)
    if kind in ("allsky_leo_overpass", "allsky_fireball_storm"):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r_out = np.hypot(xx - DOME_CX, yy - DOME_CY)
        bezel_w = np.clip((r_out - (DOME_R - 4.0)) / 8.0, 0.0, 1.0)[:, :, None]
        bezel_col = np.array([[[17.0, 8.0, 5.0]]], dtype=np.float32)  # BGR for #050811
        st_bgr = np.clip(st_bgr.astype(np.float32) * (1.0 - bezel_w) + bezel_col * bezel_w, 0, 255).astype(np.uint8)
        sg_bgr = np.clip(sg_bgr.astype(np.float32) * (1.0 - bezel_w) + bezel_col * bezel_w, 0, 255).astype(np.uint8)

    ann_bgr = _annotate_orbital_plate(st_bgr, item, meta)

    cv2.imwrite(os.path.join(ASTRO_DIR, f"{tid}_stack.jpg"), st_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(os.path.join(ASTRO_DIR, f"{tid}_annotated.jpg"), ann_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    cv2.imwrite(os.path.join(ASTRO_DIR, f"{tid}_single.jpg"), sg_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def build_orbital_composites(
    cfg: dict,
    poly_cx: np.ndarray,
    poly_cy: np.ndarray,
    archive_root: str,
    existing_objects_by_id: Dict[str, Dict[str, Any]],
    incremental: bool = False,
    progress_cb: Optional[Callable[[str, int], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Builds (or reuses in incremental mode) all 4 Earth-Orbital & All-Sky Composite Showcases.
    """
    srgb_lut = np.power(np.arange(256, dtype=np.float32) / 255.0, 2.2).astype(np.float32)
    orbital_ids = [
        "orb_geo_clarke_belt",
        "orb_leo_spacemobile_pass",
        "orb_orionid_fireball_storm",
        "orb_polaris_startrail_vortex",
    ]
    if incremental and all(
        os.path.exists(os.path.join(LINEAR_DIR, f"{oid}_linear.npz")) and oid in existing_objects_by_id
        for oid in orbital_ids
    ):
        return [existing_objects_by_id[oid] for oid in orbital_ids]

    out_items: List[Dict[str, Any]] = []
    need_seq = (not incremental) or (
        not os.path.exists(os.path.join(LINEAR_DIR, "orb_geo_clarke_belt_linear.npz"))
        or not os.path.exists(os.path.join(LINEAR_DIR, "orb_polaris_startrail_vortex_linear.npz"))
    )
    seq: List[Tuple[str, float]] = []
    if need_seq:
        if progress_cb:
            progress_cb("Locating continuous 80-min clear moonless window for Clarke Belt & Polaris Vortex...", 91)
        seq = _find_best_continuous_clear_run(cfg, archive_root, target_len=156)

    # 1. Geostationary Clarke Belt Satellite Necklace
    if incremental and os.path.exists(os.path.join(LINEAR_DIR, "orb_geo_clarke_belt_linear.npz")) and "orb_geo_clarke_belt" in existing_objects_by_id:
        out_items.append(existing_objects_by_id["orb_geo_clarke_belt"])
    elif len(seq) >= 24:
        if progress_cb:
            progress_cb("Building Geostationary Clarke Belt Satellite Necklace (35,786 km)...", 92)
        out_items.append(_build_geo_clarke_belt_plate(seq, cfg, poly_cx, poly_cy, srgb_lut))

    # Precompute 360-deg All-Sky Dome projection for LEO SpaceMobile Overpass & Orionid Fireball Storm
    mx_3552, my_3552, w_dome, _ = _build_allsky_dome_projection(cfg, poly_cx, poly_cy)

    # 2. Full-Arc SpaceMobile BlueBird (NORAD #69591) & Pre-Dawn LEO Overpass Dome
    if incremental and os.path.exists(os.path.join(LINEAR_DIR, "orb_leo_spacemobile_pass_linear.npz")) and "orb_leo_spacemobile_pass" in existing_objects_by_id:
        out_items.append(existing_objects_by_id["orb_leo_spacemobile_pass"])
    else:
        if progress_cb:
            progress_cb("Compositing SpaceMobile BlueBird (NORAD #69591) & LEO Overpass Dome...", 94)
        leo_item = _build_leo_spacemobile_plate(cfg, poly_cx, poly_cy, archive_root, srgb_lut, mx_3552, my_3552, w_dome)
        if leo_item:
            out_items.append(leo_item)

    # 3. Multi-Night October Orionid Fireball & Bolide All-Sky Dome
    if incremental and os.path.exists(os.path.join(LINEAR_DIR, "orb_orionid_fireball_storm_linear.npz")) and "orb_orionid_fireball_storm" in existing_objects_by_id:
        out_items.append(existing_objects_by_id["orb_orionid_fireball_storm"])
    else:
        if progress_cb:
            progress_cb("Compositing Multi-Night Orionid Fireball & Bolide All-Sky Storm...", 95)
        fb_item = _build_orionid_fireball_storm_plate(cfg, poly_cx, poly_cy, archive_root, srgb_lut, mx_3552, my_3552, w_dome)
        if fb_item:
            out_items.append(fb_item)

    # 4. True-Color Circumpolar Polaris Star-Trail Vortex (100% Airplane-Free)
    if incremental and os.path.exists(os.path.join(LINEAR_DIR, "orb_polaris_startrail_vortex_linear.npz")) and "orb_polaris_startrail_vortex" in existing_objects_by_id:
        out_items.append(existing_objects_by_id["orb_polaris_startrail_vortex"])
    elif len(seq) >= 24:
        if progress_cb:
            progress_cb("Building True-Color Circumpolar Polaris Star-Trail Vortex...", 96)
        out_items.append(_build_polaris_startrail_vortex_plate(seq, cfg, poly_cx, poly_cy, srgb_lut))

    return out_items

