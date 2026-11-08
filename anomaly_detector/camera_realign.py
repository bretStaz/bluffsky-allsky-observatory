import datetime
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import ephem
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
REALIGN_STATE_FILE = os.path.join(BASE_DIR, "realignment_state.json")
REF_DAY_FILE = os.path.join(BASE_DIR, "reference_day.jpg")
MASK_FILE = os.path.join(BASE_DIR, "sky_mask.png")
MASK_PREVIEW_FILE = os.path.join(BASE_DIR, "sky_mask_preview.jpg")

logger = logging.getLogger("AllskyDetector")

# Bright navigation stars (J2000 RA hours, Dec deg, V mag) for blind 360-deg plate solving
NAVIGATION_STARS = [
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
    ("Menkalinan", 5.9921, 44.9474, 1.90),
    ("Almach", 2.0650, 42.3297, 2.10),
    ("Hamal", 2.1196, 23.4624, 2.00),
    ("Alpheratz", 0.1398, 29.0904, 2.06),
    ("Mirach", 1.1622, 35.6206, 2.07),
    ("Algol", 3.1361, 40.9556, 2.12),
    ("Schedar", 0.6751, 56.5373, 2.24),
    ("Caph", 0.1529, 59.1498, 2.28),
    ("Gamma Cas", 0.9451, 60.7167, 2.15),
    ("Sadr", 20.3705, 40.2567, 2.23),
    ("Enif", 21.7364, 9.8750, 2.39),
    ("Scheat", 23.0629, 28.0828, 2.42),
    ("Markab", 23.0793, 15.2053, 2.49),
    ("Algenib", 0.2206, 15.1836, 2.84),
    ("Ruchbah", 1.4303, 60.2353, 2.68),
]


def load_realign_state() -> Dict[str, Any]:
    default_state = {
        "armed": False,
        "status": "idle",
        "armed_at_iso": None,
        "armed_night_tag": None,
        "day_mask_updated": False,
        "star_alignment_solved": False,
        "candidate_solutions": [],
        "frames_evaluated": 0,
        "last_check_iso": None,
        "last_check_msg": "Alignment locked. Click 'Camera Moved — Start Realignment' only after physically moving or rotating the camera.",
        "last_solved_result": None,
    }
    if os.path.exists(REALIGN_STATE_FILE):
        try:
            with open(REALIGN_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            default_state.update(data)
        except Exception:
            pass
    return default_state


def save_realign_state(state: Dict[str, Any]) -> None:
    try:
        with open(REALIGN_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed saving realignment_state.json: {e}")


def arm_camera_realignment(update_day_mask: bool = True) -> Dict[str, Any]:
    """
    Triggered manually by the user via the web portal when they have physically moved
    or rotated the camera. Arms the background watcher to wait for clear sky data.
    """
    now_dt = datetime.datetime.now()
    night_tag = (now_dt - datetime.timedelta(hours=12)).strftime("%Y%m%d")
    state = load_realign_state()
    state.update(
        {
            "armed": True,
            "status": "armed_waiting",
            "armed_at_iso": now_dt.isoformat(timespec="seconds"),
            "armed_night_tag": night_tag,
            "day_mask_updated": not update_day_mask,
            "star_alignment_solved": False,
            "candidate_solutions": [],
            "frames_evaluated": 0,
            "last_check_iso": now_dt.isoformat(timespec="seconds"),
            "last_check_msg": (
                "Realignment armed! Watching incoming camera frames for a clear starry night "
                "(requires >= 55 sharp stars & >= 10 catalog matches across 2 frames)..."
            ),
        }
    )
    save_realign_state(state)
    logger.info("User armed Camera Realignment mode — waiting for clear starry night frames.")
    return state


def cancel_camera_realignment() -> Dict[str, Any]:
    state = load_realign_state()
    state.update(
        {
            "armed": False,
            "status": "idle",
            "candidate_solutions": [],
            "last_check_iso": datetime.datetime.now().isoformat(timespec="seconds"),
            "last_check_msg": "Realignment cancelled by user. Previous camera calibration remains locked.",
        }
    )
    save_realign_state(state)
    logger.info("User cancelled Camera Realignment mode.")
    return state


def _project_vecs_to_pixels(
    east: np.ndarray,
    north: np.ndarray,
    up: np.ndarray,
    bottom_az_deg: float,
    tilt_x_deg: float,
    tilt_y_deg: float,
    cx: float,
    cy: float,
    r_horiz: float,
    k2: float,
    k3: float,
) -> Tuple[np.ndarray, np.ndarray]:
    phi = math.radians(bottom_az_deg)
    cb, sb = math.cos(phi), math.sin(phi)
    vx0 = cb * east - sb * north
    vy0 = sb * east + cb * north
    vz0 = up

    tx = math.radians(tilt_x_deg)
    ctx, stx = math.cos(tx), math.sin(tx)
    vx1 = vx0
    vy1 = ctx * vy0 - stx * vz0
    vz1 = stx * vy0 + ctx * vz0

    ty = math.radians(tilt_y_deg)
    cty, sty = math.cos(ty), math.sin(ty)
    vx_c = cty * vx1 + sty * vz1
    vy_c = vy1
    vz_c = -sty * vx1 + cty * vz1

    theta = np.arccos(np.clip(vz_c, -1.0, 1.0))
    u = theta / (math.pi / 2.0)
    r = r_horiz * (u + k2 * (u ** 3) + k3 * (u ** 5))
    rho = np.hypot(vx_c, vy_c)
    px = cx + r * (vx_c / np.maximum(rho, 1e-9))
    py = cy + r * (vy_c / np.maximum(rho, 1e-9))
    return px, py


def _compute_visible_catalog(
    dt_utc: datetime.datetime,
    cfg: dict,
    min_alt_deg: float = 24.0,
) -> Tuple[List[Dict[str, Any]], float]:
    obs = ephem.Observer()
    obs.lat = str(cfg.get("camera_lat", 45.0000))
    obs.lon = str(cfg.get("camera_lon", -100.0000))
    obs.elevation = float(cfg.get("camera_alt_ft", 130.0)) * 0.3048
    obs.date = dt_utc.strftime("%Y/%m/%d %H:%M:%S")

    sun = ephem.Sun()
    sun.compute(obs)
    sun_alt_deg = math.degrees(float(sun.alt))

    moon = ephem.Moon()
    moon.compute(obs)
    moon_alt_deg = math.degrees(float(moon.alt))
    moon_az_deg = math.degrees(float(moon.az))

    catalog: List[Dict[str, Any]] = []

    def _near_moon(alt_d: float, az_d: float) -> bool:
        if moon_alt_deg < -2.0:
            return False
        a1, z1 = math.radians(alt_d), math.radians(az_d)
        a2, z2 = math.radians(moon_alt_deg), math.radians(moon_az_deg)
        cos_s = math.sin(a1) * math.sin(a2) + math.cos(a1) * math.cos(a2) * math.cos(z1 - z2)
        sep = math.degrees(math.acos(max(-1.0, min(1.0, cos_s))))
        return sep < 18.0

    # 1. Bright navigation stars
    for name, ra_h, dec_d, vmag in NAVIGATION_STARS:
        star = ephem.FixedBody()
        star._ra = math.radians(ra_h * 15.0)
        star._dec = math.radians(dec_d)
        star._epoch = ephem.J2000
        star.compute(obs)
        alt_d = math.degrees(float(star.alt))
        az_d = math.degrees(float(star.az))
        if alt_d >= min_alt_deg and not _near_moon(alt_d, az_d):
            alt_r = math.radians(alt_d)
            az_r = math.radians(az_d)
            catalog.append(
                {
                    "name": name,
                    "alt_deg": alt_d,
                    "az_deg": az_d,
                    "mag": vmag,
                    "east": math.cos(alt_r) * math.sin(az_r),
                    "north": math.cos(alt_r) * math.cos(az_r),
                    "up": math.sin(alt_r),
                }
            )

    # 2. Bright planets (Venus, Jupiter, Mars, Saturn)
    for pname, pcls in [("Venus", ephem.Venus), ("Jupiter", ephem.Jupiter), ("Mars", ephem.Mars), ("Saturn", ephem.Saturn)]:
        body = pcls()
        body.compute(obs)
        alt_d = math.degrees(float(body.alt))
        az_d = math.degrees(float(body.az))
        vmag = float(body.mag)
        if alt_d >= min_alt_deg and vmag <= 2.2 and not _near_moon(alt_d, az_d):
            alt_r = math.radians(alt_d)
            az_r = math.radians(az_d)
            catalog.append(
                {
                    "name": pname,
                    "alt_deg": alt_d,
                    "az_deg": az_d,
                    "mag": vmag,
                    "east": math.cos(alt_r) * math.sin(az_r),
                    "north": math.cos(alt_r) * math.cos(az_r),
                    "up": math.sin(alt_r),
                }
            )

    catalog.sort(key=lambda c: c["mag"])
    return catalog, sun_alt_deg


def extract_star_centroids_3552(bgr: np.ndarray, cfg: dict) -> Tuple[np.ndarray, int]:
    """
    Extracts sub-pixel bright star centroids in normalized 3552x3552 sensor coordinates
    within the upper sky dome (radial distance <= 1260 px from sensor center, Alt >= ~24 deg),
    completely independent of horizon trees or roofline orientation.
    Returns (centroids_Nx3 [x_3552, y_3552, flux], total_sharp_stars_count).
    """
    h, w = bgr.shape[:2]
    scale_to_3552 = 3552.0 / float(w)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Upper dome mask centered on optical axis (radius 1260 px in 3552 space -> Alt >= ~24 deg)
    cx0, cy0 = cfg.get("zenith_center", [1750.95, 1690.25])
    cx_w = int(round(float(cx0) / scale_to_3552))
    cy_w = int(round(float(cy0) / scale_to_3552))
    r_dome = int(round(1260.0 / scale_to_3552))

    dome_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(dome_mask, (cx_w, cy_w), r_dome, 255, -1)

    # Mask out any colored overlay boxes/text (if testing on an annotated snapshot)
    b_ch = bgr[:, :, 0].astype(np.int16)
    g_ch = bgr[:, :, 1].astype(np.int16)
    r_ch = bgr[:, :, 2].astype(np.int16)
    chroma = np.maximum(np.maximum(b_ch, g_ch), r_ch) - np.minimum(np.minimum(b_ch, g_ch), r_ch)
    ann_mask = cv2.dilate((chroma >= 85).astype(np.uint8), np.ones((25, 25), dtype=np.uint8))
    dome_mask[ann_mask > 0] = 0

    bg = cv2.GaussianBlur(gray, (25, 25), 0)
    hp = cv2.subtract(gray, bg)

    # Mask bright moonlit clouds / glare
    dome_mask[bg > 135] = 0

    star_bin = ((hp > 18) & (dome_mask > 0)).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(star_bin)
    if n_labels <= 1:
        return np.zeros((0, 3), dtype=np.float32), 0

    min_area = 3 if w <= 2000 else 5
    max_area = 190 if w <= 2000 else 550
    peaks: List[Tuple[float, float, float]] = []
    sharp_stars = 0

    for idx in range(1, n_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        aspect = float(max(bw, bh)) / max(1.0, float(min(bw, bh)))
        if aspect > 2.5:
            continue
        if min_area <= area <= max_area:
            x0 = int(stats[idx, cv2.CC_STAT_LEFT])
            y0 = int(stats[idx, cv2.CC_STAT_TOP])
            patch_hp = hp[y0 : y0 + bh, x0 : x0 + bw].astype(np.float32)
            peak_val = float(np.max(patch_hp))
            if peak_val < 24.0:
                continue
            sharp_stars += 1
            # Use raw grayscale above local background for accurate centroid of bright/saturated cores
            patch_gray = gray[y0 : y0 + bh, x0 : x0 + bw].astype(np.float32)
            patch_w = np.maximum(0.0, patch_gray - float(np.min(patch_gray))) ** 2
            yy, xx = np.mgrid[y0 : y0 + bh, x0 : x0 + bw]
            w_sum = float(np.sum(patch_w))
            if w_sum > 0:
                cx_sub = float(np.sum(xx * patch_w)) / w_sum
                cy_sub = float(np.sum(yy * patch_w)) / w_sum
            else:
                cx_sub, cy_sub = float(centroids[idx][0]), float(centroids[idx][1])
            flux_score = float(np.sum(patch_hp)) + peak_val * 4.0
            peaks.append((cx_sub * scale_to_3552, cy_sub * scale_to_3552, flux_score))

    peaks.sort(key=lambda p: p[2], reverse=True)
    top_peaks = peaks[:75]
    if not top_peaks:
        return np.zeros((0, 3), dtype=np.float32), sharp_stars
    return np.array(top_peaks, dtype=np.float32), sharp_stars


def solve_star_alignment_from_frame(
    bgr: np.ndarray,
    dt_utc: datetime.datetime,
    cfg: dict,
    min_sharp_stars: int = 55,
    min_matched_catalog: int = 10,
    max_rms_deg: float = 0.22,
) -> Dict[str, Any]:
    """
    Performs a full 360-degree blind 3D spherical star-plate solve on `bgr` captured at `dt_utc`.
    1. Checks astronomical night & clear-sky star count (`>= min_sharp_stars`).
    2. Sweeps all 0..360 deg compass rotations + tilts to find global camera orientation.
    3. Refines (bottom_azimuth_deg, tilt_x_deg, tilt_y_deg, zenith_center, horizon_radius_px)
       via iterative outlier-trimmed coordinate optimization.
    """
    catalog, sun_alt_deg = _compute_visible_catalog(dt_utc, cfg, min_alt_deg=24.0)
    if sun_alt_deg > -12.0:
        return {
            "ok": False,
            "reason": f"Waiting for astronomical darkness (Sun altitude is {sun_alt_deg:.1f}°, need <= -12.0°).",
            "sun_alt_deg": round(sun_alt_deg, 1),
        }

    if len(catalog) < 10:
        return {
            "ok": False,
            "reason": f"Too few bright catalog stars above 24° altitude ({len(catalog)} available).",
        }

    det_peaks, sharp_stars = extract_star_centroids_3552(bgr, cfg)
    if sharp_stars < min_sharp_stars or len(det_peaks) < 15:
        return {
            "ok": False,
            "reason": (
                f"Sky not clear enough yet ({sharp_stars} sharp stars visible in upper dome; "
                f"waiting for a clear starry sky with >= {min_sharp_stars} stars)."
            ),
            "sharp_stars": sharp_stars,
        }

    east = np.array([c["east"] for c in catalog], dtype=np.float32)
    north = np.array([c["north"] for c in catalog], dtype=np.float32)
    up = np.array([c["up"] for c in catalog], dtype=np.float32)
    mag_weights = np.array([max(0.5, 3.2 - c["mag"]) for c in catalog], dtype=np.float32)

    det_xy = det_peaks[:, :2]  # (M, 2)
    det_flux_rank = np.linspace(1.5, 0.5, len(det_xy), dtype=np.float32)

    cx_init, cy_init = cfg.get("zenith_center", [1750.95, 1690.25])
    r_init = float(cfg.get("horizon_radius_px", 2007.74))
    k2 = float(cfg.get("radial_k2", -0.14857))
    k3 = float(cfg.get("radial_k3", 0.03125))

    # --- STEP 1: Blind 360-deg Coarse Grid Search ---
    best_coarse_score = -1e9
    best_params = (
        float(cfg.get("bottom_azimuth_deg", 265.055)),
        float(cfg.get("tilt_x_deg", 6.292)),
        float(cfg.get("tilt_y_deg", -0.947)),
    )

    for tx in np.arange(-12.0, 12.1, 3.0):
        for ty in np.arange(-12.0, 12.1, 3.0):
            for az in np.arange(0.0, 360.0, 2.0):
                px, py = _project_vecs_to_pixels(east, north, up, float(az), float(tx), float(ty), cx_init, cy_init, r_init, k2, k3)
                # Compute nearest detected star distance for each catalog star
                dx = px[:, None] - det_xy[None, :, 0]
                dy = py[:, None] - det_xy[None, :, 1]
                d2 = dx * dx + dy * dy
                min_idx = np.argmin(d2, axis=1)
                min_d = np.sqrt(d2[np.arange(len(catalog)), min_idx])
                inliers = min_d < 36.0
                n_in = int(np.sum(inliers))
                if n_in >= 6:
                    score = float(np.sum(mag_weights[inliers] * det_flux_rank[min_idx[inliers]] * (1.0 - min_d[inliers] / 42.0)))
                    if score > best_coarse_score:
                        best_coarse_score = score
                        best_params = (float(az), float(tx), float(ty))

    # --- STEP 2: Multi-Stage Fine Coordinate Refinement ---
    cur_az, cur_tx, cur_ty = best_params
    cur_cx, cur_cy = float(cx_init), float(cy_init)
    cur_r = float(r_init)

    for stage_radius, step_scale in [(34.0, 1.0), (20.0, 0.35), (12.0, 0.12), (9.0, 0.04)]:
        for _ in range(18):
            # Evaluate current inlier pairs
            px0, py0 = _project_vecs_to_pixels(east, north, up, cur_az, cur_tx, cur_ty, cur_cx, cur_cy, cur_r, k2, k3)
            dx = px0[:, None] - det_xy[None, :, 0]
            dy = py0[:, None] - det_xy[None, :, 1]
            d2 = dx * dx + dy * dy
            min_idx = np.argmin(d2, axis=1)
            min_d = np.sqrt(d2[np.arange(len(catalog)), min_idx])
            inliers = min_d < stage_radius
            if int(np.sum(inliers)) < 6:
                break

            def _loss(p_az: float, p_tx: float, p_ty: float, p_cx: float, p_cy: float, p_r: float) -> float:
                px, py = _project_vecs_to_pixels(
                    east[inliers], north[inliers], up[inliers], p_az, p_tx, p_ty, p_cx, p_cy, p_r, k2, k3
                )
                tx_det = det_xy[min_idx[inliers], 0]
                ty_det = det_xy[min_idx[inliers], 1]
                err2 = (px - tx_det) ** 2 + (py - ty_det) ** 2
                # Huber-like robust loss
                return float(np.sum(mag_weights[inliers] * np.sqrt(err2 + 4.0)))

            base_l = _loss(cur_az, cur_tx, cur_ty, cur_cx, cur_cy, cur_r)
            improved = False
            for idx_p, delta in [
                (0, 0.45 * step_scale),
                (1, 0.30 * step_scale),
                (2, 0.30 * step_scale),
                (3, 4.0 * step_scale),
                (4, 4.0 * step_scale),
                (5, 4.0 * step_scale),
            ]:
                for sign in (-1.0, 1.0):
                    cand = [cur_az, cur_tx, cur_ty, cur_cx, cur_cy, cur_r]
                    cand[idx_p] += sign * delta
                    cand[0] = cand[0] % 360.0
                    # Keep horizon radius and center within physical bounds
                    cand[3] = max(1650.0, min(1850.0, cand[3]))
                    cand[4] = max(1590.0, min(1790.0, cand[4]))
                    cand[5] = max(1920.0, min(2090.0, cand[5]))
                    l_val = _loss(*cand)
                    if l_val < base_l - 1e-4:
                        cur_az, cur_tx, cur_ty, cur_cx, cur_cy, cur_r = cand
                        base_l = l_val
                        improved = True
            if not improved:
                break

    # Final evaluation of matched catalog stars within 11.5 px (~0.33 deg)
    px_f, py_f = _project_vecs_to_pixels(east, north, up, cur_az, cur_tx, cur_ty, cur_cx, cur_cy, cur_r, k2, k3)
    dx = px_f[:, None] - det_xy[None, :, 0]
    dy = py_f[:, None] - det_xy[None, :, 1]
    d2 = dx * dx + dy * dy
    min_idx = np.argmin(d2, axis=1)
    min_d = np.sqrt(d2[np.arange(len(catalog)), min_idx])

    match_mask = min_d <= 11.5
    matched_indices = np.where(match_mask)[0]
    matched_count = len(matched_indices)

    if matched_count < min_matched_catalog:
        return {
            "ok": False,
            "reason": (
                f"Found {sharp_stars} stars, but only {matched_count}/{min_matched_catalog} bright catalog stars "
                f"matched cleanly. Waiting for clearer sky..."
            ),
            "sharp_stars": sharp_stars,
            "matched_stars": matched_count,
        }

    matched_errs_px = min_d[matched_indices]
    rms_px = float(np.sqrt(np.mean(matched_errs_px ** 2)))
    median_px = float(np.median(matched_errs_px))
    rms_deg = rms_px * (90.0 / cur_r)
    median_deg = median_px * (90.0 / cur_r)

    if rms_deg > max_rms_deg:
        return {
            "ok": False,
            "reason": f"Astrometric RMS ({rms_deg:.2f}°) exceeds threshold ({max_rms_deg:.2f}°). Waiting for steadier sky...",
            "sharp_stars": sharp_stars,
            "matched_stars": matched_count,
            "rms_deg": round(rms_deg, 3),
        }

    matched_names = [catalog[i]["name"] for i in matched_indices]
    return {
        "ok": True,
        "sharp_stars": sharp_stars,
        "matched_stars": matched_count,
        "matched_names": matched_names,
        "rms_px": round(rms_px, 2),
        "rms_deg": round(rms_deg, 3),
        "median_deg": round(median_deg, 3),
        "bottom_azimuth_deg": round(cur_az % 360.0, 3),
        "tilt_x_deg": round(cur_tx, 3),
        "tilt_y_deg": round(cur_ty, 3),
        "zenith_center": [round(cur_cx, 2), round(cur_cy, 2)],
        "horizon_radius_px": round(cur_r, 2),
    }


def process_frame_for_realignment(
    bgr: np.ndarray,
    frame_filename: str,
    dt_utc: datetime.datetime,
    cfg: dict,
    save_config_fn,
    reset_sky_mask_fn,
) -> Optional[Dict[str, Any]]:
    """
    Called by `detector.py` on each new frame ONLY when `realignment_state.json` is armed.
    - Updates the daytime sky mask (`reference_day.jpg` -> `sky_mask.png`) if a clear daytime frame is seen.
    - Runs the blind 360-deg star solver on clear night frames and commits the new calibration
      once 2 independent clear frames agree within 0.35 deg.
    """
    state = load_realign_state()
    if not state.get("armed"):
        return None

    now_iso = datetime.datetime.now().isoformat(timespec="seconds")
    state["frames_evaluated"] = int(state.get("frames_evaluated", 0)) + 1
    state["last_check_iso"] = now_iso

    # Check Sun altitude
    obs = ephem.Observer()
    obs.lat = str(cfg.get("camera_lat", 45.0000))
    obs.lon = str(cfg.get("camera_lon", -100.0000))
    obs.elevation = float(cfg.get("camera_alt_ft", 130.0)) * 0.3048
    obs.date = dt_utc.strftime("%Y/%m/%d %H:%M:%S")
    sun = ephem.Sun()
    sun.compute(obs)
    sun_alt_deg = math.degrees(float(sun.alt))

    # 1. Daytime Horizon Tree/Roofline Mask Refresh
    if not state.get("day_mask_updated") and sun_alt_deg >= 10.0:
        b_ch, g_ch, _ = cv2.split(bgr)
        sky_pixels = int(np.sum((b_ch > 85) & (g_ch > 70)))
        total_px = bgr.shape[0] * bgr.shape[1]
        if sky_pixels > int(total_px * 0.22):
            try:
                ref_3552 = bgr if bgr.shape[:2] == (3552, 3552) else cv2.resize(bgr, (3552, 3552), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(REF_DAY_FILE, ref_3552, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
                if os.path.exists(MASK_PREVIEW_FILE):
                    os.remove(MASK_PREVIEW_FILE)
                reset_sky_mask_fn(cfg, (3552, 3552))
                state["day_mask_updated"] = True
                logger.info("Realignment: Updated daytime reference_day.jpg and rebuilt sky_mask.png.")
            except Exception as e:
                logger.warning(f"Realignment daytime mask update error: {e}")

    # 2. Nighttime 360-deg Blind Star-Field Alignment
    res = solve_star_alignment_from_frame(bgr, dt_utc, cfg)
    if not res.get("ok"):
        state["last_check_msg"] = res.get("reason", "Waiting for clear starry night...")
        save_realign_state(state)
        return state

    # Valid candidate frame solution!
    cand_entry = {
        "frame_name": frame_filename,
        "solved_at_iso": now_iso,
        "bottom_azimuth_deg": res["bottom_azimuth_deg"],
        "tilt_x_deg": res["tilt_x_deg"],
        "tilt_y_deg": res["tilt_y_deg"],
        "zenith_center": res["zenith_center"],
        "horizon_radius_px": res["horizon_radius_px"],
        "matched_stars": res["matched_stars"],
        "matched_names": res["matched_names"],
        "rms_deg": res["rms_deg"],
        "sharp_stars": res["sharp_stars"],
    }
    cands = state.get("candidate_solutions", [])
    cands.append(cand_entry)
    cands = cands[-6:]
    state["candidate_solutions"] = cands

    # Check if we have at least 2 candidate solutions that agree within 0.35 deg on bottom_azimuth_deg
    agreeing = []
    for c in cands:
        diff_az = abs(((c["bottom_azimuth_deg"] - cand_entry["bottom_azimuth_deg"] + 180.0) % 360.0) - 180.0)
        if diff_az <= 0.35:
            agreeing.append(c)

    if len(agreeing) < 2:
        state["last_check_msg"] = (
            f"Pass 1/2 solved on {frame_filename}: matched {res['matched_stars']} catalog stars "
            f"(RMS {res['rms_deg']:.2f}°, Az {res['bottom_azimuth_deg']:.2f}°). "
            f"Verifying on next clear frame..."
        )
        save_realign_state(state)
        return state

    # Average the agreeing solutions (circular mean for azimuth)
    sin_sum = sum(math.sin(math.radians(c["bottom_azimuth_deg"])) for c in agreeing)
    cos_sum = sum(math.cos(math.radians(c["bottom_azimuth_deg"])) for c in agreeing)
    final_az = round(math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0, 3)
    final_tx = round(float(np.mean([c["tilt_x_deg"] for c in agreeing])), 3)
    final_ty = round(float(np.mean([c["tilt_y_deg"] for c in agreeing])), 3)
    final_cx = round(float(np.mean([c["zenith_center"][0] for c in agreeing])), 2)
    final_cy = round(float(np.mean([c["zenith_center"][1] for c in agreeing])), 2)
    final_r = round(float(np.mean([c["horizon_radius_px"] for c in agreeing])), 2)

    old_cal = {
        "bottom_azimuth_deg": cfg.get("bottom_azimuth_deg", 265.055),
        "tilt_x_deg": cfg.get("tilt_x_deg", 6.292),
        "tilt_y_deg": cfg.get("tilt_y_deg", -0.947),
        "zenith_center": cfg.get("zenith_center", [1750.95, 1690.25]),
        "horizon_radius_px": cfg.get("horizon_radius_px", 2007.74),
        "radial_k2": cfg.get("radial_k2", -0.14857),
        "radial_k3": cfg.get("radial_k3", 0.03125),
        "mask_center": cfg.get("mask_center", [1718, 1796]),
    }

    # Archive previous calibration epoch so historical nights before the move keep their alignment
    armed_night = state.get("armed_night_tag") or (datetime.datetime.now() - datetime.timedelta(hours=12)).strftime("%Y%m%d")
    epochs = cfg.get("calibration_epochs", [])
    # Only append if the parameters actually changed by > 0.05 deg or > 1 px
    az_delta = abs(((final_az - float(old_cal["bottom_azimuth_deg"]) + 180.0) % 360.0) - 180.0)
    if az_delta > 0.05 or abs(final_tx - float(old_cal["tilt_x_deg"])) > 0.05 or abs(final_ty - float(old_cal["tilt_y_deg"])) > 0.05:
        epochs.append(
            {
                "BeforeNight": armed_night,
                "archived_at_iso": now_iso,
                "calibration": old_cal,
            }
        )
        cfg["calibration_epochs"] = epochs[-20:]

    # Commit new calibration to config.json
    cfg["bottom_azimuth_deg"] = final_az
    cfg["tilt_x_deg"] = final_tx
    cfg["tilt_y_deg"] = final_ty
    cfg["zenith_center"] = [final_cx, final_cy]
    cfg["horizon_radius_px"] = final_r
    save_config_fn(cfg)

    new_cal = {
        "bottom_azimuth_deg": final_az,
        "tilt_x_deg": final_tx,
        "tilt_y_deg": final_ty,
        "zenith_center": [final_cx, final_cy],
        "horizon_radius_px": final_r,
    }

    state.update(
        {
            "armed": False,
            "status": "completed",
            "star_alignment_solved": True,
            "last_check_iso": now_iso,
            "last_check_msg": (
                f"✅ Realignment Complete & Locked! Matched {res['matched_stars']} stars/planets "
                f"(RMS {res['rms_deg']:.2f}°) — Azimuth {final_az:.2f}°, Tilt ({final_tx:.2f}°, {final_ty:.2f}°)."
            ),
            "last_solved_result": {
                "completed_at_iso": now_iso,
                "frame_name": frame_filename,
                "matched_stars": res["matched_stars"],
                "matched_names": res["matched_names"],
                "sharp_stars": res["sharp_stars"],
                "rms_deg": res["rms_deg"],
                "rms_px": res["rms_px"],
                "old_calibration": old_cal,
                "new_calibration": new_cal,
            },
        }
    )
    save_realign_state(state)
    logger.info(
        f"Camera Realignment COMPLETE: matched={res['matched_stars']} stars, RMS={res['rms_deg']:.3f} deg, "
        f"new bottom_az={final_az:.3f} (was {old_cal['bottom_azimuth_deg']}), "
        f"tilt=({final_tx:.3f}, {final_ty:.3f})."
    )
    return state
