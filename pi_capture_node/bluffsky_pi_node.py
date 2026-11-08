#!/usr/bin/env python3
"""
BluffSky Option A — Purpose-Built Allsky Science & Stereo Capture Daemon
========================================================================
Runs natively on Raspberry Pi 5 (ARM64) with ZWO ASI676MC (Sony IMX676 Starvis 2).

Key Architectural Improvements over Legacy Allsky (capture_ZWO + saveImage.sh):
1. Pure In-Memory Pipeline (/dev/shm & tmpfs):
   - Zero per-frame bash/python subprocess spawning (eliminatesflow-runner.py & lftp).
   - Single-pass JPEG encoding in RAM with atomic POSIX rename.
   - Un-annotated pristine science frames (no burned-in red text over stars/satellites).
2. 4-Regime Dynamic Exposure, Gain & White Balance Controller:
   - Regime 1 (DAY): Sun > -0.83 deg -> Gain 0 (max full-well), 32us - 2.0s exposure.
   - Regime 2 (TWILIGHT): -12 deg < Sun <= -0.83 deg -> Damped log-PI exposure & gain ramp
     with adaptive twilight cadence (eliminates the legacy 34.5s twilight dead-time bug).
   - Regime 3 (DARK_NIGHT): Sun <= -12 deg, dark sky -> Locks Sony IMX676 HCG low-read-noise
     sweet spot (Gain 300, ~0.55 e- read noise), up to 30-35s exposure, with star-core
     99.95th percentile saturation guard.
   - Regime 4 (MOONLIT_NIGHT): Sun <= -12 deg, Moon above horizon (>=20% illumination) or
     bright cloud deck -> Masks out 250px exclusion disc around live Moon (x, y) dome
     coordinates + horizon trees, meters interquartile sky background (25th-75th percentile),
     and adapts gain (180-300) and exposure so moonlight never washes out the sky.
3. Deterministic UTC Modulo Shutter Synchronization (Dual-Pi Stereo Ready):
   - Supports lockstep stereo (offset=0), 50% phase-staggered (offset=period/2), and
     adaptive single-camera cadence locked to Chrony/NTP UTC clock boundaries (<1 ms jitter).
4. 100% Drop-In Compatibility with Windows Server & Legacy Endpoints:
   - /current/tmp/image.jpg           (live frame)
   - /current/tmp/overlaydebug.txt    (legacy telemetry key-value file + microsecond fields)
   - /current/tmp/frame_meta.json     (microsecond start/end timestamps & regime sidecar)
   - /config/status.json              (monitored by allsky_workflow_guard.py)
   - /images/<YYYYMMDD>/image-*.jpg   (+ thumbnails/100x75, pruned to daystokeep)
   - Background FTP upload thread to 192.168.1.10:21 (feeds AllSkyMonitorService)
   - Live HTTP Telemetry & Control API on port 8096 (/api/status, /api/config)
"""

from __future__ import annotations

import argparse
import ctypes as c
import datetime
import ftplib
import http.server
import io
import json
import logging
import math
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import ephem
import numpy as np

# Optional ultra-fast libjpeg-turbo binding (catch ValueError if system simplejpeg ABI mismatches numpy 2.x)
try:
    import simplejpeg
    HAS_SIMPLEJPEG = True
except Exception:
    HAS_SIMPLEJPEG = False


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bluffsky_node")


# ---------------------------------------------------------------------------
# ZWO ASI SDK Ctypes Bindings (ARM64 / x86_64)
# ---------------------------------------------------------------------------
ASI_BAYER_RG = 0
ASI_BAYER_BG = 1
ASI_BAYER_GR = 2
ASI_BAYER_RB = 3

ASI_IMG_RAW8 = 0
ASI_IMG_RGB24 = 1
ASI_IMG_RAW16 = 2
ASI_IMG_Y8 = 3
ASI_IMG_END = -1

ASI_GAIN = 0
ASI_EXPOSURE = 1
ASI_GAMMA = 2
ASI_WB_R = 3
ASI_WB_B = 4
ASI_OFFSET = 5
ASI_BANDWIDTHOVERLOAD = 6
ASI_OVERCLOCK = 7
ASI_TEMPERATURE = 8       # returns 10 * Celsius
ASI_FLIP = 9
ASI_AUTO_MAX_GAIN = 10
ASI_AUTO_MAX_EXP = 11
ASI_AUTO_MAX_BRIGHTNESS = 12
ASI_HARDWARE_BIN = 13
ASI_HIGH_SPEED_MODE = 14
ASI_COOLER_POWER_PERC = 15
ASI_TARGET_TEMP = 16
ASI_COOLER_ON = 17
ASI_MONO_BIN = 18
ASI_FAN_ON = 19
ASI_PATTERN_ADJUST = 20

ASI_SUCCESS = 0


class _ASI_CAMERA_INFO(c.Structure):
    _fields_ = [
        ("Name", c.c_char * 64),
        ("CameraID", c.c_int),
        ("MaxHeight", c.c_long),
        ("MaxWidth", c.c_long),
        ("IsColorCam", c.c_int),
        ("BayerPattern", c.c_int),
        ("SupportedBins", c.c_int * 16),
        ("SupportedVideoFormat", c.c_int * 8),
        ("PixelSize", c.c_double),
        ("MechanicalShutter", c.c_int),
        ("ST4Port", c.c_int),
        ("IsCoolerCam", c.c_int),
        ("IsUSB3Host", c.c_int),
        ("IsUSB3Camera", c.c_int),
        ("ElecPerADU", c.c_float),
        ("BitDepth", c.c_int),
        ("IsTriggerCam", c.c_int),
        ("Unused", c.c_char * 16),
    ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Name": self.Name.decode("utf-8", errors="replace").strip("\x00"),
            "CameraID": int(self.CameraID),
            "MaxHeight": int(self.MaxHeight),
            "MaxWidth": int(self.MaxWidth),
            "IsColorCam": bool(self.IsColorCam),
            "BayerPattern": int(self.BayerPattern),
            "PixelSize": float(self.PixelSize),
            "IsUSB3Host": bool(self.IsUSB3Host),
            "IsUSB3Camera": bool(self.IsUSB3Camera),
            "ElecPerADU": float(self.ElecPerADU),
            "BitDepth": int(self.BitDepth),
        }


class ASICameraDriver:
    """Direct ctypes interface to ZWO libASICamera2.so."""

    def __init__(self, lib_path: str, cam_index: int = 0):
        self.lib_path = lib_path
        self.cam_index = cam_index
        self.cam_id = cam_index
        self.lib = c.CDLL(lib_path)
        self._bind_functions()
        self.info: Dict[str, Any] = {}
        self.width = 3552
        self.height = 3552
        self.image_type = ASI_IMG_RGB24
        self.bpp = 3
        self.buffer: Optional[c.Array] = None
        self.is_open = False

    def _bind_functions(self) -> None:
        self.lib.ASIGetNumOfConnectedCameras.argtypes = []
        self.lib.ASIGetNumOfConnectedCameras.restype = c.c_int

        self.lib.ASIGetCameraProperty.argtypes = [c.POINTER(_ASI_CAMERA_INFO), c.c_int]
        self.lib.ASIGetCameraProperty.restype = c.c_int

        self.lib.ASIOpenCamera.argtypes = [c.c_int]
        self.lib.ASIOpenCamera.restype = c.c_int

        self.lib.ASIInitCamera.argtypes = [c.c_int]
        self.lib.ASIInitCamera.restype = c.c_int

        self.lib.ASICloseCamera.argtypes = [c.c_int]
        self.lib.ASICloseCamera.restype = c.c_int

        self.lib.ASIGetControlValue.argtypes = [
            c.c_int,
            c.c_int,
            c.POINTER(c.c_long),
            c.POINTER(c.c_int),
        ]
        self.lib.ASIGetControlValue.restype = c.c_int

        self.lib.ASISetControlValue.argtypes = [c.c_int, c.c_int, c.c_long, c.c_int]
        self.lib.ASISetControlValue.restype = c.c_int

        self.lib.ASISetROIFormat.argtypes = [c.c_int, c.c_int, c.c_int, c.c_int, c.c_int]
        self.lib.ASISetROIFormat.restype = c.c_int

        self.lib.ASIStartVideoCapture.argtypes = [c.c_int]
        self.lib.ASIStartVideoCapture.restype = c.c_int

        self.lib.ASIStopVideoCapture.argtypes = [c.c_int]
        self.lib.ASIStopVideoCapture.restype = c.c_int

        self.lib.ASIGetVideoData.argtypes = [
            c.c_int,
            c.POINTER(c.c_char),
            c.c_long,
            c.c_int,
        ]
        self.lib.ASIGetVideoData.restype = c.c_int

    def open(
        self,
        binning: int = 1,
        image_type: int = ASI_IMG_RGB24,
        usb_bandwidth: int = 50,
        auto_usb: bool = True,
        flip: int = 0,
    ) -> Dict[str, Any]:
        num = self.lib.ASIGetNumOfConnectedCameras()
        if num <= 0:
            raise RuntimeError("No ZWO ASI cameras detected on USB bus.")

        prop = _ASI_CAMERA_INFO()
        rc = self.lib.ASIGetCameraProperty(c.byref(prop), self.cam_index)
        if rc != ASI_SUCCESS:
            raise RuntimeError(f"ASIGetCameraProperty failed with code {rc}")

        self.info = prop.to_dict()
        self.cam_id = self.info["CameraID"]

        rc = self.lib.ASIOpenCamera(self.cam_id)
        if rc != ASI_SUCCESS:
            raise RuntimeError(f"ASIOpenCamera({self.cam_id}) failed with code {rc}")

        rc = self.lib.ASIInitCamera(self.cam_id)
        if rc != ASI_SUCCESS:
            self.lib.ASICloseCamera(self.cam_id)
            raise RuntimeError(f"ASIInitCamera({self.cam_id}) failed with code {rc}")

        self.is_open = True
        self.width = (self.info["MaxWidth"] // binning // 8) * 8
        self.height = (self.info["MaxHeight"] // binning // 2) * 2
        self.image_type = image_type
        self.bpp = 3 if image_type == ASI_IMG_RGB24 else (2 if image_type == ASI_IMG_RAW16 else 1)

        rc = self.lib.ASISetROIFormat(self.cam_id, self.width, self.height, binning, image_type)
        if rc != ASI_SUCCESS:
            raise RuntimeError(
                f"ASISetROIFormat({self.width}x{self.height}, bin={binning}, type={image_type}) failed: {rc}"
            )

        self.set_control(ASI_BANDWIDTHOVERLOAD, usb_bandwidth, auto=auto_usb)
        self.set_control(ASI_FLIP, flip, auto=False)
        self.set_control(ASI_HIGH_SPEED_MODE, 0, auto=False)

        buf_size = self.width * self.height * self.bpp
        self.buffer = c.create_string_buffer(buf_size)
        logger.info(
            "Opened %s (ID=%d): %dx%d, PixelSize=%.2fum, USB3=%s, Buffer=%.2f MB",
            self.info["Name"],
            self.cam_id,
            self.width,
            self.height,
            self.info["PixelSize"],
            self.info["IsUSB3Camera"],
            buf_size / (1024 * 1024),
        )
        return self.info

    def set_control(self, ctrl_type: int, value: int, auto: bool = False) -> None:
        rc = self.lib.ASISetControlValue(
            self.cam_id, ctrl_type, c.c_long(int(value)), c.c_int(1 if auto else 0)
        )
        if rc != ASI_SUCCESS:
            logger.warning("ASISetControlValue(ctrl=%d, val=%d) returned %d", ctrl_type, value, rc)

    def get_control(self, ctrl_type: int) -> Tuple[int, bool]:
        val = c.c_long(0)
        auto = c.c_int(0)
        rc = self.lib.ASIGetControlValue(self.cam_id, ctrl_type, c.byref(val), c.byref(auto))
        if rc != ASI_SUCCESS:
            return 0, False
        return int(val.value), bool(auto.value)

    def get_temperature_c(self) -> float:
        raw_temp, _ = self.get_control(ASI_TEMPERATURE)
        return round(raw_temp / 10.0, 1)

    def capture_frame(
        self,
        exposure_us: int,
        gain: int,
        wb_r: int,
        wb_b: int,
        auto_wb: bool = False,
    ) -> Tuple[np.ndarray, int, int]:
        """
        Captures a single frame with deterministic start/end microsecond timestamps.
        Uses ZWOvideoOff sequence (ASIStartVideoCapture -> ASIGetVideoData -> ASIStopVideoCapture)
        so exposure & gain changes take effect immediately on this exact frame and integration
        starts immediately upon ASIStartVideoCapture().
        Returns (frame_bgr, exposure_start_unix_us, exposure_end_unix_us).
        """
        self.set_control(ASI_EXPOSURE, exposure_us, auto=False)
        self.set_control(ASI_GAIN, gain, auto=False)
        self.set_control(ASI_WB_R, wb_r, auto=auto_wb)
        self.set_control(ASI_WB_B, wb_b, auto=auto_wb)

        buf_size = self.width * self.height * self.bpp
        timeout_ms = int(max(2500, (exposure_us // 1000) * 2 + 5000))

        t_start_us = time.time_ns() // 1000
        rc = self.lib.ASIStartVideoCapture(self.cam_id)
        if rc != ASI_SUCCESS:
            raise RuntimeError(f"ASIStartVideoCapture failed with code {rc}")

        try:
            rc = self.lib.ASIGetVideoData(self.cam_id, self.buffer, c.c_long(buf_size), timeout_ms)
            t_end_us = time.time_ns() // 1000
        finally:
            self.lib.ASIStopVideoCapture(self.cam_id)

        if rc != ASI_SUCCESS:
            raise RuntimeError(f"ASIGetVideoData failed with code {rc} (exposure={exposure_us}us)")

        # Refine exposure_start_unix_us from readout completion minus USB transfer overhead (~85ms)
        usb_overhead_us = 85_000
        refined_start_us = max(t_start_us, t_end_us - exposure_us - usb_overhead_us)

        if self.image_type == ASI_IMG_RGB24:
            arr = np.frombuffer(self.buffer, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
            return arr, refined_start_us, t_end_us
        elif self.image_type == ASI_IMG_RAW16:
            raw16 = np.frombuffer(self.buffer, dtype=np.uint16).reshape((self.height, self.width))
            bgr16 = cv2.cvtColor(raw16, cv2.COLOR_BayerRG2BGR)
            arr8 = (bgr16 >> 8).astype(np.uint8)
            return arr8, refined_start_us, t_end_us
        else:
            raw8 = np.frombuffer(self.buffer, dtype=np.uint8).reshape((self.height, self.width)).copy()
            bgr8 = cv2.cvtColor(raw8, cv2.COLOR_BayerRG2BGR)
            return bgr8, refined_start_us, t_end_us

    def close(self) -> None:
        if self.is_open:
            try:
                self.lib.ASIStopVideoCapture(self.cam_id)
            except Exception:
                pass
            try:
                self.lib.ASICloseCamera(self.cam_id)
            except Exception:
                pass
            self.is_open = False


# ---------------------------------------------------------------------------
# Ephemeris & 4-Regime Dynamic Exposure / Gain Controller
# ---------------------------------------------------------------------------
class EphemerisTracker:
    """Computes live Sun & Moon altitude, azimuth, illumination, and dome (x, y) coords."""

    def __init__(self, lat_deg: float, lon_deg: float, elev_m: float = 39.6):
        self.lat_deg = lat_deg
        self.lon_deg = lon_deg
        self.elev_m = elev_m

    def compute(self, dt_utc: datetime.datetime, img_w: int = 3552, img_h: int = 3552) -> Dict[str, Any]:
        obs = ephem.Observer()
        obs.lat = str(self.lat_deg)
        obs.lon = str(self.lon_deg)
        obs.elevation = self.elev_m
        obs.date = dt_utc.strftime("%Y/%m/%d %H:%M:%S")

        sun = ephem.Sun(obs)
        moon = ephem.Moon(obs)

        sun_alt_deg = math.degrees(float(sun.alt))
        sun_az_deg = math.degrees(float(sun.az))
        moon_alt_deg = math.degrees(float(moon.alt))
        moon_az_deg = math.degrees(float(moon.az))
        moon_illum = float(moon.phase)

        sun_xy = self._altaz_to_dome_xy(sun_alt_deg, sun_az_deg, img_w, img_h)
        moon_xy = self._altaz_to_dome_xy(moon_alt_deg, moon_az_deg, img_w, img_h)

        return {
            "sun_alt_deg": round(sun_alt_deg, 2),
            "sun_az_deg": round(sun_az_deg, 2),
            "moon_alt_deg": round(moon_alt_deg, 2),
            "moon_az_deg": round(moon_az_deg, 2),
            "moon_illumination_pct": round(moon_illum, 2),
            "sun_xy": sun_xy,
            "moon_xy": moon_xy,
        }

    @staticmethod
    def _altaz_to_dome_xy(alt_deg: float, az_deg: float, img_w: int, img_h: int) -> Optional[Tuple[int, int]]:
        if alt_deg <= 0.0:
            return None
        cx = img_w * 0.5
        cy = img_h * 0.5
        r_max = min(img_w, img_h) * 0.45
        zd_frac = max(0.0, min(1.0, (90.0 - alt_deg) / 90.0))
        r = r_max * zd_frac
        # Standard Allsky dome orientation (North top, East left/right approximate exclusion zone)
        theta = math.radians(az_deg)
        x = int(round(cx - r * math.sin(theta)))
        y = int(round(cy - r * math.cos(theta)))
        return (x, y)


class FourRegimeExposureController:
    """
    4-Regime Dynamic Exposure, Gain & White Balance Controller for ZWO ASI676MC (IMX676).
    Uses subsampled sky-dome interquartile (25th-75th percentile) metering with dynamic
    Sun/Moon exclusion discs and star-core 99.95th percentile saturation guard.
    """

    def __init__(self, cfg: Dict[str, Any], img_w: int = 3552, img_h: int = 3552):
        self.cfg = cfg
        self.img_w = img_w
        self.img_h = img_h

        # Precompute 8x-subsampled circular sky mask (excludes trees/horizon below ~20 deg alt)
        self.step = 8
        ys = np.arange(0, img_h, self.step, dtype=np.float32)
        xs = np.arange(0, img_w, self.step, dtype=np.float32)
        self_xx, self_yy = np.meshgrid(xs, ys)
        self.grid_x = self_xx
        self.grid_y = self_yy
        cx = img_w * 0.5
        cy = img_h * 0.5
        sky_radius = min(img_w, img_h) * float(cfg.get("sky_metering_radius_frac", 0.38))
        self.base_sky_mask = ((self.grid_x - cx) ** 2 + (self.grid_y - cy) ** 2) <= (sky_radius ** 2)

        # Current state (initialized from current sky conditions on first step)
        self.exposure_us: int = int(cfg.get("initial_exposure_us", 500_000))
        self.gain: int = int(cfg.get("night_hcg_gain", 300))
        self.wb_r: int = int(cfg.get("wb_r", 55))
        self.wb_b: int = int(cfg.get("wb_b", 75))
        self.auto_wb: bool = False
        self.regime: str = "TWILIGHT"

    def initialize_for_ephemeris(self, eph: Dict[str, Any], legacy_overlay_path: Optional[Path] = None) -> None:
        """Seed initial exposure & gain from existing overlaydebug.txt if fresh, or ephemeris."""
        if legacy_overlay_path and legacy_overlay_path.exists():
            try:
                age_s = time.time() - legacy_overlay_path.stat().st_mtime
                if age_s < 600:
                    txt = legacy_overlay_path.read_text(encoding="utf-8", errors="ignore")
                    for line in txt.splitlines():
                        if line.startswith("AS_EXPOSURE_US"):
                            self.exposure_us = max(32, int(line.split()[-1]))
                        elif line.startswith("AS_GAIN"):
                            self.gain = max(0, int(line.split()[-1]))
                    logger.info(
                        "Seeded initial state from live overlaydebug.txt: exposure=%dus (%.3fs), gain=%d",
                        self.exposure_us,
                        self.exposure_us / 1e6,
                        self.gain,
                    )
                    return
            except Exception as e:
                logger.warning("Could not seed from overlaydebug.txt: %s", e)

        sun_alt = eph["sun_alt_deg"]
        if sun_alt > -0.83:
            self.regime = "DAY"
            self.exposure_us = 15_000
            self.gain = 0
        elif sun_alt > -6.0:
            self.regime = "TWILIGHT"
            self.exposure_us = 250_000
            self.gain = 180
        elif sun_alt > -12.0:
            self.regime = "TWILIGHT"
            self.exposure_us = 2_500_000
            self.gain = 300
        else:
            self.regime = "DARK_NIGHT"
            self.exposure_us = 15_000_000
            self.gain = 300

    def classify_regime(self, eph: Dict[str, Any]) -> Tuple[str, str]:
        """Returns (sky_regime, day_or_night_compat)."""
        sun_alt = eph["sun_alt_deg"]
        moon_alt = eph["moon_alt_deg"]
        moon_illum = eph["moon_illumination_pct"]
        night_boundary = float(self.cfg.get("sun_angle_night_deg", -6.0))

        day_or_night = "DAY" if sun_alt > night_boundary else "NIGHT"

        if sun_alt > -0.83:
            regime = "DAY"
        elif sun_alt > -12.0:
            regime = "TWILIGHT"
        else:
            if moon_alt > 2.0 and moon_illum >= 25.0:
                regime = "MOONLIT_NIGHT"
            else:
                regime = "DARK_NIGHT"
        return regime, day_or_night

    def meter_sky(self, frame_bgr: np.ndarray, eph: Dict[str, Any]) -> Dict[str, float]:
        """
        Computes sky background statistics over the unmasked sky dome, excluding
        a 250px radius around the Moon/Sun so bright lunar/solar discs never skew metering.
        """
        sub = frame_bgr[:: self.step, :: self.step]
        # Perceived luminance (green-weighted astronomical sky luminance)
        luma = (
            0.114 * sub[:, :, 0].astype(np.float32)
            + 0.587 * sub[:, :, 1].astype(np.float32)
            + 0.299 * sub[:, :, 2].astype(np.float32)
        ) / 255.0

        mask = self.base_sky_mask.copy()
        excl_r2 = float(self.cfg.get("celestial_exclusion_radius_px", 260)) ** 2

        for key in ("sun_xy", "moon_xy"):
            xy = eph.get(key)
            if xy is not None:
                mx, my = xy
                dist2 = (self.grid_x - mx) ** 2 + (self.grid_y - my) ** 2
                mask &= dist2 > excl_r2

        sky_pixels = luma[mask]
        if sky_pixels.size < 100:
            sky_pixels = luma.ravel()

        p25, p50, p75, p9995 = np.percentile(sky_pixels, [25.0, 50.0, 75.0, 99.95])
        iqr_mask = (sky_pixels >= p25) & (sky_pixels <= p75)
        iqr_mean = float(np.mean(sky_pixels[iqr_mask])) if np.any(iqr_mask) else float(p50)

        return {
            "sky_mean_iqr": round(iqr_mean, 5),
            "sky_median": round(float(p50), 5),
            "sky_p25": round(float(p25), 5),
            "sky_p75": round(float(p75), 5),
            "sky_p9995": round(float(p9995), 5),
        }

    def update(self, frame_bgr: np.ndarray, eph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Meters the captured frame and computes the optimal (exposure_us, gain, wb_r, wb_b)
        for the NEXT frame.
        """
        regime, day_or_night = self.classify_regime(eph)
        self.regime = regime
        metrics = self.meter_sky(frame_bgr, eph)
        measured = max(0.001, metrics["sky_mean_iqr"])
        sun_alt = eph["sun_alt_deg"]

        min_exp_us = int(self.cfg.get("min_exposure_us", 32))
        day_max_exp_us = int(self.cfg.get("day_max_exposure_us", 2_000_000))
        night_max_exp_us = int(self.cfg.get("night_max_exposure_us", 30_000_000))
        hcg_gain = int(self.cfg.get("night_hcg_gain", 300))
        moon_gain = int(self.cfg.get("moonlit_night_gain", 210))
        max_night_gain = int(self.cfg.get("night_max_gain", 450))

        # Determine target mean and gain regime
        if regime == "DAY":
            target_mean = float(self.cfg.get("day_target_mean", 0.46))
            max_exp_us = day_max_exp_us
            target_gain = 0
            self.auto_wb = bool(self.cfg.get("day_auto_wb", True))
            kp = 0.60
        elif regime == "TWILIGHT":
            # Smoothly interpolate target mean from 0.46 (at -0.83 deg) to 0.20 (at -12.0 deg)
            twilight_frac = max(0.0, min(1.0, (-0.83 - sun_alt) / (-0.83 - (-12.0))))
            day_m = float(self.cfg.get("day_target_mean", 0.46))
            night_m = float(self.cfg.get("night_target_mean", 0.20))
            target_mean = day_m + twilight_frac * (night_m - day_m)
            max_exp_us = night_max_exp_us
            self.auto_wb = False
            kp = 0.70

            # Smooth gain ramp during twilight: stay at Gain 0 while exposure < 100ms,
            # cross into HCG (180..300) as exposure climbs from 100ms to 5s
            if self.exposure_us < 100_000:
                target_gain = 0
            elif self.exposure_us < 1_000_000:
                target_gain = 180
            else:
                target_gain = hcg_gain
        elif regime == "MOONLIT_NIGHT":
            target_mean = float(self.cfg.get("moonlit_target_mean", 0.22))
            max_exp_us = int(self.cfg.get("moonlit_max_exposure_us", 20_000_000))
            target_gain = moon_gain
            self.auto_wb = False
            kp = 0.55
        else:  # DARK_NIGHT
            target_mean = float(self.cfg.get("night_target_mean", 0.20))
            max_exp_us = night_max_exp_us
            target_gain = hcg_gain
            self.auto_wb = False
            kp = 0.55

        # Account for any gain change between current frame and next frame
        # ZWO ASI gain is in 0.1 dB units: 200 units = 20 dB = 10x brightness
        gain_delta_db = (target_gain - self.gain) * 0.1
        gain_brightness_factor = 10.0 ** (gain_delta_db / 20.0)

        # Damped proportional-log exposure update
        ratio = target_mean / (measured * gain_brightness_factor)
        # Clamp per-frame step ratio to [0.25x, 3.2x] so sudden lightning/headlights don't blind the loop
        damped_scale = max(0.25, min(3.2, ratio ** kp))

        # Star-core saturation guard in DARK_NIGHT: if 99.95th percentile is saturated (>0.985)
        # and background is already reasonably bright (>0.22), nudge exposure down slightly
        if regime in ("DARK_NIGHT", "MOONLIT_NIGHT") and metrics["sky_p9995"] > 0.985 and measured > 0.22:
            damped_scale = min(damped_scale, 0.92)

        next_exp_us = int(round(self.exposure_us * damped_scale))
        next_exp_us = max(min_exp_us, min(max_exp_us, next_exp_us))

        # If DARK_NIGHT hits max exposure and sky is still very dark (<0.12), allow gentle gain boost
        if regime == "DARK_NIGHT" and next_exp_us >= max_exp_us and measured < 0.12:
            deficit = min(1.8, target_mean / max(0.02, measured))
            extra_gain = int(round(200.0 * math.log10(deficit)))
            target_gain = min(max_night_gain, hcg_gain + extra_gain)

        self.exposure_us = next_exp_us
        self.gain = target_gain

        return {
            "regime": regime,
            "day_or_night": day_or_night,
            "next_exposure_us": self.exposure_us,
            "next_gain": self.gain,
            "target_mean": round(target_mean, 4),
            **metrics,
        }


# ---------------------------------------------------------------------------
# Background Archiver & FTP Uploader Worker
# ---------------------------------------------------------------------------
class ArchiveAndUploadWorker(threading.Thread):
    """
    Asynchronous background worker so SD-card archival, thumbnail generation,
    and Windows Server FTP upload never block camera shutter timing.
    """

    def __init__(self, cfg: Dict[str, Any], allsky_home: Path):
        super().__init__(daemon=True, name="ArchiveUploadWorker")
        self.cfg = cfg
        self.allsky_home = allsky_home
        self.images_root = allsky_home / "images"
        self.q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=4)
        self.last_prune_day: str = ""
        self.ftp_cfg = self._load_ftp_config()

    def _load_ftp_config(self) -> Dict[str, Any]:
        env_path = self.allsky_home / "env.json"
        settings_path = self.allsky_home / "config" / "settings.json"
        node_id = str(self.cfg.get("node_id") or "bluffsky-pi1")
        default_remote_fname = (
            "image.jpg"
            if node_id in ("bluffsky-pi1", "cam1")
            else f"image_{node_id}.jpg"
        )
        res = {
            "enabled": True,
            "host": "192.168.1.10",
            "port": 21,
            "user": "test",
            "password": "test",
            "remote_dir": "allsky",
            "remote_filename": str(self.cfg.get("ftp_remote_filename") or default_remote_fname),
            "days_to_keep": 10,
            "thumb_w": 100,
            "thumb_h": 75,
        }
        try:
            if env_path.exists():
                env_data = json.loads(env_path.read_text(encoding="utf-8"))
                res["host"] = env_data.get("REMOTESERVER_HOST", res["host"])
                res["port"] = int(env_data.get("REMOTESERVER_PORT") or 21)
                res["user"] = env_data.get("REMOTESERVER_USER", res["user"])
                res["password"] = env_data.get("REMOTESERVER_PASSWORD", res["password"])
        except Exception as e:
            logger.warning("Could not read env.json: %s", e)

        try:
            if settings_path.exists():
                s_data = json.loads(settings_path.read_text(encoding="utf-8"))
                res["enabled"] = bool(s_data.get("useremoteserver", True))
                res["remote_dir"] = s_data.get("remoteserverimagedir", "allsky")
                res["days_to_keep"] = int(s_data.get("daystokeep", 10))
                res["thumb_w"] = int(s_data.get("thumbnailsizex", 100))
                res["thumb_h"] = int(s_data.get("thumbnailsizey", 75))
        except Exception as e:
            logger.warning("Could not read settings.json: %s", e)

        if "ftp_enabled" in self.cfg:
            res["enabled"] = bool(self.cfg["ftp_enabled"])
        if self.cfg.get("ftp_host"):
            res["host"] = str(self.cfg["ftp_host"])
        if self.cfg.get("ftp_remote_dir"):
            res["remote_dir"] = str(self.cfg["ftp_remote_dir"])

        return res

    def enqueue(self, item: Dict[str, Any]) -> None:
        try:
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put_nowait(item)
        except Exception as e:
            logger.warning("Enqueue archive task failed: %s", e)

    def run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                break
            try:
                self._process_item(item)
            except Exception as e:
                logger.error("Archive/Upload error: %s", e)
            finally:
                self.q.task_done()

    def _process_item(self, item: Dict[str, Any]) -> None:
        jpeg_bytes: bytes = item["jpeg_bytes"]
        frame_bgr: np.ndarray = item["frame_bgr"]
        date_folder: str = item["date_folder"]
        image_filename: str = item["image_filename"]

        # 1. Write full archive image & thumbnail to /home/pi/allsky/images/<YYYYMMDD>/
        date_dir = self.images_root / date_folder
        thumb_dir = date_dir / "thumbnails"
        date_dir.mkdir(parents=True, exist_ok=True)
        thumb_dir.mkdir(parents=True, exist_ok=True)

        archive_path = date_dir / image_filename
        archive_path.write_bytes(jpeg_bytes)

        # Fast in-memory thumbnail generation
        tw = self.ftp_cfg["thumb_w"]
        th = self.ftp_cfg["thumb_h"]
        thumb_bgr = cv2.resize(frame_bgr, (tw, th), interpolation=cv2.INTER_AREA)
        ok, thumb_buf = cv2.imencode(".jpg", thumb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            (thumb_dir / image_filename).write_bytes(thumb_buf.tobytes())

        # 2. Upload to Windows workstation FTP server (feeds AllSkyMonitorService)
        if self.ftp_cfg["enabled"] and self.ftp_cfg["host"]:
            self._upload_ftp(jpeg_bytes)

        # 3. Once per day, prune old archive folders beyond days_to_keep
        if date_folder != self.last_prune_day:
            self.last_prune_day = date_folder
            self._prune_old_archives(self.ftp_cfg["days_to_keep"])

    def _upload_ftp(self, jpeg_bytes: bytes) -> None:
        host = self.ftp_cfg["host"]
        port = self.ftp_cfg["port"]
        user = self.ftp_cfg["user"]
        passwd = self.ftp_cfg["password"]
        remote_dir = self.ftp_cfg["remote_dir"].strip("/")
        target_fname = self.ftp_cfg.get("remote_filename") or "image.jpg"
        tmp_name = f"SaveImage-server-{os.getpid()}-{target_fname}"

        for use_tls in (True, False):
            try:
                if use_tls:
                    ftp = ftplib.FTP_TLS()
                else:
                    ftp = ftplib.FTP()
                ftp.connect(host, port, timeout=8)
                ftp.login(user, passwd)
                if use_tls and hasattr(ftp, "prot_p"):
                    ftp.prot_p()
                if remote_dir:
                    ftp.cwd(remote_dir)
                bio = io.BytesIO(jpeg_bytes)
                ftp.storbinary(f"STOR {tmp_name}", bio)
                try:
                    ftp.delete(target_fname)
                except Exception:
                    pass
                ftp.rename(tmp_name, target_fname)
                ftp.quit()
                return
            except Exception:
                if not use_tls:
                    logger.debug("FTP upload to %s:%d skipped/failed", host, port)

    def _prune_old_archives(self, days_to_keep: int) -> None:
        if days_to_keep <= 0 or not self.images_root.exists():
            return
        cutoff = (datetime.datetime.now() - timedelta(days=days_to_keep)).strftime("%Y%m%d")
        for sub in sorted(self.images_root.iterdir()):
            if sub.is_dir() and len(sub.name) == 8 and sub.name.isdigit():
                if sub.name < cutoff:
                    logger.info("Pruning expired archive directory: %s (cutoff %s)", sub, cutoff)
                    shutil.rmtree(sub, ignore_errors=True)


# ---------------------------------------------------------------------------
# Lightweight Control & Telemetry HTTP API (Port 8096)
# ---------------------------------------------------------------------------
class NodeTelemetryServer:
    """Serves live microsecond telemetry (/api/status) and runtime config (/api/config) on :8096."""

    def __init__(self, daemon_ref: "BluffSkyCaptureDaemon", port: int = 8096):
        self.daemon_ref = daemon_ref
        self.port = port
        self.httpd: Optional[http.server.ThreadingHTTPServer] = None

    def start(self) -> None:
        daemon_ref = self.daemon_ref

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
                body = json.dumps(payload, indent=2).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def _resolve_static(self, req_path: str) -> Optional[Path]:
                clean = req_path.split("?", 1)[0]
                if clean.startswith("/current/tmp/"):
                    return daemon_ref.tmp_dir / Path(clean[len("/current/tmp/") :]).name
                if clean == "/config/status.json":
                    return daemon_ref.config_dir / "status.json"
                if clean.startswith("/images/"):
                    rel = clean[len("/images/") :].strip("/")
                    if ".." not in rel:
                        return daemon_ref.allsky_home / "images" / rel
                return None

            def do_HEAD(self) -> None:
                fp = self._resolve_static(self.path)
                if fp and fp.is_file():
                    st = fp.stat()
                    self.send_response(200)
                    self.send_header("Content-Length", str(st.st_size))
                    self.send_header("ETag", f'"{st.st_mtime_ns}-{st.st_size}"')
                    self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_GET(self) -> None:
                if self.path.startswith("/api/status") or self.path == "/":
                    self._send_json(200, daemon_ref.get_live_status())
                elif self.path.startswith("/api/adsb") or self.path.startswith("/tar1090/data/aircraft.json"):
                    self._send_json(200, daemon_ref.get_local_adsb_json())
                elif self.path.startswith("/api/config"):
                    self._send_json(200, daemon_ref.cfg)
                else:
                    fp = self._resolve_static(self.path)
                    if fp and fp.is_dir():
                        items = sorted(p.name for p in fp.glob("image-*.jpg"))
                        links = "\n".join(f'<a href="{n}">{n}</a>' for n in items)
                        body = f"<html><body>{links}</body></html>".encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    elif fp and fp.is_file():
                        data = fp.read_bytes()
                        st = fp.stat()
                        ctype = "image/jpeg" if fp.suffix.lower() in (".jpg", ".jpeg") else ("application/json" if fp.suffix.lower() == ".json" else "text/plain; charset=utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", ctype)
                        self.send_header("Content-Length", str(len(data)))
                        self.send_header("ETag", f'"{st.st_mtime_ns}-{st.st_size}"')
                        self.end_headers()
                        self.wfile.write(data)
                    else:
                        self._send_json(404, {"error": "Not found"})

            def do_POST(self) -> None:
                if self.path.startswith("/api/config"):
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    try:
                        updates = json.loads(raw.decode("utf-8"))
                        updated = daemon_ref.update_runtime_config(updates)
                        self._send_json(200, {"ok": True, "config": updated})
                    except Exception as e:
                        self._send_json(400, {"ok": False, "error": str(e)})
                else:
                    self._send_json(404, {"error": "Not found"})

        self.httpd = http.server.ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        t = threading.Thread(target=self.httpd.serve_forever, daemon=True, name="TelemetryHTTP")
        t.start()
        logger.info("Started BluffSky Option A Telemetry & Stereo Sync API on port %d", self.port)


# ---------------------------------------------------------------------------
# Main Option A Capture Daemon
# ---------------------------------------------------------------------------
class BluffSkyCaptureDaemon:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.cfg = self._load_config(config_path)
        self.allsky_home = Path(self.cfg.get("allsky_home", "/home/pi/allsky"))
        self.tmp_dir = self.allsky_home / "tmp"
        self.config_dir = self.allsky_home / "config"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.running = True
        self.frame_count = 0
        self.started_at = time.time()
        self.latest_meta: Dict[str, Any] = {}

        # Ephemeris & Exposure Controller
        self.ephemeris = EphemerisTracker(
            lat_deg=float(self.cfg.get("latitude_deg", 45.0000)),
            lon_deg=float(self.cfg.get("longitude_deg", -100.0000)),
            elev_m=float(self.cfg.get("elevation_m", 39.6)),
        )
        self.controller = FourRegimeExposureController(self.cfg)
        self.archiver = ArchiveAndUploadWorker(self.cfg, self.allsky_home)
        self.telemetry_server = NodeTelemetryServer(
            self, port=int(self.cfg.get("telemetry_port", 8096))
        )

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        defaults = {
            "node_id": "bluffsky-pi1",
            "allsky_home": "/home/pi/allsky",
            "asi_lib_path": "/home/pi/allsky/bluffsky_node/libASICamera2.so",
            "camera_index": 0,
            "latitude_deg": 45.0000,
            "longitude_deg": -100.0000,
            "elevation_m": 39.6,
            "sun_angle_night_deg": -6.0,
            "jpeg_quality": 95,
            "usb_bandwidth": 50,
            "auto_usb": True,
            "flip": 0,
            "wb_r": 55,
            "wb_b": 75,
            "day_auto_wb": True,
            "min_exposure_us": 32,
            "day_max_exposure_us": 2_000_000,
            "night_max_exposure_us": 25_000_000,
            "moonlit_max_exposure_us": 15_000_000,
            "day_target_mean": 0.46,
            "night_target_mean": 0.20,
            "moonlit_target_mean": 0.22,
            "night_hcg_gain": 300,
            "moonlit_night_gain": 210,
            "night_max_gain": 450,
            "sky_metering_radius_frac": 0.38,
            "celestial_exclusion_radius_px": 260,
            "day_period_ms": 15_000,
            "twilight_min_period_ms": 5_000,
            "night_interframe_delay_ms": 50,
            "sync_mode": "auto",        # "auto", "lockstep", "staggered", "freerun"
            "sync_grid_ms": 0,          # >0 enforces strict UTC modulo shutter lock across Pis
            "sync_offset_ms": 0,        # 0 for lockstep stereo, sync_grid_ms//2 for staggered
            "burn_overlay_text": False, # Keep science frames 100% clean
            "adsb_enabled": False,      # Optional co-located RTL-SDR ADS-B receiver on this Pi (off by default)
            "adsb_json_path": "/run/readsb/aircraft.json",
            "telemetry_port": 8096,
        }
        if path.exists():
            try:
                user_cfg = json.loads(path.read_text(encoding="utf-8"))
                defaults.update(user_cfg)
            except Exception as e:
                logger.warning("Failed reading %s, using defaults: %s", path, e)
        return defaults

    def update_runtime_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in updates.items():
            if k in self.cfg:
                self.cfg[k] = v
        try:
            self.config_path.write_text(json.dumps(self.cfg, indent=4), encoding="utf-8")
        except Exception as e:
            logger.warning("Could not persist runtime config: %s", e)
        logger.info("Updated runtime config: %s", updates)
        return self.cfg

    def get_local_adsb_json(self) -> Dict[str, Any]:
        """
        Optional co-located ADS-B receiver endpoint (/api/adsb).
        When 'adsb_enabled' is true in node_config.json, reads the live RAM aircraft.json
        written by readsb / dump1090-fa on this Raspberry Pi with zero SD card I/O.
        """
        if not self.cfg.get("adsb_enabled", False):
            return {"enabled": False, "node_id": self.cfg.get("node_id", "bluffsky-pi1"), "aircraft": [], "messages": 0}

        candidate_paths = [
            Path(str(self.cfg.get("adsb_json_path", "/run/readsb/aircraft.json"))),
            Path("/run/readsb/aircraft.json"),
            Path("/run/dump1090-fa/aircraft.json"),
            Path("/run/dump1090-mutability/aircraft.json"),
            Path("/run/adsbexchange-feed/aircraft.json"),
        ]
        for p in candidate_paths:
            if p.is_file():
                try:
                    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                    data["enabled"] = True
                    data["node_id"] = self.cfg.get("node_id", "bluffsky-pi1")
                    return data
                except Exception:
                    continue
        return {"enabled": True, "node_id": self.cfg.get("node_id", "bluffsky-pi1"), "aircraft": [], "messages": 0}

    def get_live_status(self) -> Dict[str, Any]:
        adsb_enabled = bool(self.cfg.get("adsb_enabled", False))
        adsb_count = len(self.get_local_adsb_json().get("aircraft", [])) if adsb_enabled else 0
        return {
            "status": "Running" if self.running else "Stopped",
            "engine": "BluffSky-OptionA-v1.0",
            "node_id": self.cfg.get("node_id", "bluffsky-pi1"),
            "hostname": socket.gethostname(),
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "frames_captured": self.frame_count,
            "current_regime": self.controller.regime,
            "next_exposure_us": self.controller.exposure_us,
            "next_gain": self.controller.gain,
            "sync_mode": self.cfg.get("sync_mode", "auto"),
            "sync_grid_ms": self.cfg.get("sync_grid_ms", 0),
            "sync_offset_ms": self.cfg.get("sync_offset_ms", 0),
            "adsb_enabled": adsb_enabled,
            "adsb_aircraft_count": adsb_count,
            "latest_frame": self.latest_meta,
        }

    def _wait_for_shutter_window(self, exposure_us: int, regime: str) -> None:
        """
        Deterministic UTC modulo shutter synchronization.
        - If sync_grid_ms > 0 (stereo lockstep or staggered mode), aligns shutter start
          to t_utc_ms % sync_grid_ms == sync_offset_ms (<1 ms jitter).
        - In 'auto' single-camera mode:
          * DAY: aligns to 15.0s UTC boundaries (e.g. :00, :15, :30, :45).
          * TWILIGHT: aligns to adaptive 5.0s or 10.0s UTC boundaries (eliminating the
            legacy 34.5-second twilight dead-time bug!).
          * DARK_NIGHT / MOONLIT_NIGHT: back-to-back exposures with 50ms inter-frame gap
            (or locked to 1.0s UTC boundary for deterministic sub-millisecond start time).
        """
        sync_grid_ms = int(self.cfg.get("sync_grid_ms", 0))
        sync_offset_ms = int(self.cfg.get("sync_offset_ms", 0))
        sync_mode = str(self.cfg.get("sync_mode", "auto")).lower()

        exp_ms = exposure_us / 1000.0
        if sync_grid_ms > 0 and sync_mode in ("lockstep", "staggered"):
            grid_ms = sync_grid_ms
            offset_ms = (grid_ms // 2) if sync_mode == "staggered" and sync_offset_ms == 0 else sync_offset_ms
        elif regime == "DAY":
            grid_ms = int(self.cfg.get("day_period_ms", 15_000))
            offset_ms = 0
        elif regime == "TWILIGHT":
            min_p = int(self.cfg.get("twilight_min_period_ms", 5_000))
            grid_ms = max(min_p, int(math.ceil((exp_ms + 400.0) / 1000.0) * 1000))
            offset_ms = 0
        else:
            # Night mode: if sync_grid_ms == 0, align to next 500ms UTC tick after night_interframe_delay_ms
            delay_ms = int(self.cfg.get("night_interframe_delay_ms", 50))
            time.sleep(delay_ms / 1000.0)
            grid_ms = 500
            offset_ms = 0

        now_ms = time.time_ns() // 1_000_000
        rem = (now_ms - offset_ms) % grid_ms
        wait_ms = (grid_ms - rem) % grid_ms
        if wait_ms < 15 and grid_ms >= 5000:
            wait_ms += grid_ms

        if wait_ms > 0 and self.running:
            # Sleep coarse then fine for <1 ms trigger accuracy
            if wait_ms > 20:
                time.sleep((wait_ms - 10) / 1000.0)
            target_ns = (now_ms + wait_ms) * 1_000_000
            while time.time_ns() < target_ns and self.running:
                time.sleep(0.0002)

    def _encode_jpeg(self, frame_bgr: np.ndarray, quality: int = 95) -> bytes:
        if HAS_SIMPLEJPEG:
            try:
                return simplejpeg.encode_jpeg(frame_bgr, quality=quality, colorspace="BGR")
            except Exception:
                pass
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()

    def _write_atomic(self, target_path: Path, data: bytes) -> None:
        tmp_path = target_path.parent / f".{target_path.name}.tmp"
        tmp_path.write_bytes(data)
        os.replace(tmp_path, target_path)

    def _write_status_json(self, status_text: str = "Running", extra: Optional[Dict[str, Any]] = None) -> None:
        status_file = self.config_dir / "status.json"
        payload = {
            "status": status_text,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "engine": "BluffSky Option A",
        }
        if extra:
            payload.update(extra)
        try:
            self._write_atomic(status_file, json.dumps(payload, indent=4).encode("utf-8"))
        except Exception as e:
            logger.warning("Could not write status.json: %s", e)

    @staticmethod
    def _format_exposure_str(exposure_us: int) -> str:
        if exposure_us >= 1_000_000:
            s = exposure_us / 1_000_000.0
            return f"{s:.2f} sec ({exposure_us / 1000.0:.1f} ms)"
        ms = exposure_us / 1000.0
        return f"{ms:.2f} ms ({exposure_us / 1_000_000.0:.2f} sec)"

    def _write_telemetry_files(self, meta: Dict[str, Any]) -> None:
        # 1. Write /home/pi/allsky/tmp/frame_meta.json
        meta_path = self.tmp_dir / "frame_meta.json"
        self._write_atomic(meta_path, json.dumps(meta, indent=2).encode("utf-8"))

        # 2. Write /home/pi/allsky/tmp/overlaydebug.txt (100% compatible with detector.py & guard)
        exp_us = meta["exposure_us"]
        s_exp = self._format_exposure_str(exp_us)
        temp_c = int(round(meta["sensor_temp_c"]))
        temp_f = int(round(meta["sensor_temp_f"]))
        lines = [
            f"ALLSKY_HOME                                       {self.allsky_home}",
            f"ALLSKY_TMP                                        {self.tmp_dir}",
            f"AS_CAMERA_MODEL                                   ASI676MC",
            f"AS_CAMERA_TYPE                                    ZWO",
            f"AS_DATE                                           {meta['as_date']}",
            f"AS_TIME                                           {meta['as_time']}",
            f"DATE_NAME                                         {meta['date_folder']}",
            f"DAY_OR_NIGHT                                      {meta['day_or_night']}",
            f"AS_SKY_REGIME                                     {meta['sky_regime']}",
            f"CURRENT_IMAGE                                     {self.tmp_dir / meta['image_filename']}",
            f"AS_EXPOSURE_US                                    {exp_us}",
            f"AS_EXPOSURE_START_US                              {meta['exposure_start_unix_us']}",
            f"AS_EXPOSURE_END_US                                {meta['exposure_end_unix_us']}",
            f"AS_sEXPOSURE                                      {s_exp}",
            f"AS_GAIN                                           {meta['gain']}",
            f"AS_TEMPERATURE_C                                  {temp_c}",
            f"AS_TEMPERATURE_F                                  {temp_f}",
            f"AS_MEAN                                           {meta['sky_mean_iqr']:.6f}",
            f"AS_SUN_ELEVATION                                  {meta['sun_elevation_deg']}",
            f"AS_MOON_ELEVATION                                 {meta['moon_elevation_deg']}°",
            f"AS_MOON_ILLUMINATION                              {meta['moon_illumination_pct']}",
            f"AS_WBR                                            {meta['wb_r']}",
            f"AS_WBB                                            {meta['wb_b']}",
            f"AS_CAPTURE_ENGINE                                 BluffSky-OptionA-v1.0",
        ]
        overlay_path = self.tmp_dir / "overlaydebug.txt"
        self._write_atomic(overlay_path, ("\n".join(lines) + "\n").encode("utf-8"))

    def run(self) -> None:
        def _sig_handler(signum: int, frame: Any) -> None:
            logger.info("Received signal %d, shutting down cleanly...", signum)
            self.running = False

        signal.signal(signal.SIGTERM, _sig_handler)
        signal.signal(signal.SIGINT, _sig_handler)

        self.archiver.start()
        self.telemetry_server.start()

        now_utc = datetime.datetime.now(timezone.utc)
        init_eph = self.ephemeris.compute(now_utc)
        self.controller.initialize_for_ephemeris(
            init_eph, legacy_overlay_path=self.tmp_dir / "overlaydebug.txt"
        )

        driver = ASICameraDriver(
            lib_path=self.cfg["asi_lib_path"],
            cam_index=int(self.cfg.get("camera_index", 0)),
        )
        driver.open(
            binning=1,
            image_type=ASI_IMG_RGB24,
            usb_bandwidth=int(self.cfg.get("usb_bandwidth", 50)),
            auto_usb=bool(self.cfg.get("auto_usb", True)),
            flip=int(self.cfg.get("flip", 0)),
        )

        self._write_status_json("Running")
        prior_temp_img: Optional[Path] = None

        try:
            while self.running:
                now_utc = datetime.datetime.now(timezone.utc)
                eph = self.ephemeris.compute(now_utc, driver.width, driver.height)
                regime, day_or_night = self.controller.classify_regime(eph)

                cur_exp_us = self.controller.exposure_us
                cur_gain = self.controller.gain
                cur_wb_r = self.controller.wb_r
                cur_wb_b = self.controller.wb_b
                cur_auto_wb = self.controller.auto_wb

                # Wait for deterministic UTC shutter window
                self._wait_for_shutter_window(cur_exp_us, regime)
                if not self.running:
                    break

                # Capture frame
                frame_bgr, exp_start_us, exp_end_us = driver.capture_frame(
                    exposure_us=cur_exp_us,
                    gain=cur_gain,
                    wb_r=cur_wb_r,
                    wb_b=cur_wb_b,
                    auto_wb=cur_auto_wb,
                )
                temp_c = driver.get_temperature_c()
                temp_f = round(temp_c * 9.0 / 5.0 + 32.0, 1)

                # Local timestamp at exposure start (matches Allsky STARTING EXPOSURE filename convention)
                dt_local = datetime.datetime.fromtimestamp(exp_start_us / 1_000_000.0)
                dt_utc_end = datetime.datetime.fromtimestamp(exp_end_us / 1_000_000.0, tz=timezone.utc)
                dt_utc_start = datetime.datetime.fromtimestamp(exp_start_us / 1_000_000.0, tz=timezone.utc)

                as_date = dt_local.strftime("%Y%m%d")
                as_time = dt_local.strftime("%H%M%S")
                if day_or_night == "NIGHT":
                    date_folder = (dt_local - timedelta(hours=12)).strftime("%Y%m%d")
                else:
                    date_folder = as_date
                image_filename = f"image-{as_date}{as_time}.jpg"

                # Update 4-regime exposure & gain controller for the NEXT frame
                ctrl_out = self.controller.update(frame_bgr, eph)

                # Encode pristine science JPEG in RAM
                jpeg_bytes = self._encode_jpeg(frame_bgr, quality=int(self.cfg.get("jpeg_quality", 95)))

                # Write timestamped image in tmpfs + atomic rename to /home/pi/allsky/tmp/image.jpg
                temp_named_path = self.tmp_dir / image_filename
                temp_named_path.write_bytes(jpeg_bytes)
                self._write_atomic(self.tmp_dir / "image.jpg", jpeg_bytes)
                if prior_temp_img and prior_temp_img != temp_named_path and prior_temp_img.exists():
                    try:
                        prior_temp_img.unlink()
                    except Exception:
                        pass
                prior_temp_img = temp_named_path

                self.frame_count += 1
                meta = {
                    "node_id": self.cfg.get("node_id", "bluffsky-pi1"),
                    "hostname": socket.gethostname(),
                    "camera_model": "ZWO ASI676MC",
                    "frame_id": self.frame_count,
                    "image_filename": image_filename,
                    "as_date": as_date,
                    "as_time": as_time,
                    "date_folder": date_folder,
                    "exposure_start_unix_us": exp_start_us,
                    "exposure_end_unix_us": exp_end_us,
                    "exposure_mid_unix_us": (exp_start_us + exp_end_us) // 2,
                    "exposure_start_utc": dt_utc_start.isoformat(),
                    "exposure_end_utc": dt_utc_end.isoformat(),
                    "exposure_us": cur_exp_us,
                    "exposure_ms": round(cur_exp_us / 1000.0, 2),
                    "gain": cur_gain,
                    "wb_r": cur_wb_r,
                    "wb_b": cur_wb_b,
                    "sensor_temp_c": temp_c,
                    "sensor_temp_f": temp_f,
                    "sky_regime": regime,
                    "day_or_night": day_or_night,
                    "sky_mean_iqr": ctrl_out["sky_mean_iqr"],
                    "sky_median": ctrl_out["sky_median"],
                    "sky_p9995": ctrl_out["sky_p9995"],
                    "target_mean": ctrl_out["target_mean"],
                    "next_exposure_us": ctrl_out["next_exposure_us"],
                    "next_gain": ctrl_out["next_gain"],
                    "sun_elevation_deg": eph["sun_alt_deg"],
                    "moon_elevation_deg": eph["moon_alt_deg"],
                    "moon_illumination_pct": eph["moon_illumination_pct"],
                    "sync_mode": self.cfg.get("sync_mode", "auto"),
                    "sync_grid_ms": self.cfg.get("sync_grid_ms", 0),
                    "sync_offset_ms": self.cfg.get("sync_offset_ms", 0),
                    "capture_engine": "BluffSky-OptionA-v1.0",
                }
                self.latest_meta = meta

                # Write telemetry files & status.json in RAM
                self._write_telemetry_files(meta)
                self._write_status_json("Running", {"regime": regime, "frame_id": self.frame_count})

                # Enqueue non-blocking SD archive & FTP upload
                self.archiver.enqueue(
                    {
                        "jpeg_bytes": jpeg_bytes,
                        "frame_bgr": frame_bgr,
                        "date_folder": date_folder,
                        "image_filename": image_filename,
                    }
                )

                logger.info(
                    "Frame #%d [%s] %s | exp=%.3fs gain=%d temp=%.1fC | sky_iqr=%.3f (target=%.3f) -> next_exp=%.3fs next_gain=%d",
                    self.frame_count,
                    regime,
                    image_filename,
                    cur_exp_us / 1e6,
                    cur_gain,
                    temp_c,
                    ctrl_out["sky_mean_iqr"],
                    ctrl_out["target_mean"],
                    ctrl_out["next_exposure_us"] / 1e6,
                    ctrl_out["next_gain"],
                )

        finally:
            driver.close()
            self._write_status_json("Stopped")
            logger.info("BluffSky Option A Capture Daemon stopped cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="BluffSky Option A Allsky Capture Daemon")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/pi/allsky/bluffsky_node/node_config.json",
        help="Path to node_config.json",
    )
    args = parser.parse_args()
    daemon = BluffSkyCaptureDaemon(Path(args.config))
    daemon.run()


if __name__ == "__main__":
    main()
