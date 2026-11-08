#!/usr/bin/env python3
"""
BluffSky All-Sky Observatory — 1-Command Setup & Configuration Wizard
=====================================================================
Generates or updates anomaly_detector/config.json from config.example.json
for 1..N Raspberry Pi camera nodes without touching existing user data.

Usage:
  # Interactive wizard:
  python setup_observatory.py

  # Non-interactive CLI setup for 1..N cameras:
  python setup_observatory.py --camera-ips 192.168.1.100,192.168.1.101 --lat 45.0000 --lon -100.0000 --cpu-only
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DET_DIR = ROOT / "anomaly_detector"
EXAMPLE_CFG = DET_DIR / "config.example.json"
TARGET_CFG = DET_DIR / "config.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="BluffSky Observatory 1..N Camera Setup Wizard")
    parser.add_argument("--camera-ips", type=str, default="", help="Comma-separated Raspberry Pi IPs (e.g. 192.168.1.50,192.168.1.51)")
    parser.add_argument("--lat", type=float, default=None, help="Observatory Latitude (decimal degrees)")
    parser.add_argument("--lon", type=float, default=None, help="Observatory Longitude (decimal degrees)")
    parser.add_argument("--alt-ft", type=float, default=None, help="Observatory Elevation in feet")
    parser.add_argument("--discord-webhook", type=str, default="", help="Optional Discord Webhook URL for #all-sky alerts")
    parser.add_argument("--cpu-only", action="store_true", help="Disable local VLM and daytime YOLO for Tier-1 CPU-only servers")
    parser.add_argument(
        "--enable-pi-adsb",
        action="store_true",
        help="Optional: Enable co-located 1090 MHz ADS-B SDR receiver polling from all 1..N Camera Pis (default: off)",
    )
    parser.add_argument(
        "--adsb-host",
        type=str,
        default=None,
        help="Optional: IP/hostname of a dedicated standalone ADS-B receiver Pi (or 'none' to disable)",
    )
    args = parser.parse_args()

    if not TARGET_CFG.exists():
        shutil.copyfile(EXAMPLE_CFG, TARGET_CFG)
        print(f"[+] Created {TARGET_CFG} from {EXAMPLE_CFG.name}")
    else:
        print(f"[*] Updating existing {TARGET_CFG}")

    cfg = json.loads(TARGET_CFG.read_text(encoding="utf-8"))

    ips = [ip.strip() for ip in args.camera_ips.split(",") if ip.strip()]
    if ips:
        cameras = []
        for idx, ip in enumerate(ips, start=1):
            base_url = ip if ip.startswith("http") else f"http://{ip}"
            cid = f"cam{idx}"
            nid = f"bluffsky-pi{idx}"
            cameras.append(
                {
                    "id": cid,
                    "node_id": nid,
                    "name": f"Allsky Camera #{idx} ({ip})",
                    "enabled": True,
                    "allsky_base_url": base_url,
                    "live_image_url": f"{base_url}/current/tmp/image.jpg",
                    "overlay_debug_url": f"{base_url}/current/tmp/overlaydebug.txt",
                    "telemetry_api_url": f"{base_url}:8096/api/status",
                    "adsb_enabled": bool(args.enable_pi_adsb),
                    "adsb_api_url": f"{base_url}:8096/api/adsb",
                    "camera_lat": args.lat if args.lat is not None else cfg.get("camera_lat", 45.0),
                    "camera_lon": args.lon if args.lon is not None else cfg.get("camera_lon", -100.0),
                    "camera_alt_ft": args.alt_ft if args.alt_ft is not None else cfg.get("camera_alt_ft", 130.0),
                    "east_offset_m": 0.0 if idx == 1 else 15.0 * (idx - 1),
                    "north_offset_m": 0.0,
                    "up_offset_m": 0.0,
                }
            )
        cfg["cameras"] = cameras
        cfg["allsky_base_url"] = cameras[0]["allsky_base_url"]
        cfg["live_image_url"] = cameras[0]["live_image_url"]
        cfg["overlay_debug_url"] = cameras[0]["overlay_debug_url"]
    elif args.enable_pi_adsb:
        for cam in cfg.get("cameras", []):
            cam["adsb_enabled"] = True

    if args.adsb_host is not None:
        h_clean = args.adsb_host.strip()
        cfg["adsb_tracker_host"] = "" if h_clean.lower() in ("none", "off", "false") else h_clean

    if args.lat is not None:
        cfg["camera_lat"] = args.lat
    if args.lon is not None:
        cfg["camera_lon"] = args.lon
    if args.alt_ft is not None:
        cfg["camera_alt_ft"] = args.alt_ft
    if args.discord_webhook:
        cfg["discord_webhook_url"] = args.discord_webhook.strip()
    if args.cpu_only:
        cfg["vlm_enabled"] = False
        cfg["day_enable_yolo"] = False

    TARGET_CFG.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    print(f"[OK] Saved configuration with {len(cfg.get('cameras', []))} camera node(s) to {TARGET_CFG}")
    print("\nNext Steps:")
    print("  1. Deploy capture software to your Raspberry Pi(s):")
    print("     python pi_capture_node/deploy_to_pi.py --all-from-config")
    print("  2. Start the Real-Time Allsky Detector & Web Observatory (:8094):")
    print("     python anomaly_detector/detector.py --daemon")


if __name__ == "__main__":
    main()
