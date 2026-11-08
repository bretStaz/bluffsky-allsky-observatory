#!/usr/bin/env python3
"""
BluffSky Option A — Automated Raspberry Pi 5 Deployment & Backup Manager
========================================================================
Deploys the Option A capture suite from E:\\allsky\\pi_capture_node\\ to any
Allsky Raspberry Pi 5 over SSH/SFTP, compiles libASICamera2.so from the ZWO
ARM64 SDK archive, preserves the legacy allsky.service as a 1-command backup,
and activates bluffsky-node.service.

Usage:
  # Deploy to Pi #1 (currently online):
  python E:\\allsky\\pi_capture_node\\deploy_to_pi.py --host 192.168.1.100 --node-id bluffsky-pi1

  # Deploy to Pi #2 (when brought online):
  python E:\\allsky\\pi_capture_node\\deploy_to_pi.py --host 192.168.1.101 --node-id bluffsky-pi2 --sync-mode lockstep --sync-grid-ms 20000

  # Switch a Pi back to legacy Allsky backup at any time:
  python E:\\allsky\\pi_capture_node\\deploy_to_pi.py --host 192.168.1.100 --mode legacy-backup
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import paramiko

HERE = Path(__file__).resolve().parent


def run_ssh(client: paramiko.SSHClient, cmd: str, check: bool = True) -> str:
    _, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {cmd}\nSTDOUT: {out}\nSTDERR: {err}")
    return out + ("\n" + err if err.strip() else "")


def upload_text_unix(sftp: paramiko.SFTPClient, local_path: Path, remote_path: str, mode: int = 0o644) -> None:
    content = local_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    with sftp.open(remote_path, "w") as rf:
        rf.write(content.encode("utf-8"))
    sftp.chmod(remote_path, mode)


def deploy(args: argparse.Namespace) -> None:
    print(f"[1/5] Connecting to {args.user}@{args.host}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs = {"hostname": args.host, "username": args.user, "timeout": 10}
    if args.password:
        connect_kwargs["password"] = args.password
    client.connect(**connect_kwargs)

    remote_node_dir = f"{args.allsky_home}/bluffsky_node"

    if args.mode == "legacy-backup":
        print(f"[*] Switching {args.host} back to legacy Allsky backup (allsky.service)...")
        out = run_ssh(client, f"bash {remote_node_dir}/switch_capture_mode.sh legacy-backup")
        print(out)
        client.close()
        return

    print(f"[2/5] Preparing {remote_node_dir} and uploading Option A files...")
    run_ssh(client, f"mkdir -p {remote_node_dir} {args.allsky_home}/tmp {args.allsky_home}/config {args.allsky_home}/images {args.allsky_home}/src/lib/armv8")
    sftp = client.open_sftp()

    upload_text_unix(sftp, HERE / "bluffsky_pi_node.py", f"{remote_node_dir}/bluffsky_pi_node.py", 0o755)
    upload_text_unix(sftp, HERE / "switch_capture_mode.sh", f"{remote_node_dir}/switch_capture_mode.sh", 0o755)

    # Render bluffsky-node.service dynamically for args.user and args.allsky_home
    svc_text = (HERE / "bluffsky-node.service").read_text(encoding="utf-8").replace("\r\n", "\n")
    svc_text = svc_text.replace("User=pi", f"User={args.user}").replace("Group=pi", f"Group={args.user}")
    svc_text = svc_text.replace("/home/pi/allsky", args.allsky_home)
    with sftp.open(f"{remote_node_dir}/bluffsky-node.service", "w") as sf:
        sf.write(svc_text.encode("utf-8"))

    # Optional: if user supplied --asi-sdk-lib locally, upload it to the Pi
    if getattr(args, "asi_sdk_lib", ""):
        sdk_p = Path(args.asi_sdk_lib)
        if sdk_p.exists():
            target_lib_name = "libASICamera2.so" if sdk_p.suffix == ".so" else "libASICamera2.a"
            print(f"[*] Uploading ZWO ASI SDK library {sdk_p.name} to {remote_node_dir}/{target_lib_name}...")
            sftp.put(str(sdk_p), f"{remote_node_dir}/{target_lib_name}")

    # Customize node_config.json for this specific Pi
    cfg = json.loads((HERE / "node_config.json").read_text(encoding="utf-8"))
    cfg["node_id"] = args.node_id
    cfg["allsky_home"] = args.allsky_home
    cfg["asi_lib_path"] = f"{remote_node_dir}/libASICamera2.so"
    if args.sync_mode:
        cfg["sync_mode"] = args.sync_mode
    if args.sync_grid_ms is not None:
        cfg["sync_grid_ms"] = args.sync_grid_ms
    if args.sync_offset_ms is not None:
        cfg["sync_offset_ms"] = args.sync_offset_ms
    if getattr(args, "lat", None) is not None:
        cfg["latitude_deg"] = float(args.lat)
    if getattr(args, "lon", None) is not None:
        cfg["longitude_deg"] = float(args.lon)
    if getattr(args, "enable_adsb", False):
        cfg["adsb_enabled"] = True
    elif getattr(args, "disable_adsb", False):
        cfg["adsb_enabled"] = False

    with sftp.open(f"{remote_node_dir}/node_config.json", "w") as rf:
        rf.write(json.dumps(cfg, indent=4).encode("utf-8"))
    sftp.close()

    # Optional Co-Located ADS-B SDR Receiver Setup (--enable-adsb / --disable-adsb)
    if getattr(args, "enable_adsb", False):
        lat_v = float(cfg.get("latitude_deg", 45.0))
        lon_v = float(cfg.get("longitude_deg", -100.0))
        print(f"[*] Enabling co-located ADS-B SDR receiver (readsb + tar1090) on {args.host} ({lat_v:.4f}, {lon_v:.4f})...")
        adsb_install_cmd = (
            "if ! command -v readsb >/dev/null 2>&1 && ! command -v dump1090-fa >/dev/null 2>&1; then "
            "  sudo apt-get update && sudo apt-get install -y wget curl rtl-sdr && "
            "  sudo bash -c \"$(wget -nv -O - https://github.com/wiedehopf/adsb-scripts/raw/master/readsb-install.sh)\"; "
            "fi && "
            f"if command -v readsb-set-location >/dev/null 2>&1; then sudo readsb-set-location {lat_v:.5f} {lon_v:.5f}; fi && "
            "sudo systemctl enable --now readsb 2>/dev/null || sudo systemctl enable --now dump1090-fa 2>/dev/null || true"
        )
        print(run_ssh(client, adsb_install_cmd, check=False).strip())
    elif getattr(args, "disable_adsb", False):
        print(f"[*] Disabling co-located ADS-B SDR receiver on {args.host}...")
        run_ssh(client, "sudo systemctl disable --now readsb 2>/dev/null || true", check=False)

    # Ensure Python venv & ZWO USB udev rules exist (for brand-new Raspberry Pi installs)
    bootstrap_cmd = (
        f"if [ ! -f /etc/udev/rules.d/99-asi.rules ] && [ ! -f /etc/udev/rules.d/zwo.rules ]; then "
        f'echo \'ACTION=="add", ATTR{{idVendor}}=="03c3", RUN+="/bin/sh -c \\\'/bin/echo 200 >/sys/module/usbcore/parameters/usbfs_memory_mb\\\'" , MODE="0666"\' | sudo tee /etc/udev/rules.d/99-asi.rules >/dev/null && sudo udevadm control --reload-rules; fi && '
        f"if [ ! -x {args.allsky_home}/venv/bin/python3 ]; then "
        f"sudo apt-get update && sudo apt-get install -y g++ libusb-1.0-0-dev python3-venv python3-opencv python3-numpy python3-ephem && "
        f"python3 -m venv --system-site-packages {args.allsky_home}/venv; fi"
    )
    run_ssh(client, bootstrap_cmd, check=False)

    print("[3/5] Verifying or building native ARM64 libASICamera2.so...")
    build_cmd = (
        f"if [ -f {remote_node_dir}/libASICamera2.a ]; then "
        f"  g++ -shared -Wl,--whole-archive {remote_node_dir}/libASICamera2.a -Wl,--no-whole-archive -lusb-1.0 -lpthread -o {remote_node_dir}/libASICamera2.so; "
        f"elif [ -f {args.allsky_home}/src/lib/armv8/libASICamera2.a ] && [ ! -f {remote_node_dir}/libASICamera2.so ]; then "
        f"  g++ -shared -Wl,--whole-archive {args.allsky_home}/src/lib/armv8/libASICamera2.a -Wl,--no-whole-archive -lusb-1.0 -lpthread -o {remote_node_dir}/libASICamera2.so; "
        f"elif [ -f /usr/local/lib/libASICamera2.so ] && [ ! -f {remote_node_dir}/libASICamera2.so ]; then "
        f"  cp /usr/local/lib/libASICamera2.so {remote_node_dir}/libASICamera2.so; "
        f"fi && ls -lh {remote_node_dir}/libASICamera2.so"
    )
    print(run_ssh(client, build_cmd).strip())

    print("[4/5] Installing systemd unit & preserving legacy allsky.service as clean backup...")
    install_cmd = (
        f"sudo cp {remote_node_dir}/bluffsky-node.service /etc/systemd/system/bluffsky-node.service && "
        f"bash {remote_node_dir}/switch_capture_mode.sh option-a"
    )
    print(run_ssh(client, install_cmd).strip())

    print("[5/5] Verifying live camera initialization & telemetry API on port 8096...")
    time.sleep(3.0)
    verify_cmd = (
        "journalctl -u bluffsky-node.service -n 20 --no-pager && "
        "curl -s http://127.0.0.1:8096/api/status"
    )
    print(run_ssh(client, verify_cmd, check=False).strip())
    client.close()
    print(f"\n[OK] Deployment to {args.host} ({args.node_id}) complete!")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Deploy BluffSky Option A to 1..N Raspberry Pi 5 Camera Nodes")
    parser.add_argument("--host", type=str, default="192.168.1.100", help="Pi IP or hostname (for single-node deploy)")
    parser.add_argument("--user", type=str, default="pi", help="SSH username")
    parser.add_argument(
        "--password",
        type=str,
        default=os.environ.get("BLUFFSKY_SSH_PASSWORD", ""),
        help="Optional SSH password (defaults to SSH public key auth or BLUFFSKY_SSH_PASSWORD env var)",
    )
    parser.add_argument("--node-id", type=str, default="bluffsky-pi1", help="Node identifier")
    parser.add_argument("--allsky-home", type=str, default="/home/pi/allsky", help="Allsky home dir")
    parser.add_argument("--mode", choices=["option-a", "legacy-backup"], default="option-a")
    parser.add_argument("--sync-mode", type=str, default=None, help="auto, lockstep, staggered, freerun")
    parser.add_argument("--sync-grid-ms", type=int, default=None, help="Stereo sync grid in ms (e.g. 20000)")
    parser.add_argument("--sync-offset-ms", type=int, default=None, help="Stereo sync phase offset in ms")
    parser.add_argument("--lat", type=float, default=None, help="Camera GPS latitude (decimal degrees)")
    parser.add_argument("--lon", type=float, default=None, help="Camera GPS longitude (decimal degrees)")
    parser.add_argument(
        "--enable-adsb",
        action="store_true",
        help="Optional: Enable & auto-install co-located 1090 MHz ADS-B SDR receiver (readsb/tar1090) on this Camera Pi",
    )
    parser.add_argument(
        "--disable-adsb",
        action="store_true",
        help="Optional: Disable co-located ADS-B SDR receiver on this Camera Pi (default is off)",
    )
    parser.add_argument(
        "--all-from-config",
        action="store_true",
        help="Deploy to all enabled 1..N camera nodes listed in anomaly_detector/config.json",
    )
    parser.add_argument(
        "--server-config",
        type=str,
        default=str(HERE.parent / "anomaly_detector" / "config.json"),
        help="Path to server config.json when using --all-from-config",
    )
    args = parser.parse_args()

    if args.all_from_config:
        from urllib.parse import urlparse

        cfg_path = Path(args.server_config)
        if not cfg_path.exists():
            raise SystemExit(f"Server config not found: {cfg_path}")
        srv_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cams = [c for c in (srv_cfg.get("cameras") or []) if isinstance(c, dict) and c.get("enabled", True)]
        if not cams:
            cams = [{"id": "cam1", "node_id": "bluffsky-pi1", "base_url": srv_cfg.get("allsky_base_url", "http://192.168.1.100")}]
        n_cams = len(cams)
        print(f"[*] Fleet Deployment: Found {n_cams} enabled camera node(s) in {cfg_path}")
        for idx, cam in enumerate(cams):
            base_url = str(cam.get("base_url") or cam.get("allsky_base_url") or "http://192.168.1.100")
            host = urlparse(base_url).hostname or base_url
            node_id = str(cam.get("node_id") or cam.get("id") or f"bluffsky-pi{idx + 1}")
            node_args = argparse.Namespace(**vars(args))
            node_args.host = host
            node_args.node_id = node_id
            if node_args.lat is None and cam.get("camera_lat") is not None:
                node_args.lat = float(cam["camera_lat"])
            if node_args.lon is None and cam.get("camera_lon") is not None:
                node_args.lon = float(cam["camera_lon"])
            if not node_args.enable_adsb and not node_args.disable_adsb and cam.get("adsb_enabled"):
                node_args.enable_adsb = True
            if args.sync_mode == "staggered" and args.sync_grid_ms and args.sync_offset_ms is None:
                node_args.sync_offset_ms = int((idx * args.sync_grid_ms) // max(1, n_cams))
            elif args.sync_mode == "lockstep" and args.sync_offset_ms is None:
                node_args.sync_offset_ms = 0
            print(f"\n=== Deploying Node {idx + 1}/{n_cams}: {node_id} ({host}) ===")
            deploy(node_args)
    else:
        deploy(args)


if __name__ == "__main__":
    main()
