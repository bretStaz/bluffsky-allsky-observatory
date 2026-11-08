# 🍓 BluffSky Option A — Raspberry Pi Capture Node (`pi_capture_node/`)

This directory contains the lightweight **Camera Capture & Telemetry Daemon** that runs on each of your **`1..N` Raspberry Pis** connected to a ZWO ASI USB 3.0 camera.

> **What runs here vs. the Workstation?**
> - **Each Raspberry Pi (`1..N`)** runs **ONLY** this folder (`bluffsky_pi_node.py` installed at `/opt/bluffsky-node` and managed by `bluffsky-node.service`).
> - **Your Workstation / Server** runs the `anomaly_detector/` folder, which polls each Raspberry Pi over HTTP on port **`:8096`**.

---

## Files in `pi_capture_node/`

| File | Where It Runs | Description |
| :--- | :---: | :--- |
| **[`bluffsky_pi_node.py`](bluffsky_pi_node.py)** | **🍓 Raspberry Pi** (`/opt/bluffsky-node/`) | Native ARM64/x64 ZWO ASI SDK (`ctypes` `libASICamera2.so`) capture daemon with 4-regime dynamic exposure/gain control (`DAY`, `TWILIGHT`, `MOONLIT_NIGHT`, `DARK_NIGHT`), lunar exclusion metering, deterministic UTC shutter synchronization, `/dev/shm` RAM buffering, optional background SD archival + FTP upload, and built-in port `:8096` HTTP server. |
| **[`node_config.json`](node_config.json)** | **🍓 Raspberry Pi** (`/opt/bluffsky-node/`) | Per-camera exposure, gain, white-balance, GPS latitude/longitude, and stereo shutter sync configuration. |
| **[`bluffsky-node.service`](bluffsky-node.service)** | **🍓 Raspberry Pi** (`/etc/systemd/system/`) | `systemd` service unit (`Conflicts=allsky.service`) that starts `bluffsky_pi_node.py` automatically on boot. |
| **[`switch_capture_mode.sh`](switch_capture_mode.sh)** | **🍓 Raspberry Pi** (`/opt/bluffsky-node/`) | One-command switcher on the Pi between `option-a` (`bluffsky-node.service`) and `legacy-backup` (`allsky.service`). |
| **[`requirements.txt`](requirements.txt)** | **🍓 Raspberry Pi** | Lightweight Python dependencies required on the Pi (`numpy`, `opencv-python-headless`, `Pillow`, `ephem`). |
| **[`deploy_to_pi.py`](deploy_to_pi.py)** | **🖥️ Workstation** | Automated SSH/SFTP deployment script that pushes this folder from your Workstation to `1..N` Raspberry Pis and starts `bluffsky-node.service`. |

---

## HTTP Endpoints Served by Each Raspberry Pi (`Port :8096`)

Once `bluffsky-node.service` is running on a Raspberry Pi, your Workstation (and any browser on your LAN) can access:
- **`http://<pi-ip>:8096/current/tmp/image.jpg`**: Latest full-resolution (`3552 × 3552`) frame served directly from `/dev/shm` RAM.
- **`http://<pi-ip>:8096/api/status`**: Live JSON telemetry (current regime, exposure time, gain, sensor temperature, Sun/Moon altitude, shutter sync offset, `adsb_enabled`, and frame count).
- **`http://<pi-ip>:8096/api/adsb`**: *(Optional, off by default)* When `"adsb_enabled": true`, serves the live `/run/readsb/aircraft.json` from a co-located 1090 MHz USB RTL-SDR dongle plugged into the Pi's black USB 2.0 port with zero SD card writes.
- **`http://<pi-ip>:8096/current/tmp/overlaydebug.txt`**: Key-value exposure/gain/temperature metadata compatible with legacy tools.
- **`http://<pi-ip>:8096/images/<YYYYMMDD>/`**: Browse archived daily captures stored on the Pi's SD/NVMe drive.

---

## Deploying to Your Raspberry Pi(s) from Your Workstation

Run these commands from the root of the repository **on your Workstation**:

### 1. Deploy to All `1..N` Cameras Configured in `anomaly_detector/config.json`
```bash
python pi_capture_node/deploy_to_pi.py --all-from-config --user pi
```

### 2. Deploy in Lockstep Stereo Mode Across `1..N` Cameras (Shutters Locked to a 20s UTC Grid)
```bash
python pi_capture_node/deploy_to_pi.py --all-from-config --user pi --sync-mode lockstep --sync-grid-ms 20000
```

### 3. Deploy in Staggered Continuous Mode Across `1..N` Cameras (Shutters Spaced by `grid_ms / N`)
```bash
python pi_capture_node/deploy_to_pi.py --all-from-config --user pi --sync-mode staggered --sync-grid-ms 20000
```

### 4. Deploy to a Single Specific Raspberry Pi
```bash
python pi_capture_node/deploy_to_pi.py --host 192.168.1.100 --user pi --node-id bluffsky-pi1
```

### 5. Optional: Enable or Disable Co-Located 1090 MHz ADS-B SDR Receiver on Camera Pi(s)
By default, co-located ADS-B on the Camera Pi is **OFF** (`"adsb_enabled": false`). Because Option A uses only `~12%` CPU and `~235 MB` RAM, you can optionally plug a 1090 MHz RTL-SDR dongle (e.g., FlightAware Pro Stick Plus) into a **black USB 2.0 port** on one or all of your Camera Pis (keeping the ZWO ASI camera on the **blue USB 3.0 port**) and enable it with one flag:
```bash
# Enable & auto-install readsb + tar1090 (RAM /run/readsb) on all Camera Pis:
python pi_capture_node/deploy_to_pi.py --all-from-config --user pi --enable-adsb

# Or enable on a single Camera Pi only:
python pi_capture_node/deploy_to_pi.py --host 192.168.1.100 --user pi --node-id bluffsky-pi1 --enable-adsb

# Disable co-located ADS-B at any time (returns to default off state):
python pi_capture_node/deploy_to_pi.py --all-from-config --user pi --disable-adsb
```

### 6. Useful Commands Directly on the Raspberry Pi (via SSH)
```bash
# Check capture daemon status & live logs on the Raspberry Pi:
sudo systemctl status bluffsky-node.service
journalctl -u bluffsky-node.service -f

# If your Pi also has a legacy allsky.service installed, switch between Option A and Legacy at any time:
sudo /opt/bluffsky-node/switch_capture_mode.sh option-a
sudo /opt/bluffsky-node/switch_capture_mode.sh legacy-backup
```
