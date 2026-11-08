# 🌌 BluffSky Multi-Camera (`1..N`) All-Sky Observatory & AI Sky Analysis Suite

**Turn `1` to `N` Raspberry Pi + ZWO ASI Fisheye Cameras into an Automated Sky Survey, 3D Stereo Meteor & Bolide Triangulator, Live SGP4 Space & ADS-B Airspace Radar, and Super-Resolution Deep-Sky Astrophotography Engine.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Cameras: 1..N Stereo Array](https://img.shields.io/badge/Cameras-1..N%20Multi--Camera%20Stereo-22c55e)](docs/SHOWCASE.md)
[![Raspberry Pi: Pi 4 / Pi 5](https://img.shields.io/badge/Raspberry%20Pi-Pi%204%20%7C%20Pi%205%20(~12%25%20CPU)-ef4444)](pi_capture_node/README.md)
[![Workstation: CPU-Only or GPU](https://img.shields.io/badge/Workstation-Windows%20%7C%20Linux%20%7C%20macOS-38bdf8)](#-hardware-requirements)
[![Showcase: 51 Real Captures](https://img.shields.io/badge/Visual%20Showcase-51%20Full--Res%20Plates-f59e0b)](docs/SHOWCASE.md)

![4K Northern Milky Way Galactic Plane Panorama — Captured & Stacked by BluffSky](docs/showcase/milky_way_panorama_annotated.jpg)
*Above: **`4K` Northern Milky Way Galactic Plane Panorama (`3840 × 1248`)** — automatically sidereally de-rotated and co-added from `247 × 30s` (`2.06 hours`) of untracked **ZWO ASI676MC** (`3552 × 3552` 180° fisheye) exposures.*

---

## 🚀 Why Install BluffSky? What Makes This Stack Different

Traditional all-sky camera software treats an astronomy camera like a basic weather webcam: it burns text overlays permanently into your pixels on the Raspberry Pi, hammers the MicroSD card with constant writes, and leaves you with a folder of unanalyzed JPEGs.

**BluffSky is a complete, observatory-grade software stack** that separates **deterministic, zero-SD-wear raw science capture on `1..N` Raspberry Pis** from a **real-time astrometric, orbital, and deep-sky analysis engine on your Workstation/Server**. Everything shown below was captured and generated automatically on a **single untracked ZWO ASI676MC (`1.55 mm` 180° fisheye) + Raspberry Pi**:

| Orionid Bolide Fireball (`138σ` 4-Panel Forensic Card) | Geostationary "Clarke Belt" (`41` Satellites at `35,786 km`) | Ice Giant Neptune at `2.69 Billion Miles` (`mag +7.82`) |
| :---: | :---: | :---: |
| ![Orionid Bolide Fireball](docs/showcase/top1_orionid_bolide_20261020055700.jpg) | ![Geostationary Clarke Belt](docs/showcase/orb_geo_clarke_belt_annotated.jpg) | ![Planet Neptune Ephemeris Shift-and-Stack](docs/showcase/sol_neptune_annotated.jpg) |
| **Daytime Sky Radar: Unseen Satellites + Live ADS-B Planes** | **Nighttime Sky Radar: Earth-Shadowed Satellites + Planes** | **Bode's (`M81`) & Cigar (`M82`) Galaxies (`11.8 Mly`, `2.5×` Drizzle)** |
| ![Daytime Sky Radar](docs/showcase/sky_radar_day_orbital_and_planes.jpg) | ![Nighttime Sky Radar](docs/showcase/sky_radar_night_orbital_and_planes.jpg) | ![M81 and M82 Galaxies](docs/showcase/m81_m82_galaxies_annotated.jpg) |

![Documented 23-Night Master J2000.0 Multi-Epoch Deep-Sky & Solar System Atlas](docs/showcase/master_j2000_atlas_documented.jpg)
*Above: **23-Night (`1,832`-Frame) Sidereally De-Rotated Master J2000.0 Sky Atlas (`+4.08 mag` depth gain to `mag +8.5`)** — automatically recovering **Mars** (`14` nights), **Jupiter** (`7` nights), **Ice Giant Uranus** (`7` nights, `mag +5.62`), **Main-Belt Asteroid `4 Vesta`** (`2` nights, `mag +5.68`), `6` Messier deep-sky objects, and `3` multi-night moving tracklets (`BLUFF-TRK-01..03`).*

---

### ✨ 10 Flagship Capabilities Built Into This Repository

| # | Capability | What BluffSky Actually Does Under the Hood |
| :-: | :--- | :--- |
| **1** | **🍓 Featherweight, Zero-SD-Wear Raspberry Pi Capture (`~12%` CPU)** | [`bluffsky_pi_node.py`](pi_capture_node/bluffsky_pi_node.py) talks directly to ZWO's `libASICamera2.so` C-SDK, buffers pristine science frames in `/dev/shm` RAM (`0 MB/day` SD card wear in the capture loop), runs a **4-Regime Dynamic Exposure/Gain Controller** (`DAY` / `TWILIGHT` / `MOONLIT_NIGHT` / `DARK_NIGHT`) with dynamic Moon-glare exclusion, and locks shutter open times across `1..N` cameras to `< 1 ms` UTC grid boundaries. |
| **2** | **🧭 Compass-Free 6-DOF Star-Field Calibration (`< 0.11°` RMS)** | Mount your camera at any arbitrary rotation. Enter only your GPS Lat/Lon once, and [`camera_realign.py`](anomaly_detector/camera_realign.py) matches up to 75 live stars against the Yale Bright Star catalog to solve your exact zenith pixel center, 2-axis mount tilt, compass azimuth (`±0.05°`), and 5th-order polynomial fisheye distortion (`k2`, `k3`) — while archiving prior `BeforeNight` epochs if the camera ever moves. |
| **3** | **🤝 True `1..N` Multi-Camera 3D Stereo Parallax & Triangulation** | Deploy 1, 2, or $N$ cameras (`5 m` to `50+ km` apart). When 2+ cameras detect an event within $|\Delta t| \le 12\text{ s}$, [`detector.py`](anomaly_detector/detector.py) computes angular parallax ($\Delta\theta$), **vetoes dome bugs (`< 3 m`), birds (`15–150 m`), and single-sensor cosmic rays (`100%`)**, and triangulates true 3D altitude (`80–120 km` meteors vs. `> 400 km` orbital passes). |
| **4** | **🛰️ Live SGP4 Orbital Radar (`16,600+` TLEs) — Visible & Unseen** | [`orbital_tracker.py`](anomaly_detector/orbital_tracker.py) propagates 16,600+ NORAD TLEs (`stations`, `visual`, `active`, `analyst` classified payloads, `starlink`, `geo`) to tag sunlit LEO passes, tumbling rocket bodies (`SL-16`, `CZ-2C`, `BREEZE-M`), classified recon satellites (`USA / NROL`), stationary **GEO Clarke Belt** satellites at `35,786 km`, and **optically unseen satellites** (daylight passes & nighttime Earth-shadowed passes). |
| **5** | **✈️ Real-Time ADS-B Aircraft Tracking + Optional Co-Located Pi SDR** | [`adsb_tracker.py`](anomaly_detector/adsb_tracker.py) matches night strobes and daytime contrails to exact airline callsigns, tail numbers, aircraft models, altitudes, and speeds. You can even plug a 1090 MHz RTL-SDR dongle directly into your Camera Pi(s) (`--enable-adsb`, off by default) for **multi-receiver union merging** across all cameras! |
| **6** | **☄️ Multi-Epoch J2000.0 Survey, Asteroid Recovery & Blind Tracklets** | [`deep_sky_survey.py`](anomaly_detector/deep_sky_survey.py) and [`astrophysics_patrol.py`](anomaly_detector/astrophysics_patrol.py) co-add clear nights onto a `2880 × 864` J2000 equatorial grid (`+4.08 mag` gain to `mag +8.5`), perform **Reference Image Subtraction ($\Delta I = I_{\text{night}} - I_{\text{master}}$)**, recover planets & main-belt asteroids (`4 Vesta` at `mag +5.68`), link 3-night collinear moving tracklets (`BLUFF-TRK`), and flag sub-arcminute optical flashes (`BLUFF-OT`). |
| **7** | **🪐 Solar System Ephemeris "Shift-and-Stack" Out to `2.69 Billion Miles`** | [`orbital_and_planetary_showcase.py`](anomaly_detector/orbital_and_planetary_showcase.py) tracks moving planets along their exact sub-pixel J2000 ephemerides across weeks to recover **Mars, Jupiter, Saturn, Uranus (`mag +5.61`), and 8th-magnitude Ice Giant Neptune (`mag +7.82` at `28.99 AU`)** — over `10×` fainter than naked-eye visibility — from an untracked `1.55 mm` fisheye lens. |
| **8** | **🌌 `4K` Milky Way Panorama & `22` Super-Resolution (`2.5×` Drizzle) Stacks** | [`deep_sky_astrophoto.py`](anomaly_detector/deep_sky_astrophoto.py) extracts `960 × 720` J2000 tangential windows from the all-sky archive, registers stars to `< 0.08 px` via sub-tile phase correlation, removes sky gradients, applies asymmetric sigma-clipping (erasing 100% of airplanes/satellites), and runs Richardson-Lucy PSF deconvolution across `22` Messier/NGC galaxies, nebulae, supernova remnants, and globular clusters. |
| **9** | **🧠 Two-Stage AI Classification + 1-Click Active Learning (`11-D` k-NN)** | Every detection produces a 4-panel forensic verification card (`BEFORE` / `EVENT` / `DIFFERENCE` / `2.5× ZOOM`). A fast `~180 ms` mathematical classifier pairs with [`feedback_learner.py`](anomaly_detector/feedback_learner.py) (`11-D` k-NN active learner trained with 1-click buttons in the Web UI or Discord) and optional Stage-2 local Vision LLM verification (Ollama / LM Studio). Runs on **CPU-only mini-PCs** (`--cpu-only`) or **GPU workstations**. |
| **10** | **🖥️ 4-Portal Interactive Web Observatory (`:8094`) & Self-Healing Guard** | [`web_portal.py`](anomaly_detector/web_portal.py) serves **`/`** (Forensic Event Dashboard with `1..N` camera & stereo filters), **`/live`** (Live Sky Dome Radar with plane & satellite vectors), **`/discoveries`** (Interactive J2000 Blink Comparator), and **`/astrophoto`** (`Single 30s ⇄ Deep Stack` split-screen comparison sliders & `4000%` zoom lightbox), supervised 24/7 by [`allsky_workflow_guard.py`](anomaly_detector/allsky_workflow_guard.py). |

> 👉 **[Explore All 51 Full-Resolution Captures & Plates in `docs/SHOWCASE.md` →](docs/SHOWCASE.md)**  
> 👉 **[Jump Straight to the Step-by-Step Installation Guide →](#-complete-step-by-step-installation-guide)**

---

## 🏗️ System Architecture: What Runs on the Raspberry Pi vs. the Workstation?

BluffSky splits the workload cleanly across your local network so your outdoor **Raspberry Pi(s)** stay cool, lightweight, and 100% focused on deterministic camera exposures, while your indoor **Workstation / Server** handles all heavy computer vision, orbital mechanics, deep-sky stacking, and the Web UI:

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  🍓 OUTDOOR CAMERA NODE(S): 1..N Raspberry Pis (Raspberry Pi OS 64-bit)      │
│  Runs ONLY: pi_capture_node/ (bluffsky_pi_node.py via bluffsky-node.service) │
│  ──────────────────────────────────────────────────────────────────────────  │
│  • Talks directly to USB 3.0 ZWO ASI camera via ZWO's libASICamera2.so       │
│  • 4-Regime Auto-Exposure/Gain (DAY / TWILIGHT / MOONLIT_NIGHT / DARK_NIGHT) │
│  • Writes live frames to RAM (/dev/shm) — zero SD card wear in capture loop  │
│  • Locks shutter timestamps to UTC grid (< 2 ms) for multi-camera stereo     │
│  • Serves latest frame & telemetry over HTTP on Port :8096                   │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │  LAN (Wi-Fi or Gigabit Ethernet)
                                       │  Polls http://<pi-ip>:8096/current/tmp/image.jpg
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  🖥️ INDOOR WORKSTATION / SERVER: 1 PC, Server, or Mac (Windows/Linux/macOS)  │
│  Runs: anomaly_detector/ (detector.py + web_portal.py + analysis modules)    │
│  ──────────────────────────────────────────────────────────────────────────  │
│  • detector.py          → Polls 1..N Pis, detects meteors/bolides/transients,│
│                           computes 3D multi-camera stereo triangulation,     │
│                           and runs Stage-2 Vision LLM verification           │
│  • adsb_tracker.py      → Live ADS-B aircraft tracking & flight-path tagging │
│  • orbital_tracker.py   → SGP4 satellite, Starlink & rocket body matching    │
│  • camera_realign.py    → Automated 6-DOF star-field astrometric calibration │
│  • deep_sky_survey.py &                                                      │
│    astrophysics_patrol.py → J2000 sky atlas, optical flash transients        │
│                             (BLUFF-OT) & moving object tracklets (BLUFF-TRK) │
│  • deep_sky_astrophoto.py &                                                  │
│    orbital_and_planetary_showcase.py → 4K Milky Way Panorama, 22 Deep-Sky    │
│                             2.5x Drizzle Stacks, Solar System Planets        │
│                             (out to Neptune) & GEO Clarke Belt Composites    │
│  • feedback_learner.py  → 11-D k-NN active learning & Discord bot            │
│  • web_portal.py (:8094)→ Interactive Web Observatory UI (http://localhost:8094)│
└──────────────────────────────────────────────────────────────────────────────┘
```

### Quick Summary Table: What Runs Where?

| Component / Script | Runs On | Purpose | Network Port |
| :--- | :---: | :--- | :---: |
| **[`pi_capture_node/bluffsky_pi_node.py`](pi_capture_node/bluffsky_pi_node.py)**<br>*(via `bluffsky-node.service`)* | **🍓 Each Raspberry Pi (`1..N`)** | Direct ZWO ASI camera capture, 4-regime dynamic exposure/gain control, `/dev/shm` RAM frame buffer, and HTTP frame/telemetry server. | **`:8096`** *(on each Pi)* |
| **[`setup_observatory.py`](setup_observatory.py)** | **🖥️ Workstation** *(run once)* | Interactive wizard that generates `anomaly_detector/config.json` with your `1..N` Pi IPs and GPS coordinates. | — |
| **[`pi_capture_node/deploy_to_pi.py`](pi_capture_node/deploy_to_pi.py)** | **🖥️ Workstation** *(run once)* | Connects to your `1..N` Raspberry Pis over SSH, copies `pi_capture_node/` to `/opt/bluffsky-node` on each Pi, and starts `bluffsky-node.service`. | Uses SSH (`:22`) |
| **[`anomaly_detector/detector.py`](anomaly_detector/detector.py)** | **🖥️ Workstation** *(24/7 daemon)* | Main analysis engine: polls all `1..N` Pis, runs anomaly detection, 3D stereo parallax, ADS-B planes, SGP4 satellites, VLM verification, and auto-starts `web_portal.py`. | Outbound HTTP to Pis (`:8096`) |
| **[`anomaly_detector/web_portal.py`](anomaly_detector/web_portal.py)** | **🖥️ Workstation** *(24/7 daemon)* | Interactive Web Observatory UI (`/`, `/live`, `/discoveries`, `/astrophoto`). Automatically started by `detector.py --daemon`. | **`:8094`** *(on Workstation)* |
| **[`anomaly_detector/allsky_workflow_guard.py`](anomaly_detector/allsky_workflow_guard.py)** | **🖥️ Workstation** *(optional 24/7 watchdog)* | Monitors all `1..N` Raspberry Pis, `detector.py`, and `web_portal.py`, automatically restarting any stalled service. | — |

> 💡 **Important**: You do **NOT** need to install `anomaly_detector/`, PyTorch, Ollama, or the Web Portal on your Raspberry Pi. The Raspberry Pi only runs the lightweight [`pi_capture_node/`](pi_capture_node/) folder (~235 MB RAM, ~12% CPU).

---

## 📸 Visual Showcase — What You Will Capture & Discover with BluffSky

> 👉 **[View the Complete 51-Image High-Resolution Observatory Showcase (`docs/SHOWCASE.md`)](docs/SHOWCASE.md)**

Every image below was captured and automatically processed from a single untracked **ZWO ASI676MC** (`3552 × 3552` 180° fisheye) + **Raspberry Pi** running BluffSky:

### 1. `4K` Northern Milky Way Galactic Plane Panorama (`247 × 30s = 2.06 Hours`)
![4K Northern Milky Way Galactic Plane Panorama — Annotated](docs/showcase/milky_way_panorama_annotated.jpg)

### 2. Documented Master J2000.0 Multi-Epoch Deep-Sky & Solar System Atlas (`23` Clear Nights, `1,832` De-Rotated Frames)
![Documented Master J2000.0 Multi-Epoch Deep-Sky & Solar System Atlas](docs/showcase/master_j2000_atlas_documented.jpg)

#### How to Read the Master J2000.0 Multi-Epoch Atlas (`deep_sky_survey.py`)
Across **23 clear astronomical nights** (`4,478` local night exposures scanned, **`1,832` clear dark frames** sidereally de-rotated onto a `2880 × 864` J2000.0 equatorial grid at `7.5 arcmin/px`), single-frame outlier rejection (`flux_sum - flux_max`) removes 100% of meteors, aircraft strobes, and LEO satellites, while achieving a **`+4.08 mag` depth gain** ($\Delta m = 1.25 \log_{10}(1832)$, reaching **`mag +8.5`** on an untracked 180° fisheye):

| Atlas Feature / Color Code | Object(s) Recovered in the 23-Night Stack | Astrometric & Photometric Details |
| :--- | :--- | :--- |
| **🟢 Green Circles**<br>*(Solar System Planets, Ice Giants & Main-Belt Asteroids)* | **Planet Mars** (`14` clear nights)<br>**Gas Giant Jupiter** (`7` clear nights)<br>**Ice Giant Uranus** (`7` clear nights)<br>**Main-Belt Asteroid `4 Vesta`** (`2` clear nights) | • **Mars**: `11.51°` ecliptic arc in Cancer/Leo (`0.523°/day` / `31.4'/day`, `mag +0.84`, Peak SNR `106.9σ`)<br>• **Jupiter**: `2.58°` arc in Cancer (`0.136°/day` / `8.2'/day`, `mag -1.80`, Peak SNR `53.7σ`)<br>• **Uranus**: `0.49°` retrograde arc in Taurus (`0.024°/day`, **`mag +5.62`** at `1.9B miles`, SNR `8.1σ`)<br>• **Asteroid `4 Vesta`**: `1.88°` arc in Leo (`0.267°/day` / `16.0'/day`, sub-visual **`mag +5.68`**, SNR `5.9σ`) |
| **🟡 Gold / Cyan Circles**<br>*(Messier Deep-Sky Landmarks)* | **`M31` Andromeda Galaxy**, **`M33` Triangulum Galaxy**, **Perseus Double Cluster**, **`M45` Pleiades**, **`M44` Beehive Cluster**, **`M42` Orion Nebula** | Stationary deep-sky galaxies (`2.5–2.7 Mly`), emission nebulae, and open star clusters resolved directly in the multi-night J2000 median reference atlas. |
| **🟠 Orange Circles**<br>*(Candidate 3-Night Moving Tracklets)* | **`BLUFF-TRK-01`** (`2.53°/day`, Pisces/Cetus)<br>**`BLUFF-TRK-02`** (`2.68°/day`, Leo/Sextans)<br>**`BLUFF-TRK-03`** (`2.59°/day`, Hydra) | Unmatched multi-frame point-source residuals after Reference Image Subtraction ($\Delta I = I_{\text{night}} - I_{\text{master}}$) linked across 3 nights along collinear trajectories ($\le 0.23^\circ$ error). |
| **⬛ Black Triangular Cutout**<br>*(RA `13h–17h`, Southern Horizon)* | **Daytime Solar Transit & Camera Elevation Mask (`Alt ≥ 25°`)** | During the Oct–Nov survey epoch, the Sun resides in Virgo/Libra/Scorpius (`RA 13.5h–15.5h`), so those Right Ascensions transit only in **daylight**; the curved lower boundary is the camera's `Alt ≥ 25°` astrometric horizon cutoff from `~48° N` latitude. |

### 3. Live Day & Night Sky Radar: Unseen Orbital Objects (Daylight, Earth-Shadowed & Deep-Space) + ADS-B Aircraft
Because [`camera_realign.py`](anomaly_detector/camera_realign.py) locks the 6-DOF fisheye lens geometry to `< 0.11°` against the stars and [`orbital_tracker.py`](anomaly_detector/orbital_tracker.py) propagates **16,600+ NORAD TLE space objects** via SGP4 alongside live **ADS-B aircraft transponders** ([`adsb_tracker.py`](anomaly_detector/adsb_tracker.py)), BluffSky maps the exact overhead pixel coordinates of both **optically visible** targets and **optically unseen orbital objects** (satellites passing overhead in bright daylight, LEO satellites in Earth's shadow at night, and sub-visual MEO GPS / GEO Clarke Belt / classified spacecraft):

| Daytime Sky Radar: Unseen Orbital Objects + Live ADS-B Planes | Nighttime Sky Radar: Earth-Shadowed (Dark) + Sunlit Satellites & Planes |
| :---: | :---: |
| ![Daytime Sky Radar — Unseen Orbital Objects and ADS-B Planes](docs/showcase/sky_radar_day_orbital_and_planes.jpg) | ![Nighttime Sky Radar — Earth-Shadowed and Sunlit Orbital Objects and ADS-B Planes](docs/showcase/sky_radar_night_orbital_and_planes.jpg) |
| **Diamond Reticles** plot exact SGP4 sky positions of spacecraft invisible in daylight (`0.48 ms` exposure): `NAVSTAR` GPS (`20,091 km`), `BEIDOU-3`, `MERIDIAN`, `STARLINK`, `KUIPER`, `BREEZE-M` rocket bodies, and `CXO` (Chandra X-Ray Observatory at `135,210 km`) alongside `3` live **ADS-B commercial flights** (`ASA412`, `ACA572`, `DAL881`). | Plots **Earth-shadowed (eclipsed) LEO satellites** (`ONEWEB-0266`, `STARLINK-32894`, `YAM-13`, `CZ-2C R/B` rocket body) that do not reflect sunlight alongside sub-visual MEO/GEO spacecraft (`GALILEO`, `COSMOS 2518`, `NAVSTAR`) and live night **ADS-B aircraft** (`UAL1083`, `ASA329`). |

| Night ADS-B Flight Capture (`Air Canada ACA572` — Airbus A220-300) | Simultaneous ADS-B Flight (`ASA225`) + LEO Satellite (`STARLINK-3136`) | Daytime Aircraft Contrail Detection (`1.2 ms` Exposure) |
| :---: | :---: | :---: |
| ![Night ADS-B Aircraft Detection](docs/showcase/adsb_aircraft_detection_night.jpg) | ![Simultaneous ADS-B Aircraft and Starlink Satellite](docs/showcase/adsb_aircraft_and_starlink_night.jpg) | ![Daytime Aircraft Contrail Detection](docs/showcase/adsb_aircraft_daytime_contrail.jpg) |

### 4. Real-Time 4-Panel Bolide Fireball, Meteor & Optical Transient Detections
| Orionid Bolide Fireball (`138σ` Peak) | Pleiades Fireball Composite | Simultaneous Double Meteor |
| :---: | :---: | :---: |
| ![Orionid Bolide Fireball](docs/showcase/top1_orionid_bolide_20261020055700.jpg) | ![Pleiades Fireball](docs/showcase/top2_pleiades_fireball_20261020060255.jpg) | ![Simultaneous Double Meteor](docs/showcase/top4_double_meteor_20261017203728.jpg) |
| **Live Option A `3552 × 3552` Capture** | **Optical Flash Transient (`BLUFF-OT-20261020-01`)** | **Multi-Epoch Moving Tracklet (`BLUFF-TRK-02`)** |
| ![Live Option A Capture](docs/showcase/option_a_live_frame.jpg) | ![Optical Transient](docs/showcase/transient_bluff-ot-20261020-01.jpg) | ![Moving Tracklet](docs/showcase/tracklet_bluff-trk-02.jpg) |

### 5. Earth-Orbital Space Situational Awareness (GEO Clarke Belt, LEO Passes & Meteor Storms)
| Geostationary "Clarke Belt" Necklace (`41` GEO Satellites at `35,786 km`) | SpaceMobile BlueBird (`NORAD #69591`) & LEO Overpass |
| :---: | :---: |
| ![Geostationary Clarke Belt](docs/showcase/orb_geo_clarke_belt_annotated.jpg) | ![SpaceMobile LEO Overpass](docs/showcase/orb_leo_spacemobile_pass_annotated.jpg) |
| **Multi-Night Orionid Fireball & Bolide Storm (`22` Meteors)** | **Airplane-Gated Circumpolar Polaris Star-Trail Vortex (`156 × 30s`)** |
| ![Orionid Fireball Storm](docs/showcase/orb_orionid_fireball_storm_annotated.jpg) | ![Polaris Star-Trail Vortex](docs/showcase/orb_polaris_startrail_vortex_stack.jpg) |

### 6. Solar System Ephemeris Shift-and-Stack (Out to `2.69 Billion Miles`!) & Super-Resolution Deep-Sky Portraits
| Ice Giant Neptune (`mag +7.82`, `2.69B mi` / `28.99 AU`) | Ice Giant Uranus (`mag +5.61`, `1.73B mi` / `18.65 AU`) | Ringed Giant Saturn (`787M mi` / `8.47 AU`) |
| :---: | :---: | :---: |
| ![Neptune Shift-and-Stack](docs/showcase/sol_neptune_annotated.jpg) | ![Uranus 16-Day Track](docs/showcase/sol_uranus_annotated.jpg) | ![Saturn 16-Day Track](docs/showcase/sol_saturn_annotated.jpg) |
| **Andromeda Galaxy (`M31`, `M32` & `M110` — `2.54 Mly`)** | **Bode's (`M81`) & Cigar (`M82`) Galaxies (`11.8 Mly`)** | **Great Orion Nebula (`M42` Annotated Plate)** |
| ![Andromeda Galaxy M31](docs/showcase/m31_andromeda_stack.jpg) | ![M81 and M82 Galaxies](docs/showcase/m81_m82_galaxies_annotated.jpg) | ![Orion Nebula M42](docs/showcase/m42_orion_annotated.jpg) |
| **Crab Nebula Supernova Remnant (`M1` — SN 1054)** | **Rosette Nebula (`NGC 2237`) & Christmas Tree** | **6-DOF Astrometric Star Solver (`camera_realign.py`)** |
| ![Crab Nebula M1](docs/showcase/m1_crab_annotated.jpg) | ![Rosette Nebula](docs/showcase/ngc2237_rosette_stack.jpg) | ![6-DOF Star Solver](docs/showcase/dome_stars_overlay_3d.jpg) |

> 📖 **[Browse all 51 full-resolution plates in `docs/SHOWCASE.md` →](docs/SHOWCASE.md)**

---

## 💻 Hardware Requirements

### Part A: Raspberry Pi Camera Capture Node Requirements (`1` to `N` Nodes)

| Component | Minimum Requirement | Recommended (Observatory Grade) | Notes |
| :--- | :--- | :--- | :--- |
| **Single-Board Computer** | **Raspberry Pi 4B (2 GB RAM)** | **Raspberry Pi 5 (4 GB or 8 GB RAM)** | Option A uses only **~235 MB RAM** and **~12% CPU** on Pi 5. |
| **Operating System** | **Raspberry Pi OS (64-bit / `aarch64`)** | **Raspberry Pi OS Bookworm (64-bit Lite or Desktop)** | SSH enabled (`sudo raspi-config` → Interface Options → SSH). |
| **Camera Sensor** | Any USB 3.0 **ZWO ASI** camera (`ASI178MC`, `ASI224MC`, `ASI662MC`, `ASI678MC`) | **ZWO ASI676MC** (`3552×3552` square 1:1 fisheye, Sony Starvis 2 IMX676) | Plug into the blue **USB 3.0** port on the Raspberry Pi. |
| **Lens** | 180° All-Sky Fisheye Lens (1.55 mm – 2.5 mm CS/C-mount) | **1.55 mm – 1.85 mm f/1.4–f/2.0 180° Fisheye** | Provides full horizon-to-horizon (`360° × 180°`) hemisphere coverage. |
| **Storage** | 32 GB MicroSD (Class 10 / A2) | **64 GB – 128 GB High-Endurance MicroSD or NVMe SSD** | Live frames are buffered in RAM (`/dev/shm`), reducing SD card wear by >90%. |
| **Network** | Wi-Fi (2.4 / 5 GHz) | **5 GHz Wi-Fi or Gigabit Ethernet (PoE+ HAT)** | NTP time sync (`systemd-timesyncd` or `chrony`) keeps multi-camera shutters locked within `< 2 ms`. |

### Part B: Workstation / Server Requirements (3 Tiers)

Because the core Stage-1 mathematical pipeline, SGP4 orbital propagator, ADS-B matcher, 3D stereo triangulator, and 11-D k-NN active learner run in vectorized `numpy`/`OpenCV`, **a dedicated GPU is optional**:

| Tier | Target Hardware | CPU & RAM | GPU / VRAM | Capabilities Unlocked on the Workstation |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Entry / CPU-Only** *(Mini-PC, Laptop, NAS, or Mac)* | Intel N100 / Core i3 / Ryzen 3 / Apple M1 | 4+ Cores, **4 GB RAM** min (`8 GB` recommended) | **None required** (`--cpu-only` mode: `vlm_enabled: false`, `day_enable_yolo: false`) | ✅ Full `1..N` Camera Night & Day Anomaly Detection (`~180 ms/frame`)<br>✅ 3D Multi-Camera Stereo Triangulation<br>✅ SGP4 Satellite & Rocket Body Tagging (14,000+ objects)<br>✅ ADS-B Aircraft Matching (Local SDR or Public API)<br>✅ 11-D k-NN Active Learning + Web Observatory (`:8094`)<br>✅ CPU-fallback Deep-Sky Stacking |
| **Tier 2: Enthusiast GPU** *(Gaming PC, Home Server, or Mac Studio)* | Core i5/i7 / Ryzen 5/7 or Apple Silicon M1–M4 | 6+ Cores, **16 GB RAM** | **NVIDIA GPU with 8 GB – 16 GB VRAM** (e.g., RTX 3060 / 4060 / 4070) or **Apple Unified Memory (16 GB+)** | Everything in Tier 1, **plus**:<br>✅ Stage-2 Local Vision LLM Verification via Ollama / LM Studio (`7B–12B` VLM like `Qwen2.5-VL-7B` or `Gemma3-12B`)<br>✅ Daytime YOLO11 Bird / Aircraft Detection<br>✅ CUDA-Accelerated 4K Milky Way & Messier Deep-Sky Stacking |
| **Tier 3: Flagship Observatory** *(Workstation Server)* | Multi-core Xeon / Threadripper / Core i9 / Ryzen 9 | 8+ Cores, **32 GB – 64+ GB RAM** | **24 GB – 48+ GB VRAM** (e.g., RTX 3090 / 4090 / RTX A6000 / **Quadro RTX 8000 48 GB**) | Everything in Tier 1 & 2, **plus**:<br>✅ Frontier-scale Local Vision LLMs (`Qwen3.5-122B-A10B` / `72B` VLMs)<br>✅ Sub-second `3552×3552` 2.5× Super-Resolution Drizzle Co-Addition across 400+ frames<br>✅ Zero-latency parallel processing for large `1..N` camera arrays |

---

## 🚀 Complete Step-by-Step Installation Guide

> **Where do I start?** Start on your **Workstation / Server**. You only need to clone this repository onto your Workstation — it includes an automated deployer (`deploy_to_pi.py`) that pushes the Raspberry Pi capture daemon directly to your `1..N` Raspberry Pis over SSH! *(Manual Raspberry Pi setup instructions are also included in Part 2B below).*

---

### 🖥️ PART 1: Workstation / Server Setup (Run on Your PC, Server, or Mac)

#### 1.1 Clone the Repository & Install Workstation Python Dependencies
Open a terminal (PowerShell on Windows, or bash/zsh on Linux/macOS) **on your Workstation**:
```bash
git clone https://github.com/bretStaz/bluffsky-allsky-observatory.git
cd bluffsky-allsky-observatory

python -m venv venv
# Activate the virtual environment:
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
# source venv/bin/activate

# Install Workstation dependencies:
pip install -r requirements.txt
```
*(Optional GPU acceleration)*: If your Workstation has an NVIDIA GPU and you want CUDA-accelerated deep-sky stacking and daytime YOLO11 bird/aircraft filtering, also install PyTorch with CUDA (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` and `pip install ultralytics`).

#### 1.2 Generate Your Observatory Configuration (`anomaly_detector/config.json`)
Still **on your Workstation**, run `setup_observatory.py` with the IP address(es) of your `1..N` Raspberry Pi(s) and your observatory's GPS latitude/longitude:
```bash
# Example A — Single Raspberry Pi Camera + Local GPU VLM enabled:
python setup_observatory.py --camera-ips 192.168.1.100 --lat 45.0000 --lon -100.0000

# Example B — Dual / Multi-Camera Stereo Array (1..N Pis) + CPU-Only Mode:
python setup_observatory.py --camera-ips 192.168.1.100,192.168.1.101 --lat 45.0000 --lon -100.0000 --cpu-only

# Example C (Optional) — Also enable co-located 1090 MHz ADS-B SDR receivers on your Camera Pi(s) (default: off):
python setup_observatory.py --camera-ips 192.168.1.100,192.168.1.101 --lat 45.0000 --lon -100.0000 --enable-pi-adsb
```
This creates `anomaly_detector/config.json` configured for all of your cameras.

---

### 🍓 PART 2: Raspberry Pi Capture Node Setup (`1..N` Pis)

Each Raspberry Pi connected to a ZWO ASI camera only needs the lightweight **`pi_capture_node/`** daemon installed to `/opt/bluffsky-node`. You can install it either **automatically from your Workstation (Method 2A)** or **manually on each Raspberry Pi (Method 2B)**.

#### Method 2A (Recommended): Automated Remote Push from Your Workstation
Make sure your ZWO ASI camera is plugged into a blue USB 3.0 port on your Raspberry Pi and SSH is enabled on the Pi. Then run this single command **on your Workstation**:
```bash
# Deploy to all 1..N Raspberry Pis listed in anomaly_detector/config.json (ADS-B off by default unless --enable-adsb is passed):
python pi_capture_node/deploy_to_pi.py --all-from-config --user pi

# Or deploy to a single Raspberry Pi IP (prompts for SSH password if --password is omitted):
python pi_capture_node/deploy_to_pi.py --host 192.168.1.100 --user pi --node-id bluffsky-pi1

# OPTIONAL: If you plugged a 1090 MHz USB RTL-SDR dongle into a black USB 2.0 port on your Camera Pi(s),
# add --enable-adsb to auto-install readsb + tar1090 and serve live ADS-B telemetry on :8096/api/adsb:
python pi_capture_node/deploy_to_pi.py --all-from-config --user pi --enable-adsb

# (You can turn co-located ADS-B back off at any time with --disable-adsb):
python pi_capture_node/deploy_to_pi.py --all-from-config --user pi --disable-adsb
```
**What `deploy_to_pi.py` automatically does on each Raspberry Pi:**
1. Creates `/opt/bluffsky-node` on the Pi and uploads `bluffsky_pi_node.py`, `node_config.json`, `bluffsky-node.service`, and `switch_capture_mode.sh`.
2. Installs ZWO USB `udev` rules (`/etc/udev/rules.d/99-asi.rules`) so the camera can be accessed without root privileges.
3. Locates (or uploads via `--asi-sdk-lib`) `libASICamera2.so`.
4. If the Pi was previously running legacy `allsky.service`, cleanly stops it while preserving it intact as a 1-command backup (`sudo /opt/bluffsky-node/switch_capture_mode.sh legacy-backup`).
5. Enables and starts **`bluffsky-node.service`** so the Raspberry Pi automatically captures and serves frames on port `:8096` on every boot.

---

#### Method 2B (Alternative): Manual Installation Directly on the Raspberry Pi
If you prefer to set up the Raspberry Pi yourself by logging directly into the Pi over SSH (`ssh pi@<pi-ip>`), run these commands **on the Raspberry Pi**:
```bash
# 1. Install system packages & Python dependencies on the Raspberry Pi:
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv python3-numpy python3-pil python3-ephem libusb-1.0-0

# 2. Copy the pi_capture_node/ folder to /opt/bluffsky-node on the Raspberry Pi:
sudo mkdir -p /opt/bluffsky-node
# (Copy bluffsky_pi_node.py, node_config.json, bluffsky-node.service, switch_capture_mode.sh into /opt/bluffsky-node)

# 3. Copy ZWO's ARM64 shared library (libASICamera2.so) into /opt/bluffsky-node/libASICamera2.so
#    and install the ZWO USB udev rule:
echo 'ACTION=="add", ATTR{idVendor}=="03c3", RUN+="/bin/sh -c '\''/bin/echo 200 > /sys/module/usbcore/parameters/usbfs_memory_mb'\''", MODE="0666"' | sudo tee /etc/udev/rules.d/99-asi.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# 4. Edit /opt/bluffsky-node/node_config.json to set your latitude_deg and longitude_deg

# 5. Install and start the systemd background service on the Raspberry Pi:
sudo cp /opt/bluffsky-node/bluffsky-node.service /etc/systemd/system/bluffsky-node.service
sudo systemctl daemon-reload
sudo systemctl enable --now bluffsky-node.service
```

#### Verify Your Raspberry Pi Capture Node Is Running
From any browser on your network, check your Raspberry Pi's built-in `:8096` endpoints:
- **Live Telemetry JSON**: `http://<pi-ip>:8096/api/status` *(shows exposure regime `DAY`/`DARK_NIGHT`, sensor temp, gain, exposure time, and frame counter)*
- **Live Camera Frame**: `http://<pi-ip>:8096/current/tmp/image.jpg`

---

### 🔭 PART 3: Starting the Analysis Engine & Web Observatory (Run on Your Workstation)

Once your Raspberry Pi(s) are serving images on `:8096`, run the analysis & web observatory suite **on your Workstation**:

#### 3.1 Start the Detector Daemon & Web Observatory (`:8094`)
On your **Workstation**:
```bash
python anomaly_detector/detector.py --daemon
```
> **Note**: Running `detector.py --daemon` on your Workstation automatically starts **both**:
> 1. The real-time `1..N` camera anomaly detector, plane tracker, satellite tracker, and stereo correlator.
> 2. The **Interactive Web Observatory Portal (`web_portal.py`) on port `8094`**.
>
> *(On Windows, you can also use `anomaly_detector\start_detector.bat` and `anomaly_detector\run_guard.bat` to keep the services running in the background).*

#### 3.2 Open the Web Observatory Interface on Your Workstation
Open **`http://localhost:8094`** (or `http://<workstation-hostname>:8094` from any phone/tablet/PC on your LAN):

| Web Observatory URL | What You Can View & Do |
| :--- | :--- |
| **`http://localhost:8094/`** | **Main Sky Event Dashboard**: Real-time meteor, bolide fireball, satellite flare, aircraft, and optical transient cards; `1..N` camera filter (`🎥 All Cameras`); `🤝 STEREO COINCIDENCE` 3D altitude badges; 1-click human reclassification buttons (trains the 11-D k-NN active learner); and **`🧭 Camera Moved — Realign`** button. |
| **`http://localhost:8094/live`** | **Live Multi-Camera Sky Dome Radar**: Live fisheye camera feed with camera selector (`#liveCameraSelect`), real-time ADS-B aircraft flight vectors, SGP4 satellite pass arcs, and constellation overlays. |
| **`http://localhost:8094/discoveries`** | **Deep-Space Transients & Tracklets**: Sub-arcminute optical flashes (`BLUFF-OT`), multi-epoch moving object tracklets (`BLUFF-TRK`), variable star photometry, and interactive J2000 reference sky blink comparator. |
| **`http://localhost:8094/astrophoto`** | **`4K` Milky Way, Deep-Sky, Solar System & Orbital Showcase**: Interactive gallery of the `4K` Milky Way panorama, 22 super-resolution (`2.5×` drizzle) Messier/NGC deep-sky stacks, 5 Solar System planets (out to Neptune), and Earth-orbital composites with split-screen `Single 30s ⇄ Deep Stack` comparison sliders and `4000%` zoom lightbox. |

#### 3.3 First-Night Astrometric Calibration (1 Click)
On your first clear starry night, open `http://localhost:8094` and click **`🧭 Camera Moved — Realign`** in the top header. The Workstation's `camera_realign.py` solver will match the live stars in your camera's image against the Yale Bright Star catalog and automatically calibrate your exact zenith pixel center, lens distortion (`k2`, `k3`), 2-axis camera tilt, and compass azimuth.

---

## 🧭 Automated Camera-to-GPS 6-DOF Star-Field Alignment (`camera_realign.py`)

When you mount an outdoor all-sky fisheye camera, you don't need a compass, bubble level, or manual star-click tool. You only enter your **GPS Latitude & Longitude** (`camera_lat`, `camera_lon`) once, and [`camera_realign.py`](anomaly_detector/camera_realign.py) automatically solves the exact 3D geometric relationship between your camera sensor and the Earth/Sky coordinate frame:

```text
GPS (lat, lon, elev) + Frame UTC Timestamp
      │
      ▼
1. Ephemeris Sky Truth (pyephem)
   Computes exact topocentric (East, North, Up) unit vectors for 36 J2000 navigation
   stars (Polaris, Vega, Deneb, Capella, Sirius, Rigel, etc.) + bright planets above
   24° altitude, automatically excluding stars within 18° of the Moon.
      │
      ▼
2. Sub-Pixel Fisheye Star Centroiding (extract_star_centroids_3552)
   25×25 Gaussian background subtraction + intensity-squared (I²) center-of-mass
   centroiding isolates the top 75 pinpoint stars while masking horizon trees/clouds.
      │
      ▼
3. Blind 360° 3D Spherical Grid Search
   Sweeps all compass orientations (0°..360° in 2° steps) and 2-axis mount tilts
   (-12°..+12° in 3° steps) so the camera can be mounted at any arbitrary rotation.
      │
      ▼
4. 4-Stage Huber-Loss 6-DOF Refinement (~0.11° / ~4 px RMS)
   Coarse-to-fine inlier gates (34 px → 20 px → 12 px → 9 px) jointly optimize:
   • bottom_azimuth_deg   (Exact compass rotation of the sensor to ±0.05°)
   • tilt_x_deg, tilt_y_deg (2-axis physical level tilt of the camera housing)
   • zenith_center [cx, cy] (True optical axis zenith pixel on the sensor)
   • horizon_radius_px      (Zenith-to-horizon scale of the fisheye lens)
   • radial_k2, radial_k3   (5th-order polynomial fisheye distortion: r = R*(u + k2*u³ + k3*u⁵))
      │
      ▼
5. Multi-Frame Consensus Lock + Daytime Horizon Mask Rebuild + Epoch History
   • Requires 2 independent clear frames to agree within ≤ 0.35° before locking.
   • Automatically rebuilds the daytime horizon tree/roofline sky mask (sky_mask.png).
   • Archives the prior calibration in `calibration_epochs` (`BeforeNight`) so
     historical nights captured before the camera was moved retain their alignment!
```

---

## 📂 Repository Structure (Grouped by Machine)

```text
bluffsky-allsky-observatory/
├── README.md                              # Architecture, visual showcase & step-by-step setup guide
├── LICENSE                                # MIT License
├── docs/
│   ├── SHOWCASE.md                        # Complete 51-image high-resolution visual showcase
│   └── showcase/                          # 51 real captures, bolide detections & deep-sky stacks
│
├── pi_capture_node/                       # 🍓 RUNS ON EACH RASPBERRY PI (1..N Camera Nodes)
│   ├── bluffsky_pi_node.py                # Direct ZWO ASI SDK capture daemon + 4-regime controller + :8096 server
│   ├── node_config.json                   # Per-node exposure, gain, GPS & stereo sync configuration
│   ├── bluffsky-node.service              # systemd service unit for Raspberry Pi OS (/opt/bluffsky-node)
│   ├── switch_capture_mode.sh             # Instant 1-command toggle between option-a and legacy-backup on Pi
│   ├── requirements.txt                   # Lightweight Raspberry Pi Python dependencies
│   ├── deploy_to_pi.py                    # (Run from Workstation) Automated SSH/SFTP deployer to 1..N Pis
│   └── README.md                          # Detailed Raspberry Pi node reference
│
├── setup_observatory.py                   # 🖥️ RUNS ON WORKSTATION: 1-command config wizard for 1..N cameras
├── requirements.txt                       # 🖥️ RUNS ON WORKSTATION: Server Python dependencies
└── anomaly_detector/                      # 🖥️ RUNS ON WORKSTATION: Analysis, Stereo Correlator & Web Portal
    ├── config.example.json                # Sanitized 1..N camera configuration template
    ├── detector.py                        # Real-time 1..N camera watcher, Stage-1/2 VLM & 3D stereo correlator
    ├── web_portal.py                      # Interactive Observatory Web Portal (:8094)
    ├── allsky_workflow_guard.py           # Automated health monitor & self-healing watchdog for 1..N nodes
    ├── adsb_tracker.py                    # Live ADS-B aircraft dome projection & flight-path matcher
    ├── orbital_tracker.py                 # SGP4 satellite, rocket body & deep-space glint matcher
    ├── feedback_learner.py                # Active learning 11-D k-NN + Discord reaction/reply bot
    ├── camera_realign.py                  # Automated 6-DOF star-field astrometric solver
    ├── deep_sky_survey.py                 # Multi-epoch J2000 co-addition & moving tracklet discovery
    ├── deep_sky_astrophoto.py             # 4K Milky Way & 22-object super-resolution deep-sky engine
    ├── orbital_and_planetary_showcase.py  # Solar System shift-and-stack & Earth-orbital composite engine
    └── astrophysics_patrol.py             # Automated astrophysical transient patrol & cataloger
```

---

## 📐 Multi-Camera Stereo Baseline Offsets (`east_offset_m`, `north_offset_m`, `up_offset_m`)

- For local property setups ($5\text{ m}$ to $100\text{ m}$ separation), set `cam1` to `(0, 0, 0)` and enter the East, North, and Up distance in meters from `cam1` to `cam2..camN` in `anomaly_detector/config.json` (or set distinct `camera_lat` / `camera_lon` for wide-baseline kilometer separations).
- When two or more cameras detect an event within $|\Delta t| \le 12\text{ s}$, `match_multi_camera_stereo()` on the Workstation automatically computes:
  - **Angular Parallax ($\Delta\theta$)**: Vetoes dome bugs (`< 3 m`), birds (`15–150 m`), and single-sensor cosmic-ray hits (`100%` rejected).
  - **3D Triangulated Altitude & Slant Range**: Displayed directly on the Web Observatory event card (`🤝 STEREO COINCIDENCE`) and in the AI Sky Analysis description.
