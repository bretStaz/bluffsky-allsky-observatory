import asyncio
from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import re
import sqlite3
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "allsky_detections.db")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
DISCORD_CONFIG_FILE = r"E:\discord_messages\config.json"
BOT_LOG_FILE = os.path.join(BASE_DIR, "discord_bot.log")
BOT_STATUS_FILE = os.path.join(BASE_DIR, "discord_bot_status.json")

logger = logging.getLogger("AllskyFeedbackLearner")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = RotatingFileHandler(BOT_LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

DB_LOCK = threading.Lock()

# Canonical categories and their default human-friendly subjects
CANONICAL_LABELS: Dict[str, Tuple[str, str]] = {
    "meteor": ("meteor", "Shooting Star / Meteor Streak"),
    "fireball": ("fireball", "Bright Fireball / Bolide Streak"),
    "space_object": ("space_object", "LEO Satellite / Space Object"),
    "aircraft": ("aircraft", "Night Aircraft Strobe Trail"),
    "transient_flash": ("transient_flash", "Optical Space Flash / Glint"),
    "sky_phenomenon": ("sky_phenomenon", "Luminous Sky Phenomenon"),
    "day_aircraft": ("day_aircraft", "Daytime Aircraft / Contrail"),
    "day_bird": ("day_bird", "Soaring Bird / Wildlife"),
    "day_anomaly": ("day_anomaly", "Daytime Sky Anomaly"),
    "false_positive": ("false_positive", "False Positive / Non-Anomaly"),
}

CATEGORY_ICONS: Dict[str, str] = {
    "meteor": "🌠",
    "fireball": "🔥",
    "space_object": "🛰️",
    "aircraft": "✈️",
    "transient_flash": "✨",
    "sky_phenomenon": "🌌",
    "day_aircraft": "🛩️",
    "day_bird": "🦅",
    "day_anomaly": "☀️",
    "false_positive": "❌",
}

EMOJI_TO_CATEGORY: Dict[str, str] = {
    "☄️": "meteor",
    "☄": "meteor",
    "🌠": "meteor",
    "🔥": "fireball",
    "💥": "fireball",
    "🛰️": "space_object",
    "🛰": "space_object",
    "🚀": "space_object",
    "🛸": "space_object",
    "✈️": "aircraft",
    "✈": "aircraft",
    "🛩️": "aircraft",
    "🛩": "aircraft",
    "🛫": "aircraft",
    "✨": "transient_flash",
    "⚡": "transient_flash",
    "🌌": "sky_phenomenon",
    "🌈": "sky_phenomenon",
    "🦅": "day_bird",
    "🦉": "day_bird",
    "🐦": "day_bird",
    "❌": "false_positive",
    "🗑️": "false_positive",
    "🗑": "false_positive",
    "👎": "false_positive",
    "🚫": "false_positive",
}


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE, timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_feedback_schema() -> None:
    """Ensures `sky_feedback` table and feedback columns on `sky_events` exist."""
    with DB_LOCK:
        conn = get_db_connection()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sky_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_filename TEXT NOT NULL,
                    mode TEXT DEFAULT 'NIGHT',
                    original_category TEXT NOT NULL,
                    corrected_category TEXT NOT NULL,
                    corrected_subject TEXT,
                    user_note TEXT,
                    source TEXT DEFAULT 'discord',
                    discord_user TEXT,
                    features_json TEXT,
                    bbox_json TEXT,
                    center_x REAL DEFAULT 0.0,
                    center_y REAL DEFAULT 0.0,
                    alt_deg REAL DEFAULT 0.0,
                    az_deg REAL DEFAULT 0.0,
                    snr REAL DEFAULT 0.0,
                    length_px REAL DEFAULT 0.0,
                    aspect REAL DEFAULT 1.0,
                    linearity REAL DEFAULT 0.0,
                    fill_ratio REAL DEFAULT 1.0,
                    strobe_peaks INTEGER DEFAULT 0,
                    deep_gaps INTEGER DEFAULT 0,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sky_feedback_frame ON sky_feedback(frame_filename);
                CREATE INDEX IF NOT EXISTS idx_sky_feedback_cat ON sky_feedback(corrected_category);
                CREATE TABLE IF NOT EXISTS processed_discord_actions (
                    action_key TEXT PRIMARY KEY,
                    created_at REAL NOT NULL
                );
                """
            )
            # Clean up any historic duplicate rows per frame_filename in sky_feedback
            conn.execute(
                """
                DELETE FROM sky_feedback
                WHERE id NOT IN (
                    SELECT MAX(id) FROM sky_feedback GROUP BY frame_filename
                )
                """
            )
            # Check if sky_events exists before altering
            tbl = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sky_events'").fetchone()
            if tbl:
                cols = {r["name"] for r in conn.execute("PRAGMA table_info(sky_events)").fetchall()}
                if "discord_message_id" not in cols:
                    conn.execute("ALTER TABLE sky_events ADD COLUMN discord_message_id TEXT DEFAULT '';")
                if "user_corrected_category" not in cols:
                    conn.execute("ALTER TABLE sky_events ADD COLUMN user_corrected_category TEXT DEFAULT '';")
                if "user_feedback_note" not in cols:
                    conn.execute("ALTER TABLE sky_events ADD COLUMN user_feedback_note TEXT DEFAULT '';")
                if "features_json" not in cols:
                    conn.execute("ALTER TABLE sky_events ADD COLUMN features_json TEXT DEFAULT '';")
            conn.commit()
        finally:
            conn.close()


def claim_discord_action(action_key: str) -> bool:
    """
    Atomically claims a Discord reply or reaction ID in SQLite (`processed_discord_actions`)
    across all threads and processes. Returns True ONLY for the very first caller.
    """
    if not action_key:
        return False
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE, timeout=15, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "INSERT OR IGNORE INTO processed_discord_actions (action_key, created_at) VALUES (?, ?)",
                (str(action_key), time.time()),
            )
            claimed = cur.rowcount == 1
            conn.execute("COMMIT")
            return claimed
        except Exception as e:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            logger.warning(f"claim_discord_action error for {action_key}: {e}")
            return False
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Natural Language & Emoji Correction Parser
# ---------------------------------------------------------------------------
def parse_correction_intent(text: str, mode: str = "NIGHT") -> Optional[Tuple[str, str]]:
    """
    Parses a user's Discord reply text (or single word/emoji) into (canonical_category, custom_subject_note).
    Supports natural phrases like:
      - "plane", "that's an airplane", "jet", "aircraft strobe", "helicopter"
      - "meteor", "shooting star"
      - "fireball", "bolide"
      - "satellite", "starlink", "space object", "iss", "rocket"
      - "flash", "glint", "iridium flare"
      - "aurora", "northern lights", "lightning"
      - "bird", "owl", "bat", "hawk"
      - "false", "noise", "cloud", "glare", "roof", "tree", "bug", "nothing", "wrong"
    """
    if not text:
        return None
    raw = text.strip()
    low = raw.lower()

    # Check direct emoji match first
    for em, cat in EMOJI_TO_CATEGORY.items():
        if em in raw:
            if cat == "aircraft" and mode == "DAY":
                cat = "day_aircraft"
            return CANONICAL_LABELS[cat][0], CANONICAL_LABELS[cat][1]

    # Ordered keyword patterns (most specific first)
    patterns: List[Tuple[str, str, str]] = [
        (r"\b(fireball|bolide|bright\s+meteor)\b", "fireball", "Bright Fireball / Bolide Streak"),
        (r"\b(shooting\s*star|meteor|meteorite|perseid|geminid|orionid)\b", "meteor", "Shooting Star / Meteor Streak"),
        (
            r"\b(false|false\s*positive|not\s+an?\s+anomaly|nothing|noise|hot\s*pixel|cloud|clouds|glare|lens\s*flare|flare\s+from\s+house|roof|house|window|tree|branch|bug|insect|spider|web|junk|trash|delete|hide|ignore)\b",
            "false_positive",
            f"False Positive ({raw[:40]})",
        ),
        (
            r"\b(plane|airplane|aeroplane|aircraft|jet|airliner|cessna|helicopter|chopper|strobe|wingtip|contrail)\b",
            "day_aircraft" if mode == "DAY" else "aircraft",
            "Daytime Aircraft / Contrail" if mode == "DAY" else "Night Aircraft Strobe Trail",
        ),
        (
            r"\b(satellite|starlink|iss|space\s*station|space\s*object|orbital|rocket|booster|debris)\b",
            "space_object",
            "LEO Satellite / Space Object",
        ),
        (
            r"\b(glint|flash|optical\s*flash|geostationary|tumbling)\b",
            "transient_flash",
            "Optical Space Flash / Glint",
        ),
        (
            r"\b(aurora|northern\s*lights|steve|lightning|sprite|sky\s*phenomenon)\b",
            "sky_phenomenon",
            "Luminous Sky Phenomenon",
        ),
        (
            r"\b(bird|hawk|eagle|owl|bat|goose|duck|gull|crow|raven|wildlife)\b",
            "day_bird",
            "Bird / Flying Wildlife",
        ),
    ]

    for pat, cat, subj in patterns:
        if re.search(pat, low):
            return cat, subj

    return None


# ---------------------------------------------------------------------------
# Feature Vector Extraction & Active Learning Classifier (k-NN + Hotspot + Few-Shot VLM)
# ---------------------------------------------------------------------------
def extract_feature_vector(cand_or_ev: Dict[str, Any], mode: str = "NIGHT") -> List[float]:
    """
    Converts a detection dictionary into an 11-dimensional normalized numerical feature vector
    for similarity matching and k-NN classification:
      0: log1p(length_px) / 6.0
      1: log1p(width_px) / 4.0
      2: min(aspect, 40.0) / 40.0
      3: linearity (0..1)
      4: fill_ratio (0..1)
      5: min(strobe_peaks, 10) / 10.0
      6: min(deep_gaps, 8) / 8.0
      7: log1p(snr) / 5.0
      8: min(chroma_spread, 80.0) / 80.0
      9: alt_deg / 90.0
     10: 1.0 if NIGHT else 0.0
    """
    length_px = max(0.0, float(cand_or_ev.get("length_px") or 0.0))
    width_px = max(1.0, float(cand_or_ev.get("width_px") or 2.0))
    aspect = max(1.0, float(cand_or_ev.get("aspect") or 1.0))
    linearity = max(0.0, min(1.0, float(cand_or_ev.get("linearity") or 0.0)))
    fill_ratio = max(0.0, min(1.0, float(cand_or_ev.get("fill_ratio") or 1.0)))
    strobe_peaks = max(0, int(cand_or_ev.get("strobe_peaks") or 0))
    deep_gaps = max(0, int(cand_or_ev.get("deep_gaps") or 0))
    snr = max(0.0, float(cand_or_ev.get("snr") or 0.0))
    chroma = max(0.0, float(cand_or_ev.get("chroma_spread") or 0.0))
    alt_deg = max(0.0, min(90.0, float(cand_or_ev.get("alt_deg") or 45.0)))
    ev_mode = str(cand_or_ev.get("mode") or mode).upper()

    return [
        math.log1p(length_px) / 6.0,
        math.log1p(width_px) / 4.0,
        min(aspect, 40.0) / 40.0,
        linearity,
        fill_ratio,
        min(strobe_peaks, 10) / 10.0,
        min(deep_gaps, 8) / 8.0,
        math.log1p(snr) / 5.0,
        min(chroma, 80.0) / 80.0,
        alt_deg / 90.0,
        1.0 if ev_mode == "NIGHT" else 0.0,
    ]


# Feature importance weights for distance calculation (emphasizing continuity fill_ratio, strobe_peaks, deep_gaps, length, linearity)
FEATURE_WEIGHTS = np.array([1.5, 0.7, 1.1, 1.6, 2.4, 2.2, 2.2, 1.0, 0.8, 0.8, 2.5], dtype=np.float32)


def compute_feature_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Returns similarity in [0.0, 1.0] between two 11-D feature vectors."""
    a = np.array(vec1, dtype=np.float32)
    b = np.array(vec2, dtype=np.float32)
    if a.shape != b.shape:
        return 0.0
    diff = (a - b) * FEATURE_WEIGHTS
    dist = float(np.linalg.norm(diff))
    return float(math.exp(-0.85 * dist))


def get_all_feedback_samples() -> List[Dict[str, Any]]:
    ensure_feedback_schema()
    with DB_LOCK:
        conn = get_db_connection()
        try:
            rows = conn.execute("SELECT * FROM sky_feedback ORDER BY created_at DESC LIMIT 500").fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def evaluate_candidate_with_feedback(cand: Dict[str, Any], mode: str = "NIGHT") -> Dict[str, Any]:
    """
    Uses accumulated human ground-truth feedback to:
      1. Suppress recurrent spatial false-positive hotspots (e.g. roof glare, tree branch, persistent artifact).
      2. Run weighted k-NN classification over human-corrected feature vectors.
      3. Retrieve the top-K most similar human-corrected examples for Stage-2 VLM few-shot prompt injection.
    Returns a dict:
      {
        "veto_as_false_positive": bool,
        "veto_reason": str,
        "suggested_category": Optional[str],
        "suggested_subject": Optional[str],
        "knn_confidence": float,
        "similar_examples": List[dict],
      }
    """
    samples = get_all_feedback_samples()
    if not samples:
        return {
            "veto_as_false_positive": False,
            "veto_reason": "",
            "suggested_category": None,
            "suggested_subject": None,
            "knn_confidence": 0.0,
            "similar_examples": [],
        }

    cand_vec = extract_feature_vector(cand, mode=mode)
    cx, cy = cand.get("center", (0.0, 0.0))
    cand_snr = float(cand.get("snr") or 0.0)

    # 1. Check Spatial False-Positive Hotspots (if user marked a spot as false_positive at the same pixel region)
    fp_spatial_hits = 0
    for s in samples:
        if s.get("corrected_category") == "false_positive" and s.get("mode") == mode:
            sx, sy = float(s.get("center_x") or 0.0), float(s.get("center_y") or 0.0)
            if sx > 0 and sy > 0 and math.hypot(cx - sx, cy - sy) <= 95.0:
                fp_spatial_hits += 1

    if fp_spatial_hits >= 1 and cand_snr < 65.0:
        return {
            "veto_as_false_positive": True,
            "veto_reason": f"Matched user-marked false-positive spatial hotspot near ({int(cx)}, {int(cy)})",
            "suggested_category": "false_positive",
            "suggested_subject": "False Positive Hotspot",
            "knn_confidence": 0.92,
            "similar_examples": [],
        }

    # 2. Score similarity against all feedback samples of the same mode
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for s in samples:
        if s.get("mode", "NIGHT") != mode:
            continue
        try:
            s_vec = json.loads(s["features_json"]) if s.get("features_json") else extract_feature_vector(s, mode=mode)
        except Exception:
            s_vec = extract_feature_vector(s, mode=mode)
        sim = compute_feature_similarity(cand_vec, s_vec)
        scored.append((sim, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_k = scored[:5]

    similar_examples = []
    for sim, s in top_k[:4]:
        if sim >= 0.45:
            similar_examples.append({
                "similarity": round(sim, 2),
                "frame_filename": s.get("frame_filename"),
                "original_category": s.get("original_category"),
                "corrected_category": s.get("corrected_category"),
                "corrected_subject": s.get("corrected_subject"),
                "length_px": s.get("length_px"),
                "aspect": s.get("aspect"),
                "linearity": s.get("linearity"),
                "fill_ratio": s.get("fill_ratio"),
                "strobe_peaks": s.get("strobe_peaks"),
                "deep_gaps": s.get("deep_gaps"),
                "snr": s.get("snr"),
                "alt_deg": s.get("alt_deg"),
                "cardinal": s.get("user_note") or "",
            })

    # 3. Weighted k-NN Vote if we have high-similarity matches (sim >= 0.76)
    votes: Dict[str, float] = {}
    total_w = 0.0
    for sim, s in top_k:
        if sim < 0.76:
            continue
        w = sim ** 2
        cat = s["corrected_category"]
        votes[cat] = votes.get(cat, 0.0) + w
        total_w += w

    suggested_cat = None
    suggested_subj = None
    knn_conf = 0.0
    if total_w > 0:
        best_cat, best_w = max(votes.items(), key=lambda kv: kv[1])
        knn_conf = best_w / total_w
        # Require strong nearest-neighbor agreement
        if knn_conf >= 0.68 and top_k[0][0] >= 0.80:
            suggested_cat = best_cat
            suggested_subj = CANONICAL_LABELS.get(best_cat, (best_cat, best_cat))[1]

    if suggested_cat == "false_positive" and top_k[0][0] >= 0.85 and cand_snr < 70.0:
        return {
            "veto_as_false_positive": True,
            "veto_reason": f"Learned k-NN classifier matched prior user false-positive correction (sim={top_k[0][0]:.2f})",
            "suggested_category": "false_positive",
            "suggested_subject": "Learned False Positive",
            "knn_confidence": round(knn_conf, 2),
            "similar_examples": similar_examples,
        }

    return {
        "veto_as_false_positive": False,
        "veto_reason": "",
        "suggested_category": suggested_cat,
        "suggested_subject": suggested_subj,
        "knn_confidence": round(knn_conf, 2),
        "similar_examples": similar_examples,
    }


def format_vlm_few_shot_block(similar_examples: List[Dict[str, Any]]) -> str:
    """
    Formats the closest human-corrected examples into a concise prompt block for the Local Vision LLM.
    """
    if not similar_examples:
        return ""
    lines = [
        "\nLEARNED GROUND-TRUTH CORRECTIONS FROM CAMERA OWNER (Use these past human-verified examples from this exact camera to avoid repeating mistakes):"
    ]
    for idx, ex in enumerate(similar_examples, start=1):
        lines.append(
            f"  {idx}. Past Event ({ex['frame_filename']}, similarity={ex['similarity']}): "
            f"length={ex.get('length_px', 0)}px, aspect={ex.get('aspect', 1)}, linearity={ex.get('linearity', 0)}, "
            f"fill_ratio={ex.get('fill_ratio', 1)}, strobe_peaks={ex.get('strobe_peaks', 0)}, gaps={ex.get('deep_gaps', 0)}, "
            f"SNR={ex.get('snr', 0)}σ, Alt={ex.get('alt_deg', 0)}° -> "
            f"AI originally guessed '{ex['original_category']}', Owner corrected to '{ex['corrected_category']}' ({ex['corrected_subject']})."
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Record User Correction & Update Event in Database + History
# ---------------------------------------------------------------------------
def record_user_correction(
    frame_filename: str,
    corrected_category: str,
    user_note: str = "",
    source: str = "discord",
    discord_user: str = "Owner",
) -> Optional[Dict[str, Any]]:
    """
    Applies a user classification correction to `frame_filename`:
      1. Looks up the event in `sky_events`.
      2. Inserts the ground-truth training record + feature vector into `sky_feedback`.
      3. Updates `sky_events` (`category`, `subject`, `user_corrected_category`, `user_feedback_note`, `interest_score`, or `is_hidden=1` if `false_positive`).
      4. Returns a summary dict of the change.
    """
    ensure_feedback_schema()
    if corrected_category not in CANONICAL_LABELS:
        return None

    _, default_subject = CANONICAL_LABELS[corrected_category]

    with DB_LOCK:
        conn = get_db_connection()
        try:
            row = conn.execute(
                "SELECT * FROM sky_events WHERE frame_filename = ?",
                (frame_filename,),
            ).fetchone()
            if not row:
                return None
            ev = dict(row)
            old_category = ev.get("category") or "unknown"
            old_subject = ev.get("subject") or ev.get("subtype") or ""
            mode = ev.get("mode") or "NIGHT"

            # Adjust aircraft label for DAY vs NIGHT
            if corrected_category == "aircraft" and mode == "DAY":
                corrected_category = "day_aircraft"
                _, default_subject = CANONICAL_LABELS[corrected_category]

            # Extract stored features if available, or build from row columns
            feat_dict = {}
            if ev.get("features_json"):
                try:
                    feat_dict = json.loads(ev["features_json"])
                except Exception:
                    feat_dict = {}
            merged_ev = {**ev, **feat_dict}
            f_vec = extract_feature_vector(merged_ev, mode=mode)

            cx, cy = 0.0, 0.0
            bbox_raw = feat_dict.get("bbox")
            if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
                cx = (float(bbox_raw[0]) + float(bbox_raw[2])) / 2.0
                cy = (float(bbox_raw[1]) + float(bbox_raw[3])) / 2.0

            # Avoid duplicate feedback row from the same frame; update latest correction if re-corrected
            existing_fb = conn.execute(
                "SELECT id FROM sky_feedback WHERE frame_filename = ?",
                (frame_filename,),
            ).fetchone()
            if existing_fb:
                conn.execute(
                    """
                    UPDATE sky_feedback SET
                        corrected_category = ?,
                        corrected_subject = ?,
                        user_note = ?,
                        source = ?,
                        discord_user = ?,
                        features_json = ?,
                        created_at = ?
                    WHERE id = ?
                    """,
                    (
                        corrected_category,
                        default_subject,
                        user_note,
                        source,
                        discord_user,
                        json.dumps(f_vec),
                        time.time(),
                        existing_fb["id"],
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO sky_feedback (
                        frame_filename, mode, original_category, corrected_category,
                        corrected_subject, user_note, source, discord_user,
                        features_json, bbox_json, center_x, center_y,
                        alt_deg, az_deg, snr, length_px, aspect, linearity,
                        fill_ratio, strobe_peaks, deep_gaps, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        frame_filename,
                        mode,
                        old_category,
                        corrected_category,
                        default_subject,
                        user_note,
                        source,
                        discord_user,
                        json.dumps(f_vec),
                        json.dumps(bbox_raw or []),
                        cx,
                        cy,
                        float(ev.get("alt_deg") or 0.0),
                        float(ev.get("az_deg") or 0.0),
                        float(ev.get("snr") or 0.0),
                        float(ev.get("length_px") or 0.0),
                        float(ev.get("aspect") or 1.0),
                        float(ev.get("linearity") or 0.0),
                        float(feat_dict.get("fill_ratio") or 1.0),
                        int(feat_dict.get("strobe_peaks") or 0),
                        int(feat_dict.get("deep_gaps") or 0),
                        time.time(),
                    ),
                )

            if corrected_category == "false_positive":
                conn.execute(
                    """
                    UPDATE sky_events SET
                        user_corrected_category = 'false_positive',
                        user_feedback_note = ?,
                        is_hidden = 1,
                        is_starred = 0
                    WHERE frame_filename = ?
                    """,
                    (user_note, frame_filename),
                )
                new_score = 0.0
            else:
                import web_portal
                updated_ev = dict(ev)
                updated_ev["category"] = corrected_category
                new_score = web_portal.compute_interest_score(updated_ev)
                # Give human-verified meteors/fireballs/space objects a small verification bonus
                if corrected_category in ("meteor", "fireball", "space_object", "sky_phenomenon"):
                    new_score = min(99.9, round(new_score + 4.0, 1))
                conn.execute(
                    """
                    UPDATE sky_events SET
                        category = ?,
                        subject = ?,
                        subtype = ?,
                        user_corrected_category = ?,
                        user_feedback_note = ?,
                        interest_score = ?,
                        is_hidden = 0
                    WHERE frame_filename = ?
                    """,
                    (
                        corrected_category,
                        default_subject,
                        default_subject,
                        corrected_category,
                        user_note,
                        new_score,
                        frame_filename,
                    ),
                )

            total_samples = conn.execute("SELECT COUNT(*) AS c FROM sky_feedback").fetchone()["c"]
            conn.commit()
        finally:
            conn.close()

    logger.info(
        f"Recorded ground-truth correction for {frame_filename} via {source}: "
        f"{old_category} -> {corrected_category} ({default_subject}) [Total training samples: {total_samples}]"
    )
    return {
        "frame_filename": frame_filename,
        "old_category": old_category,
        "old_subject": old_subject,
        "corrected_category": corrected_category,
        "corrected_subject": default_subject,
        "new_interest_score": new_score,
        "total_training_samples": total_samples,
        "source": source,
    }


def find_event_by_discord_message(
    discord_msg_id: str,
    referenced_msg_obj: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Finds the corresponding `sky_events` row when a user replies to or reacts to a Discord message in `#all-sky`.
    1. First checks `sky_events.discord_message_id == discord_msg_id`.
    2. Next, inspects `referenced_msg_obj` (embeds, content, attachments) for `image-YYYYMMDDHHMMSS*.jpg`.
    """
    ensure_feedback_schema()
    with DB_LOCK:
        conn = get_db_connection()
        try:
            if discord_msg_id:
                row = conn.execute(
                    "SELECT * FROM sky_events WHERE discord_message_id = ?",
                    (str(discord_msg_id),),
                ).fetchone()
                if row:
                    return dict(row)

            if referenced_msg_obj:
                blob = json.dumps(referenced_msg_obj)
                # Look for `image-20261015193804.jpg` or `image-20261015193804_anomaly.jpg`
                m = re.search(r"(image-\d{14}(?:-sim-[a-z]+)?)(?:_anomaly)?\.jpg", blob)
                if m:
                    base_fname = m.group(1) + ".jpg"
                    row = conn.execute(
                        "SELECT * FROM sky_events WHERE frame_filename = ?",
                        (base_fname,),
                    ).fetchone()
                    if row:
                        # Also backfill discord_message_id for future lookups
                        if discord_msg_id:
                            conn.execute(
                                "UPDATE sky_events SET discord_message_id = ? WHERE frame_filename = ?",
                                (str(discord_msg_id), base_fname),
                            )
                            conn.commit()
                        return dict(row)
        finally:
            conn.close()
    return None


def get_feedback_summary() -> Dict[str, Any]:
    ensure_feedback_schema()
    samples = get_all_feedback_samples()
    by_corrected: Dict[str, int] = {}
    confusion: Dict[str, int] = {}
    for s in samples:
        c_cat = s.get("corrected_category") or "unknown"
        o_cat = s.get("original_category") or "unknown"
        by_corrected[c_cat] = by_corrected.get(c_cat, 0) + 1
        if o_cat != c_cat:
            pair = f"{o_cat} → {c_cat}"
            confusion[pair] = confusion.get(pair, 0) + 1

    bot_status = {}
    if os.path.exists(BOT_STATUS_FILE):
        try:
            with open(BOT_STATUS_FILE, "r", encoding="utf-8") as f:
                bot_status = json.load(f)
        except Exception:
            pass

    client_id = str(bot_status.get("bot_user_id") or "")
    guild_id = str(bot_status.get("guild_id") or "")
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                c = json.load(f)
            client_id = client_id or str(c.get("discord_client_id") or "")
            guild_id = guild_id or str(c.get("discord_guild_id") or "")
        except Exception:
            pass
    if client_id:
        bot_status["client_id"] = client_id
        guild_param = f"&guild_id={guild_id}&disable_guild_select=false" if guild_id else ""
        bot_status["invite_url"] = (
            f"https://discord.com/oauth2/authorize?client_id={client_id}&permissions=68672&scope=bot{guild_param}"
        )

    return {
        "total_samples": len(samples),
        "by_corrected_category": by_corrected,
        "corrections_breakdown": confusion,
        "recent_feedback": samples[:40],
        "bot_status": bot_status,
    }


# ---------------------------------------------------------------------------
# Auto-Discover Channel ID & Guild ID from Existing #all-sky Webhook URL
# ---------------------------------------------------------------------------
def discover_channel_from_webhook() -> Dict[str, str]:
    """
    Calls GET on the user's configured `#all-sky` Discord Webhook URL, which returns
    the exact `channel_id` and `guild_id` of the `#all-sky` channel without needing manual setup!
    """
    wh_url = ""
    if os.path.exists(DISCORD_CONFIG_FILE):
        try:
            with open(DISCORD_CONFIG_FILE, "r", encoding="utf-8") as f:
                d_cfg = json.load(f)
            whs = d_cfg.get("webhooks") or {}
            wh_url = whs.get("allsky") or whs.get("all sky") or d_cfg.get("default_webhook_url") or ""
        except Exception:
            pass
    if not wh_url or not wh_url.startswith("https://discord"):
        return {}
    try:
        r = requests.get(wh_url, timeout=6)
        if r.status_code == 200:
            info = r.json()
            return {
                "channel_id": str(info.get("channel_id") or ""),
                "guild_id": str(info.get("guild_id") or ""),
                "webhook_name": str(info.get("name") or ""),
                "application_id": str(info.get("application_id") or ""),
            }
    except Exception as e:
        logger.debug(f"Could not inspect webhook metadata: {e}")
    return {}


def update_bot_status(update: Dict[str, Any]) -> None:
    try:
        cur = {}
        if os.path.exists(BOT_STATUS_FILE):
            try:
                with open(BOT_STATUS_FILE, "r", encoding="utf-8") as f:
                    cur = json.load(f)
            except Exception:
                cur = {}
        cur.update(update)
        cur["updated_at"] = time.time()
        cur["updated_iso"] = datetime.now().isoformat()
        with open(BOT_STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(cur, f, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Hybrid Discord Bot Listener (Real-Time Gateway WebSocket + REST Poll Fallback)
# ---------------------------------------------------------------------------
class DiscordFeedbackBot:
    """
    Listens to `#all-sky` for:
      1. User replies to any Allsky alert image (`message_reference.message_id`)
      2. User emoji reactions on any Allsky alert image (`☄️`, `🔥`, `🛰️`, `✈️`, `❌`, etc.)
      3. Direct commands like `!stats` or `!help` in `#all-sky`
    Uses BOTH Discord Gateway WebSocket (for instant <0.5s feedback) AND periodic REST polling
    of `#all-sky` messages so it works reliably even if Gateway intents are still being configured.
    """

    def __init__(self):
        self.processed_reply_ids: set = set()
        self.processed_reaction_keys: set = set()
        self.bot_user_id: str = ""
        self.bot_username: str = ""
        self.channel_id: str = ""
        self.guild_id: str = ""

    def load_bot_settings(self) -> Tuple[str, str]:
        token = ""
        channel_id = ""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                token = str(cfg.get("discord_bot_token") or "").strip()
                channel_id = str(cfg.get("discord_channel_id") or "").strip()
            except Exception:
                pass

        if not token and os.path.exists(DISCORD_CONFIG_FILE):
            try:
                with open(DISCORD_CONFIG_FILE, "r", encoding="utf-8") as f:
                    d_cfg = json.load(f)
                token = str(d_cfg.get("bot_token") or d_cfg.get("discord_bot_token") or "").strip()
            except Exception:
                pass

        if not channel_id:
            wh_meta = discover_channel_from_webhook()
            channel_id = wh_meta.get("channel_id", "")
            self.guild_id = wh_meta.get("guild_id", "")
            if channel_id and os.path.exists(CONFIG_FILE):
                try:
                    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    cfg["discord_channel_id"] = channel_id
                    if self.guild_id:
                        cfg["discord_guild_id"] = self.guild_id
                    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                        json.dump(cfg, f, indent=4)
                except Exception:
                    pass

        return token, channel_id

    def _headers(self, token: str) -> Dict[str, str]:
        clean = token.replace("Bot ", "").strip()
        return {
            "Authorization": f"Bot {clean}",
            "Content-Type": "application/json",
            "User-Agent": "BluffSkyAllskyBot (https://localhost, 1.0)",
        }

    def send_channel_reply(self, token: str, channel_id: str, reply_to_msg_id: str, text: str) -> None:
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        payload = {
            "content": text,
            "message_reference": {"message_id": str(reply_to_msg_id)},
            "allowed_mentions": {"replied_user": False},
        }
        try:
            r = requests.post(url, headers=self._headers(token), json=payload, timeout=8)
            if r.status_code not in (200, 201):
                logger.warning(f"Bot reply HTTP {r.status_code}: {r.text[:160]}")
        except Exception as e:
            logger.warning(f"Failed sending bot reply: {e}")

    def add_reaction_ack(self, token: str, channel_id: str, msg_id: str, emoji: str = "✅") -> None:
        from urllib.parse import quote
        enc = quote(emoji)
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages/{msg_id}/reactions/{enc}/@me"
        try:
            requests.put(url, headers=self._headers(token), timeout=6)
        except Exception:
            pass

    def handle_reply_message(self, token: str, msg: Dict[str, Any]) -> None:
        msg_id = str(msg.get("id") or "")
        if not msg_id or msg_id in self.processed_reply_ids:
            return

        author = msg.get("author") or {}
        if author.get("bot"):
            return

        # Skip if the bot already added a ✅ reaction to this user reply
        for r_item in msg.get("reactions") or []:
            if r_item.get("me") and (r_item.get("emoji") or {}).get("name") == "✅":
                self.processed_reply_ids.add(msg_id)
                claim_discord_action(f"reply:{msg_id}")
                return

        content = str(msg.get("content") or "").strip()
        channel_id = str(msg.get("channel_id") or self.channel_id)
        if self.channel_id and channel_id != self.channel_id:
            return

        # Support quick status/help command in #all-sky
        if content.lower() in ("!help", "!allsky", "!training", "!stats"):
            self.processed_reply_ids.add(msg_id)
            if not claim_discord_action(f"reply:{msg_id}"):
                return
            summary = get_feedback_summary()
            help_txt = (
                f"🌌 **BluffSky Allsky Active Learning Bot**\n"
                f"• **Total Ground-Truth Corrections Learned:** `{summary['total_samples']}`\n"
                f"• **How to Teach Me:** Reply directly to any Allsky alert image with:\n"
                f"  `plane` | `meteor` | `fireball` | `satellite` | `flash` | `aurora` | `bird` | `false` (or `cloud`/`glare`/`tree`)\n"
                f"• **Or React with an Emoji:** ☄️ (Meteor), 🔥 (Fireball), 🛰️ (Satellite), ✈️ (Plane), ✨ (Flash), 🦅 (Bird), ❌ (False Positive)"
            )
            self.send_channel_reply(token, channel_id, msg_id, help_txt)
            return

        # Check if this message is a reply to an Allsky alert message
        ref = msg.get("message_reference") or {}
        ref_msg_id = str(ref.get("message_id") or "")
        ref_msg_obj = msg.get("referenced_message")

        if not ref_msg_id:
            return

        self.processed_reply_ids.add(msg_id)
        # Atomically claim this reply ID in SQLite so multiple threads/processes or restarts never double-reply
        if not claim_discord_action(f"reply:{msg_id}"):
            return

        # If referenced_message wasn't included in payload, fetch it via REST
        if not ref_msg_obj:
            try:
                r = requests.get(
                    f"https://discord.com/api/v10/channels/{channel_id}/messages/{ref_msg_id}",
                    headers=self._headers(token),
                    timeout=6,
                )
                if r.status_code == 200:
                    ref_msg_obj = r.json()
            except Exception:
                pass

        ev = find_event_by_discord_message(ref_msg_id, ref_msg_obj)
        if not ev:
            return

        parsed = parse_correction_intent(content, mode=ev.get("mode", "NIGHT"))
        if not parsed:
            self.send_channel_reply(
                token,
                channel_id,
                msg_id,
                "🤔 I saw your reply to that sky image, but wasn't sure which label you meant! Reply with `plane`, `meteor`, `fireball`, `satellite`, `flash`, `bird`, or `false` (or react with ☄️ 🛰️ ✈️ ❌).",
            )
            return

        new_cat, _ = parsed
        user_name = author.get("global_name") or author.get("username") or "Owner"
        res = record_user_correction(
            frame_filename=ev["frame_filename"],
            corrected_category=new_cat,
            user_note=content,
            source="discord_reply",
            discord_user=user_name,
        )
        if res:
            self.add_reaction_ack(token, channel_id, msg_id, "✅")
            old_ic = CATEGORY_ICONS.get(res["old_category"], "🌌")
            new_ic = CATEGORY_ICONS.get(res["corrected_category"], "🌌")
            if res["corrected_category"] == "false_positive":
                reply_text = (
                    f"🧠 **Learned & Hidden!** Marked `{res['frame_filename']}` (was {old_ic} `{res['old_category']}`) "
                    f"as ❌ **False Positive** (`{content}`).\n"
                    f"Removed from the Top 50 gallery and added spatial/feature suppression (`{res['total_training_samples']}` total ground-truth samples)."
                )
            else:
                reply_text = (
                    f"🧠 **Learned!** Reclassified `{res['frame_filename']}` from {old_ic} `{res['old_category']}` → "
                    f"{new_ic} **`{res['corrected_category']}`** (*{res['corrected_subject']}*, Score `{res['new_interest_score']}`).\n"
                    f"Updated Web Observatory & Active Learning memory (`{res['total_training_samples']}` ground-truth samples)."
                )
            self.send_channel_reply(token, channel_id, msg_id, reply_text)

    def handle_reaction_event(self, token: str, channel_id: str, message_id: str, user_id: str, emoji_name: str) -> None:
        if user_id == self.bot_user_id:
            return
        if self.channel_id and channel_id != self.channel_id:
            return
        cat = EMOJI_TO_CATEGORY.get(emoji_name)
        if not cat:
            return

        dedup_key = f"{message_id}:{user_id}:{cat}"
        if dedup_key in self.processed_reaction_keys:
            return
        self.processed_reaction_keys.add(dedup_key)
        if not claim_discord_action(f"react:{dedup_key}"):
            return

        ref_msg_obj = None
        try:
            r = requests.get(
                f"https://discord.com/api/v10/channels/{channel_id}/messages/{message_id}",
                headers=self._headers(token),
                timeout=6,
            )
            if r.status_code == 200:
                ref_msg_obj = r.json()
        except Exception:
            pass

        ev = find_event_by_discord_message(message_id, ref_msg_obj)
        if not ev:
            return

        res = record_user_correction(
            frame_filename=ev["frame_filename"],
            corrected_category=cat,
            user_note=f"Discord reaction {emoji_name}",
            source="discord_reaction",
            discord_user=user_id,
        )
        if res:
            self.add_reaction_ack(token, channel_id, message_id, "✅")
            old_ic = CATEGORY_ICONS.get(res["old_category"], "🌌")
            new_ic = CATEGORY_ICONS.get(res["corrected_category"], "🌌")
            self.send_channel_reply(
                token,
                channel_id,
                message_id,
                f"🧠 **Learned from {emoji_name} reaction!** Reclassified `{res['frame_filename']}`: "
                f"{old_ic} `{res['old_category']}` → {new_ic} **`{res['corrected_category']}`** (*{res['corrected_subject']}*) "
                f"[`{res['total_training_samples']}` training samples].",
            )

    def poll_channel_messages_rest(self, token: str, channel_id: str) -> None:
        """
        Polls the latest 25 messages in `#all-sky` via REST API to catch any replies or reactions
        even if Gateway privileged Message Content intent is not enabled.
        """
        if not token or not channel_id:
            return
        try:
            r = requests.get(
                f"https://discord.com/api/v10/channels/{channel_id}/messages?limit=25",
                headers=self._headers(token),
                timeout=8,
            )
            if r.status_code == 200:
                msgs = r.json()
                # First mark any user message that already has a bot reply in the channel as processed
                for m in msgs:
                    if (m.get("author") or {}).get("bot"):
                        ref_id = str((m.get("message_reference") or {}).get("message_id") or "")
                        if ref_id:
                            self.processed_reply_ids.add(ref_id)
                            claim_discord_action(f"reply:{ref_id}")
                for m in reversed(msgs):
                    self.handle_reply_message(token, m)
            elif r.status_code in (401, 403):
                update_bot_status({
                    "state": "auth_or_perm_error",
                    "detail": f"Discord REST returned HTTP {r.status_code}. Make sure the Bot is invited to your server and has Read/Send permissions in #all-sky.",
                })
        except Exception as e:
            logger.debug(f"REST poll error: {e}")

    async def run_forever(self) -> None:
        ensure_feedback_schema()
        logger.info("Starting Allsky Discord Feedback & Active Learning Bot service...")

        while True:
            token, channel_id = self.load_bot_settings()
            self.channel_id = channel_id

            if not token:
                update_bot_status({
                    "state": "waiting_for_token",
                    "channel_id": channel_id,
                    "detail": "Waiting for Discord Bot Token (configure on http://localhost:8094/ under Discord Bot Setup).",
                })
                await asyncio.sleep(6)
                continue

            # Verify token via /users/@me
            try:
                me_resp = requests.get("https://discord.com/api/v10/users/@me", headers=self._headers(token), timeout=6)
                if me_resp.status_code != 200:
                    update_bot_status({
                        "state": "invalid_token",
                        "channel_id": channel_id,
                        "detail": f"Discord rejected Bot Token (HTTP {me_resp.status_code}). Check token in Web Observatory.",
                    })
                    await asyncio.sleep(10)
                    continue
                me_data = me_resp.json()
                self.bot_user_id = str(me_data.get("id") or "")
                self.bot_username = f"{me_data.get('username', 'Bot')}"
            except Exception as e:
                update_bot_status({"state": "connection_error", "detail": str(e)})
                await asyncio.sleep(10)
                continue

            update_bot_status({
                "state": "connected",
                "bot_username": self.bot_username,
                "bot_user_id": self.bot_user_id,
                "channel_id": self.channel_id,
                "guild_id": self.guild_id,
                "detail": f"Connected as @{self.bot_username} watching #all-sky (channel {self.channel_id})",
            })
            logger.info(f"Discord Feedback Bot authenticated as @{self.bot_username} (watching channel_id={self.channel_id})")

            # Initial REST catch-up scan
            self.poll_channel_messages_rest(token, self.channel_id)

            # Connect to Discord Gateway WebSocket + run periodic REST poll task concurrently
            try:
                # INTENTS: GUILDS (1<<0) | GUILD_MESSAGES (1<<9) | GUILD_MESSAGE_REACTIONS (1<<10) | MESSAGE_CONTENT (1<<15) = 34305
                # If MESSAGE_CONTENT (32768) is not enabled in portal, fallback to 1537 (GUILDS | GUILD_MESSAGES | GUILD_MESSAGE_REACTIONS)
                for intents_val in (34305, 1537):
                    closed_code = await self._connect_gateway_once(token, intents_val)
                    if closed_code == 4014:
                        logger.warning("Gateway closed with 4014 (Message Content Intent not enabled in Dev Portal); reconnecting with standard intents + REST content fetch.")
                        continue
                    break
            except Exception as e:
                logger.warning(f"Discord Gateway loop error: {e}")

            await asyncio.sleep(5)

    async def _connect_gateway_once(self, token: str, intents_val: int) -> Optional[int]:
        clean_token = token.replace("Bot ", "").strip()
        gw_url = "wss://gateway.discord.gg/?v=10&encoding=json"

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(gw_url, heartbeat=30.0) as ws:
                hello = await ws.receive_json()
                hb_interval = (hello.get("d") or {}).get("heartbeat_interval", 41250) / 1000.0

                # Send IDENTIFY
                await ws.send_json({
                    "op": 2,
                    "d": {
                        "token": clean_token,
                        "intents": intents_val,
                        "properties": {"os": "windows", "browser": "allsky_ai", "device": "allsky_ai"},
                    },
                })

                seq_holder = {"seq": None, "running": True}

                async def heartbeat_and_rest_loop():
                    last_rest = time.time()
                    while seq_holder["running"] and not ws.closed:
                        await asyncio.sleep(min(10.0, hb_interval * 0.4))
                        if ws.closed:
                            break
                        try:
                            await ws.send_json({"op": 1, "d": seq_holder["seq"]})
                        except Exception:
                            break
                        # Also run lightweight REST poll every 18s in case MESSAGE_CONTENT intent is off
                        if time.time() - last_rest >= 18.0:
                            last_rest = time.time()
                            await asyncio.to_thread(self.poll_channel_messages_rest, token, self.channel_id)

                hb_task = asyncio.create_task(heartbeat_and_rest_loop())
                try:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if data.get("s") is not None:
                                seq_holder["seq"] = data["s"]
                            op = data.get("op")
                            t = data.get("t")
                            d = data.get("d") or {}

                            if op == 11:
                                continue
                            if op in (7, 9):
                                break

                            if t == "MESSAGE_CREATE":
                                # If content is empty because MESSAGE_CONTENT intent wasn't toggled yet, fetch message via REST
                                if not d.get("content") and d.get("message_reference") and d.get("id"):
                                    ch_id = str(d.get("channel_id") or self.channel_id)
                                    m_id = str(d.get("id"))
                                    r = await asyncio.to_thread(
                                        requests.get,
                                        f"https://discord.com/api/v10/channels/{ch_id}/messages/{m_id}",
                                        headers=self._headers(token),
                                        timeout=6,
                                    )
                                    if r.status_code == 200:
                                        d = r.json()
                                await asyncio.to_thread(self.handle_reply_message, token, d)

                            elif t == "MESSAGE_REACTION_ADD":
                                emoji_name = (d.get("emoji") or {}).get("name") or ""
                                ch_id = str(d.get("channel_id") or "")
                                m_id = str(d.get("message_id") or "")
                                u_id = str(d.get("user_id") or "")
                                if emoji_name and m_id and u_id:
                                    await asyncio.to_thread(
                                        self.handle_reaction_event,
                                        token,
                                        ch_id,
                                        m_id,
                                        u_id,
                                        emoji_name,
                                    )
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
                finally:
                    seq_holder["running"] = False
                    hb_task.cancel()

                return ws.close_code


_BOT_THREAD: Optional[threading.Thread] = None


def start_background_bot_thread() -> None:
    """Starts the Discord Feedback Bot in a background daemon thread inside web_portal or standalone."""
    global _BOT_THREAD
    if _BOT_THREAD is not None and _BOT_THREAD.is_alive():
        return

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bot = DiscordFeedbackBot()
        loop.run_until_complete(bot.run_forever())

    _BOT_THREAD = threading.Thread(target=_runner, name="DiscordFeedbackBotThread", daemon=True)
    _BOT_THREAD.start()


if __name__ == "__main__":
    bot = DiscordFeedbackBot()
    asyncio.run(bot.run_forever())
