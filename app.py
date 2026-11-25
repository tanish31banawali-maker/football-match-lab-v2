import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import math
import os
import re
import shutil
import urllib.parse
import requests
from datetime import datetime
from contextlib import contextmanager

try:
    from PIL import Image
except Exception:
    Image = None
try:
    import pytesseract
except Exception:
    pytesseract = None

# ---------------- App & Database Configuration ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, 'football_lab_v5_1.db')
API_BASE = 'https://v3.football.api-sports.io'

st.set_page_config(
    page_title='Football Match Lab V6.1 Advanced Prediction & Risk Analysis',
    page_icon='⚽',
    layout='wide',
    initial_sidebar_state='expanded'
)

@contextmanager
def db():
    """Connection context that commits/rolls back and always closes SQLite."""
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

def backup_existing_database():
    """Create one consistent pre-upgrade SQLite backup without touching history."""
    if not os.path.exists(DB_PATH):
        return None
    backup_path = DB_PATH + '.pre_v6_backup'
    if os.path.exists(backup_path):
        return backup_path
    source = sqlite3.connect(DB_PATH)
    target = sqlite3.connect(backup_path)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    return backup_path


def init_db():
    with db() as con:
        con.execute('''CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source_mode TEXT NOT NULL,
            fixture_id INTEGER,
            home TEXT NOT NULL,
            away TEXT NOT NULL,
            neutral_venue INTEGER DEFAULT 0,
            p_home REAL NOT NULL,
            p_draw REAL NOT NULL,
            p_away REAL NOT NULL,
            p_over15 REAL,
            p_over25 REAL,
            p_over35 REAL,
            p_btts REAL,
            exp_home REAL,
            exp_away REAL,
            odd_h REAL,
            odd_d REAL,
            odd_a REAL,
            actual_home INTEGER,
            actual_away INTEGER,
            settled_at TEXT,
            home_attributes_json TEXT,
            away_attributes_json TEXT,
            home_players_json TEXT,
            away_players_json TEXT
        )''')
        # Add columns to an existing V5.1 database without deleting its history.
        existing_columns = {
            row[1] for row in con.execute("PRAGMA table_info(predictions)").fetchall()
        }
        for column in (
            "home_attributes_json", "away_attributes_json",
            "home_players_json", "away_players_json",
            "match_context_json", "home_form_json", "away_form_json"
        ):
            if column not in existing_columns:
                con.execute(f"ALTER TABLE predictions ADD COLUMN {column} TEXT")
        # One record per tracked market and prediction. This is separate from
        # the match table so each market can be audited independently.
        con.execute('''CREATE TABLE IF NOT EXISTS market_settlements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            market TEXT NOT NULL,
            selection TEXT NOT NULL,
            predicted_yes_probability REAL NOT NULL,
            selected_probability REAL NOT NULL,
            actual_yes INTEGER NOT NULL,
            result TEXT NOT NULL,
            settled_at TEXT NOT NULL,
            UNIQUE(prediction_id, market),
            FOREIGN KEY(prediction_id) REFERENCES predictions(id)
        )''')


def settle_market_records(prediction_id, home_goals, away_goals, settled_at=None):
    """Save Over 1.5, Over 2.5 and BTTS selections and their actual outcomes."""
    settled_at = settled_at or datetime.now().strftime("%Y-%m-%d %H:%M")
    total_goals = int(home_goals) + int(away_goals)
    with db() as con:
        row = con.execute("SELECT * FROM predictions WHERE id = ?", (int(prediction_id),)).fetchone()
        if row is None:
            return
        markets = [
            ("Over 1.5", float(row["p_over15"] or 0.0), total_goals >= 2),
            ("Over 2.5", float(row["p_over25"] or 0.0), total_goals >= 3),
            ("BTTS", float(row["p_btts"] or 0.0), int(home_goals) >= 1 and int(away_goals) >= 1),
        ]
        for market, p_yes, actual_yes in markets:
            selection = "Yes" if p_yes >= 0.5 else "No"
            selected_probability = p_yes if selection == "Yes" else 1.0 - p_yes
            won = (selection == "Yes" and actual_yes) or (selection == "No" and not actual_yes)
            con.execute('''INSERT INTO market_settlements
                (prediction_id, market, selection, predicted_yes_probability,
                 selected_probability, actual_yes, result, settled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(prediction_id, market) DO UPDATE SET
                selection=excluded.selection,
                predicted_yes_probability=excluded.predicted_yes_probability,
                selected_probability=excluded.selected_probability,
                actual_yes=excluded.actual_yes,
                result=excluded.result,
                settled_at=COALESCE(market_settlements.settled_at, excluded.settled_at)''',
                (int(prediction_id), market, selection, p_yes, selected_probability,
                 int(actual_yes), "WIN" if won else "LOSS", settled_at))


def backfill_market_settlements():
    """Create market records for matches settled before this tracking update."""
    with db() as con:
        old_rows = con.execute("SELECT id, actual_home, actual_away FROM predictions WHERE actual_home IS NOT NULL AND actual_away IS NOT NULL").fetchall()
    for row in old_rows:
        settle_market_records(row["id"], row["actual_home"], row["actual_away"])

backup_existing_database()
init_db()
backfill_market_settlements()

# ---------------- Vectorized Dixon-Coles Score Matrix ----------------
G = np.arange(11)
LG = np.array([math.lgamma(k + 1) for k in G])

def compute_score_matrix(lh, ma, rho=-0.13, home_adv=1.08, unc=0.15, n=2500, seed=42):
    """Exact vectorized integration over lognormal rate uncertainty."""
    rng = np.random.default_rng(int(seed))
    lh = max(0.05, float(lh))
    ma = max(0.05, float(ma))

    lh_s = lh * home_adv * np.exp(rng.normal(-unc**2 / 2.0, unc, int(n)))
    ma_s = ma * np.exp(rng.normal(-unc**2 / 2.0, unc, int(n)))

    pm = lambda lam: np.exp(-lam[:, None] + G * np.log(lam[:, None]) - LG)
    M = pm(lh_s)[:, :, None] * pm(ma_s)[:, None, :]

    # Dixon-Coles low-score correlation adjustments
    M[:, 0, 0] *= np.maximum(1e-4, 1.0 - lh_s * ma_s * rho)
    M[:, 0, 1] *= np.maximum(1e-4, 1.0 + lh_s * rho)
    M[:, 1, 0] *= np.maximum(1e-4, 1.0 + ma_s * rho)
    M[:, 1, 1] *= np.maximum(1e-4, 1.0 - rho)

    M /= M.sum(axis=(1, 2), keepdims=True)
    return M.mean(axis=0)

def extract_markets(M):
    """Extracts 1X2, totals, BTTS, expected goals, and top scorelines directly from the probability matrix."""
    T = G[:, None] + G[None, :]
    p_home = float(np.tril(M, -1).sum())
    p_draw = float(np.trace(M))
    p_away = float(np.triu(M, 1).sum())

    tot = p_home + p_draw + p_away
    p_home, p_draw, p_away = p_home / tot, p_draw / tot, p_away / tot

    exp_home = float((M.sum(axis=1) * G).sum())
    exp_away = float((M.sum(axis=0) * G).sum())

    scores = []
    for h in range(11):
        for a in range(11):
            scores.append({'Score': f"{h}-{a}", 'Probability': float(M[h, a])})
    scores_df = pd.DataFrame(scores).sort_values('Probability', ascending=False).head(10)

    return {
        'p_home': p_home,
        'p_draw': p_draw,
        'p_away': p_away,
        'p_over15': float(M[T >= 2].sum()),
        'p_over25': float(M[T >= 3].sum()),
        'p_over35': float(M[T >= 4].sum()),
        'p_btts': float(M[1:, 1:].sum()),
        'exp_home': exp_home,
        'exp_away': exp_away,
        'scores': scores_df,
        'matrix': M
    }

# ---------------- Calibrated Strength & Role-Based Absence Model ----------------
def shrink_rate(obs_rate, sample_size, baseline=1.35, k=8.0):
    w = sample_size / (sample_size + k)
    return w * obs_rate + (1.0 - w) * baseline

def calculate_expected_goals(data, elo_weight=0.35):
    """
    Combines Bayesian-shrunk, opponent-adjusted goal averages with a calibrated Elo blend
    and role-based injury adjustments (attackers cut own xG; defenders/GK raise opponent xG).
    """
    h = data['home']
    a = data['away']

    h_gf_s = shrink_rate(h['gf'], h.get('sample_size', 6))
    h_ga_s = shrink_rate(h['ga'], h.get('sample_size', 6))
    a_gf_s = shrink_rate(a['gf'], a.get('sample_size', 6))
    a_ga_s = shrink_rate(a['ga'], a.get('sample_size', 6))

    # Opponent strength adjustment on recent schedule
    h_opp_fac = 10.0 ** ((h.get('avg_opp_elo', 1750.0) - 1750.0) / 1000.0)
    a_opp_fac = 10.0 ** ((a.get('avg_opp_elo', 1750.0) - 1750.0) / 1000.0)

    h_gf_adj = h_gf_s * h_opp_fac
    h_ga_adj = h_ga_s / h_opp_fac
    a_gf_adj = a_gf_s * a_opp_fac
    a_ga_adj = a_ga_s / a_opp_fac

    base_lh = 0.55 * h_gf_adj + 0.45 * a_ga_adj
    base_ma = 0.55 * a_gf_adj + 0.45 * h_ga_adj

    # Calibrated Elo multiplier (divisor 1200 blended via elo_weight to avoid double-counting)
    elo_diff = h['elo'] - a['elo']
    raw_elo_h = 10.0 ** (elo_diff / 1200.0)
    raw_elo_a = 10.0 ** (-elo_diff / 1200.0)

    elo_mult_h = (1.0 - elo_weight) + elo_weight * raw_elo_h
    elo_mult_a = (1.0 - elo_weight) + elo_weight * raw_elo_a

    lh = base_lh * elo_mult_h
    ma = base_ma * elo_mult_a

    # Optional attribute ratings: neutral at 50; capped to avoid overpowering form/Elo.
    def attribute_attack_factor(team):
        r = team.get("attributes") or {}
        attack = float(r.get("attack", 50))
        midfield = float(r.get("midfield", 50))
        form = float(r.get("form", 50))
        squad = float(r.get("squad_strength", 50))
        return min(1.12, max(0.88, 1.0 + (
            0.40 * (attack - 50) + 0.20 * (midfield - 50) +
            0.20 * (form - 50) + 0.20 * (squad - 50)
        ) / 1000.0))

    def attribute_defence_factor(team):
        r = team.get("attributes") or {}
        defence = float(r.get("defence", 50))
        goalkeeping = float(r.get("goalkeeping", 50))
        return min(1.10, max(0.90, 1.0 - (
            0.65 * (defence - 50) + 0.35 * (goalkeeping - 50)
        ) / 1000.0))

    lh *= attribute_attack_factor(h) * attribute_defence_factor(a)
    ma *= attribute_attack_factor(a) * attribute_defence_factor(h)

    # Role-based & importance-weighted absence impact
    imp_weights = {'star': 0.065, 'starter': 0.035, 'rotation': 0.012}
    status_weights = {'injured': 1.0, 'suspended': 1.0, 'doubtful': 0.45, 'unknown': 0.0, 'available': 0.0}

    def calc_absence_effects(absences):
        att_pen, def_leak = 0.0, 0.0
        for ab in absences:
            w = imp_weights.get(str(ab.get('importance', 'starter')).lower(), 0.035)
            w *= status_weights.get(str(ab.get('status', 'injured')).lower(), 0.0)
            pos = str(ab.get('position', 'MID')).upper()
            if pos in ('FWD', 'ST', 'ATT', 'FORWARD'):
                att_pen += w
            elif pos in ('DEF', 'GK', 'DEFENDER', 'GOALKEEPER'):
                def_leak += w
            else:
                att_pen += w * 0.5
                def_leak += w * 0.5
        return att_pen, def_leak

    h_att_pen, h_def_leak = calc_absence_effects(h.get('absences', []))
    a_att_pen, a_def_leak = calc_absence_effects(a.get('absences', []))

    lh = lh * max(0.65, 1.0 - h_att_pen) * min(1.35, 1.0 + a_def_leak)
    ma = ma * max(0.65, 1.0 - a_att_pen) * min(1.35, 1.0 + h_def_leak)

    return round(lh, 3), round(ma, 3)


# ---------------- Screenshot OCR & Attribute Ratings ----------------
def extract_screenshot_text(uploaded_file):
    """Try OCR when pytesseract and the local Tesseract engine are installed."""
    if Image is None or pytesseract is None:
        raise RuntimeError(
            "OCR is not installed. Install Pillow and pytesseract, plus the Tesseract OCR "
            "application, or type/paste the screenshot details into the review box."
        )
    image = Image.open(uploaded_file).convert("RGB")
    return pytesseract.image_to_string(image)

def render_team_attribute_ratings(prefix):
    """Optional 0–100 team ratings. Missing ratings remain neutral and are not guessed."""
    st.markdown("#### Team Attribute Ratings (optional)")
    st.caption("Enter ratings only when you have a source. Leave sliders at 50 for neutral/no adjustment.")
    labels = ["Attack", "Defence", "Midfield", "Goalkeeping", "Form", "Squad Strength"]
    cols = st.columns(3)
    ratings = {}
    for i, label in enumerate(labels):
        ratings[label.lower().replace(" ", "_")] = cols[i % 3].slider(
            f"{prefix} {label}", 0, 100, 50, 1, key=f"{prefix}_{label}_rating"
        )
    ratings["overall"] = round(sum(ratings.values()) / len(ratings), 1)
    return ratings

def render_player_attribute_table(prefix):
    """Optional player ratings; these are displayed/stored with the match payload."""
    st.markdown("#### Optional Player Attribute Ratings")
    st.caption("Add players only when you have ratings. These ratings are not inferred from names.")
    return st.data_editor(
        pd.DataFrame(columns=[
            "player", "position", "pace", "shooting", "passing",
            "dribbling", "defending", "physical", "overall"
        ]),
        column_config={
            "position": st.column_config.SelectboxColumn(
                "Position", options=["GK", "DEF", "MID", "FWD"]
            ),
            **{
                col: st.column_config.NumberColumn(col.title(), min_value=0, max_value=100, step=1)
                for col in ["pace", "shooting", "passing", "dribbling", "defending", "physical", "overall"]
            },
        },
        num_rows="dynamic",
        key=f"{prefix}_player_ratings",
        use_container_width=True
    )

def apply_attribute_ratings(payload, home_ratings=None, away_ratings=None,
                            home_players=None, away_players=None):
    """Attach optional ratings and apply a deliberately small rating-based xG modifier."""
    for side, ratings, players in [
        ("home", home_ratings, home_players),
        ("away", away_ratings, away_players),
    ]:
        if ratings:
            payload[side]["attributes"] = ratings
        if players is not None:
            payload[side]["player_ratings"] = players
    return payload


def parse_recent_match_rows(raw_text):
    """Parse a bulk-pasted recent-match log; optional stats remain None."""
    columns = ["date", "opponent", "venue", "gf", "ga", "xg", "xga", "opponent_elo"]
    rows, errors = [], []
    for line_no, raw_line in enumerate((raw_text or "").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        delimiter = "|" if "|" in line else ("\t" if "\t" in line else ",")
        parts = [part.strip() for part in line.split(delimiter)]
        if parts and parts[0].lower() in {"date", "match date"}:
            continue
        if len(parts) < 5:
            errors.append(f"Line {line_no}: enter Date | Opponent | Venue | GF | GA (optional: xG | xGA | Opponent Elo).")
            continue
        parts += [""] * (len(columns) - len(parts))
        item = dict(zip(columns, parts[:len(columns)]))
        if not item["opponent"] or not item["venue"] or not item["gf"] or not item["ga"]:
            errors.append(f"Line {line_no}: opponent, venue, GF and GA are required.")
            continue
        if item["venue"].title() not in {"Home", "Away", "Neutral"}:
            errors.append(f"Line {line_no}: venue must be Home, Away or Neutral.")
            continue
        bad = False
        for field in ("gf", "ga", "xg", "xga", "opponent_elo"):
            raw_value = item[field]
            if not raw_value:
                item[field] = None
                continue
            try:
                value = float(raw_value)
                upper = 2600 if field == "opponent_elo" else 20
                if value < 0 or value > upper or (field in {"gf", "ga"} and not value.is_integer()):
                    raise ValueError
                item[field] = int(value) if value.is_integer() else value
            except ValueError:
                errors.append(f"Line {line_no}: {field.upper()} has an invalid value.")
                bad = True
        item["venue"] = item["venue"].title()
        if not bad:
            rows.append(item)
    return rows, errors

def parse_pasted_rows(raw_text, columns, required_fields, allowed_values=None):
    """Parse bulk-pasted pipe/tab/comma rows into validated dictionaries."""
    rows, errors = [], []
    for line_no, raw_line in enumerate((raw_text or "").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        delimiter = "|" if "|" in line else ("\t" if "\t" in line else ",")
        parts = [part.strip() for part in line.split(delimiter)]
        if parts and parts[0].strip().lower() in {"player", "name"}:
            continue  # allow a pasted column-heading row
        if len(parts) == 1 and len(columns) > 1:
            errors.append(f"Line {line_no}: use '|' between fields.")
            continue
        parts += [""] * (len(columns) - len(parts))
        item = dict(zip(columns, parts[:len(columns)]))
        if not all(item.get(field) for field in required_fields):
            errors.append(f"Line {line_no}: missing required field(s): {', '.join(f for f in required_fields if not item.get(f))}.")
            continue
        bad = False
        for field, choices in (allowed_values or {}).items():
            if item.get(field) and item[field].upper() not in choices:
                errors.append(f"Line {line_no}: {field} must be one of {', '.join(choices)}.")
                bad = True
        for field in ("rating_0_100", "fitness_0_100"):
            if item.get(field):
                try:
                    value = float(item[field])
                    if not 0 <= value <= 100:
                        raise ValueError
                    item[field] = int(value) if value.is_integer() else value
                except ValueError:
                    errors.append(f"Line {line_no}: {field} must be between 0 and 100.")
                    bad = True
            else:
                item[field] = None
        if not bad:
            item["position"] = item.get("position", "").upper()
            item["role"] = item.get("role", "").title()
            rows.append(item)
    return rows, errors

# ---------------- Strict Validation & Parsers (No Silent Defaults) ----------------
def validate_match_payload(data):
    """Strictly validates a match payload; never substitutes silent defaults."""
    if not isinstance(data, dict):
        return None, ["Payload must be a valid dictionary/JSON object."]

    errors = []
    for side in ('home', 'away'):
        if side not in data or not isinstance(data[side], dict):
            errors.append(f"Missing '{side}' section.")
            continue
        team_obj = data[side]
        if not team_obj.get('name') or not str(team_obj['name']).strip():
            errors.append(f"Missing '{side}.name'.")
        for num_field in ('gf', 'ga', 'elo'):
            val = team_obj.get(num_field)
            if val is None or isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                errors.append(f"Missing or invalid numeric '{side}.{num_field}'.")
            elif num_field == 'elo' and not (800 <= val <= 2600):
                errors.append(f"'{side}.elo' ({val}) out of realistic bounds (800–2600).")
            elif num_field in ('gf', 'ga') and val < 0:
                errors.append(f"'{side}.{num_field}' cannot be negative.")

    if 'odds' not in data or not isinstance(data['odds'], dict):
        errors.append("Missing 'odds' section.")
    else:
        for ok in ('home', 'draw', 'away'):
            val = data['odds'].get(ok)
            if val is None or isinstance(val, bool) or not isinstance(val, (int, float)) or val <= 1.0:
                errors.append(f"Missing or invalid '{ok}' decimal odds (must be > 1.0).")

    if errors:
        return None, errors
    return data, []

def parse_number_after_colon(line):
    if ':' not in line:
        return None
    after = line.split(':', 1)[1].replace(',', '').strip()
    m = re.search(r'^-?\d+(?:\.\d+)?', after)
    return float(m.group(0)) if m else None

def _normalise_absence_position(value):
    v = str(value or '').strip().lower()
    if any(x in v for x in ('goalkeeper', 'keeper', ' gk', 'gk')): return 'GK'
    if any(x in v for x in ('defender', 'defence', 'defense', 'back', 'centre-back', 'center-back', 'cb', 'lb', 'rb')): return 'DEF'
    if re.search(r'\b(forward|striker|winger|attacker|fwd|st)\b', v): return 'FWD'
    if any(x in v for x in ('midfielder', 'midfield', ' mid', ' dm', ' cm', ' am')): return 'MID'
    return 'MID'

def _normalise_absence_importance(value):
    v = str(value or '').strip().lower()
    if re.search(r'\b(star|key|world class)\b', v): return 'star'
    if re.search(r'\b(rotation|backup|bench)\b', v): return 'rotation'
    return 'starter'

def _normalise_absence_status(value):
    v = str(value or '').strip().lower()
    if any(x in v for x in ('doubt', 'questionable', 'late test', '50/50')): return 'doubtful'
    if any(x in v for x in ('suspend', 'banned', 'red card')): return 'suspended'
    if any(x in v for x in ('injur', 'out', 'unavailable', 'ruled out', 'not available')): return 'injured'
    if any(x in v for x in ('available', 'fit', 'returned', 'back in training')): return 'available'
    return 'unknown'

def _parse_absence_line(line):
    """Accept pipe, comma, dash and parenthesised absence formats."""
    raw = line.strip().lstrip('-*• ').strip()
    if not raw: return None
    parts = [x.strip() for x in re.split(r'\s*\|\s*|\s*;\s*|\s*,\s*', raw) if x.strip()]
    if len(parts) >= 2:
        player = parts[0]
        joined = ' '.join(parts[1:])
        position = _normalise_absence_position(joined)
        importance = _normalise_absence_importance(joined)
        status = _normalise_absence_status(joined)
    else:
        player = re.split(r'\s+[—–-]\s+|\s*\(', raw, maxsplit=1)[0].strip()
        position = _normalise_absence_position(raw)
        importance = _normalise_absence_importance(raw)
        status = _normalise_absence_status(raw)
    if not player or player.lower() in {'player', 'none', 'n/a', 'no absences', 'nil'}:
        return None
    return {'player': player, 'position': position, 'importance': importance, 'status': status}

def parse_structured_text_block(raw_text):
    """Robust parser for JSON and human-written match blocks.

    Supports explicit Home/Away team fields, standalone team headings, sectioned
    absences, pipe-delimited player rows, comma/dash formats, and common odds labels.
    It never invents missing numeric inputs or bookmaker odds.
    """
    text = str(raw_text or '').strip()
    if not text:
        return None, ['Input block is empty.']
    if text.startswith('{'):
        try:
            obj = json.loads(text)
            # Normalise JSON absence rows as well as text rows.
            for side in ('home', 'away'):
                if isinstance(obj.get(side), dict):
                    fixed=[]
                    for item in obj[side].get('absences', []) or []:
                        if isinstance(item, dict):
                            fixed.append({
                                'player': str(item.get('player', item.get('name', ''))).strip(),
                                'position': _normalise_absence_position(item.get('position')),
                                'importance': _normalise_absence_importance(item.get('importance')),
                                'status': _normalise_absence_status(item.get('status', 'injured')),
                            })
                    obj[side]['absences'] = fixed
            return validate_match_payload(obj)
        except json.JSONDecodeError as e:
            return None, [f'JSON syntax error: {e}']

    payload = {
        'home': {'name': None, 'gf': None, 'ga': None, 'elo': None, 'sample_size': 6, 'avg_opp_elo': 1750.0, 'absences': []},
        'away': {'name': None, 'gf': None, 'ga': None, 'elo': None, 'sample_size': 6, 'avg_opp_elo': 1750.0, 'absences': []},
        'odds': {'home': None, 'draw': None, 'away': None}, 'h2h': []
    }
    side = None
    section = None
    for original in text.splitlines():
        line = original.strip()
        if not line or line.startswith('```'):
            continue
        low = line.lower().strip(' :')
        # Side-specific section labels, including the format previously rejected.
        m = re.match(r'^(home|away)\s+(absences?|injuries|injured|unavailable|team news)\s*:?$', line, re.I)
        if m:
            side, section = m.group(1).lower(), 'absences'
            continue
        m = re.match(r'^(home|away)\s+(team)\s*:\s*(.+)$', line, re.I)
        if m:
            side = m.group(1).lower(); payload[side]['name'] = m.group(3).strip(); section = 'stats'; continue
        m = re.match(r'^(home|away)\s+(gf|ga|elo|matches sample size|sample size|avg opponent elo)\s*:\s*(.+)$', line, re.I)
        if m:
            side = m.group(1).lower(); key=m.group(2).lower(); val=parse_number_after_colon(line)
            field={'gf':'gf','ga':'ga','elo':'elo','matches sample size':'sample_size','sample size':'sample_size','avg opponent elo':'avg_opp_elo'}[key]
            if val is not None: payload[side][field] = int(val) if field == 'sample_size' else val
            section='stats'; continue
        # Common block headings
        if re.match(r'^(home|away)\s+absences\s*:', line, re.I):
            side=re.match(r'^(home|away)', line, re.I).group(1).lower(); section='absences'; continue
        if re.match(r'^(home|away)\s+team\s*:', line, re.I):
            continue
        # A standalone team heading selects side while in a stats/absence block.
        if ':' not in line and not line.startswith(('-', '*', '•')):
            candidate=line.strip()
            if candidate.lower() in {'home absences','away absences','home injuries','away injuries'}:
                side=candidate.split()[0].lower(); section='absences'; continue
            if payload['home']['name'] and candidate.lower()==payload['home']['name'].lower(): side='home'; continue
            if payload['away']['name'] and candidate.lower()==payload['away']['name'].lower(): side='away'; continue
            if section is None and payload['home']['name'] is None:
                payload['home']['name']=candidate; side='home'; section='stats'; continue
            if section is None and payload['away']['name'] is None:
                payload['away']['name']=candidate; side='away'; section='stats'; continue
        # Odds lines terminate an active absence section; otherwise they could
        # accidentally be interpreted as player names.
        if re.match(r'^(home odds|draw odds|away odds|odds\s*[1x2])\s*:', line, re.I):
            section = 'odds'
        # Absence rows: parse while in absence section, and also accept explicit
        # 'Player | Position | Importance | Status' rows beneath team headers.
        if section == 'absences' and side in ('home','away'):
            item = _parse_absence_line(line)
            if item: payload[side]['absences'].append(item)
            continue
        # Inline absence declarations such as 'Home Absences: Player | DEF | Star | Injured'
        inline = re.match(r'^(home|away)\s+(?:absences?|injuries)\s*:\s*(.+)$', line, re.I)
        if inline:
            side=inline.group(1).lower(); section='absences'
            item=_parse_absence_line(inline.group(2))
            if item: payload[side]['absences'].append(item)
            continue
        # Stats labels with or without 'Avg' and common variants.
        if side in ('home','away') and ':' in line:
            val=parse_number_after_colon(line)
            if val is not None:
                if re.search(r'\b(gf|avg\.?\s*goals?\s*scored|goals?\s*scored)\b', low): payload[side]['gf']=val; continue
                if re.search(r'\b(ga|avg\.?\s*goals?\s*conceded|goals?\s*conceded)\b', low): payload[side]['ga']=val; continue
                if re.search(r'\belo\b', low): payload[side]['elo']=val; continue
                if re.search(r'\b(sample size|matches sample size)\b', low): payload[side]['sample_size']=int(val); continue
                if re.search(r'\b(avg opponent elo|opponent elo)\b', low): payload[side]['avg_opp_elo']=val; continue
        # Odds labels work regardless of whether an Odds heading exists.
        if ':' in line:
            val=parse_number_after_colon(line)
            if val is not None:
                if re.search(r'\b(home odds|odds\s*1)\b', low): payload['odds']['home']=val; continue
                if re.search(r'\b(draw odds|odds\s*x)\b', low): payload['odds']['draw']=val; continue
                if re.search(r'\b(away odds|odds\s*2)\b', low): payload['odds']['away']=val; continue
        if low in {'odds','bookmaker odds','market odds'}:
            section='odds'; continue
        if section == 'odds' and ':' in line:
            val=parse_number_after_colon(line)
            if val is not None:
                if re.search(r'\b(home|1)\b', low): payload['odds']['home']=val
                elif re.search(r'\b(draw|x)\b', low): payload['odds']['draw']=val
                elif re.search(r'\b(away|2)\b', low): payload['odds']['away']=val
    # Add actionable parser diagnostics when absences were mentioned but no rows parsed.
    errors=[]
    if re.search(r'\b(absences?|injuries|injured|suspended)\b', text, re.I) and not (payload['home']['absences'] or payload['away']['absences']):
        errors.append('Absence section was detected but no player rows were parsed. Use: Player | Position | Importance | Status.')
    valid, validation_errors = validate_match_payload(payload)
    return valid, errors + validation_errors


# ---------------- SofaScore Match-Link Import ----------------
SOFASCORE_API = "https://www.sofascore.com/api/v1"

def sofascore_get(path):
    """Read publicly accessible SofaScore match data. This endpoint is unofficial and may be rate-limited."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json, text/plain, */*",
    }
    response = requests.get(f"{SOFASCORE_API}/{path.lstrip('/')}", headers=headers, timeout=20)
    if response.status_code in (403, 429):
        raise RuntimeError("SofaScore blocked or rate-limited this request. Try again later or use Manual/Block mode.")
    response.raise_for_status()
    return response.json()

def sofascore_event_id(url):
    """Supports exact SofaScore match URLs containing #id:123456 or /event/123456."""
    if not re.match(r"^https?://([a-z0-9-]+\.)?sofascore\.com/", url.strip(), re.I):
        raise ValueError("Please paste a valid sofascore.com URL.")
    match = re.search(r"(?:id:|/event/)(\d+)", url)
    if not match:
        raise ValueError("This is not a specific match URL. If it is a Team Compare link, the importer will use the two team IDs instead.")
    return int(match.group(1))


def sofascore_compare_team_ids(url):
    """Read the two team IDs from SofaScore /football/team/compare?ids=... links."""
    parsed = urllib.parse.urlparse(url.strip())
    if not re.match(r"^https?://([a-z0-9-]+\.)?sofascore\.com$", f"{parsed.scheme}://{parsed.netloc}", re.I):
        raise ValueError("Please paste a valid sofascore.com URL.")
    if "/team/compare" not in parsed.path:
        return None
    params = urllib.parse.parse_qs(parsed.query)
    ids = params.get("ids", [""])[0].split(",")
    ids = [x.strip() for x in ids if x.strip().isdigit()]
    if len(ids) != 2:
        raise ValueError("This Team Compare link must contain exactly two team IDs in its ids= parameter.")
    return int(ids[0]), int(ids[1])


def fetch_sofascore_team_compare(url):
    """Import both teams and recent form from a SofaScore Team Compare URL.

    A compare URL identifies two teams, not one scheduled fixture. Therefore
    no single match status, kickoff time or event ID can be inferred from it.
    """
    home_id, away_id = sofascore_compare_team_ids(url)
    team_data = []
    for team_id in (home_id, away_id):
        body = sofascore_get(f"team/{team_id}")
        team = body.get("team") or body.get("teamInfo") or {}
        name = team.get("name") or team.get("shortName")
        if not name:
            raise RuntimeError(f"SofaScore did not return the name for team ID {team_id}.")
        gf, ga, n, rows = sofascore_recent_average(team_id)
        team_data.append({"id": team_id, "name": name, "gf": gf, "ga": ga,
                          "sample_size": n, "elo": None, "avg_opp_elo": 1750.0,
                          "absences": [], "recent": rows})

    # Compare the teams' recent event lists to identify shared past meetings.
    h2h_rows = []
    try:
        h_events = sofascore_get(f"team/{home_id}/events/last/0").get("events", [])
        a_events = sofascore_get(f"team/{away_id}/events/last/0").get("events", [])
        away_event_ids = {e.get("id") for e in a_events}
        for e in h_events:
            if e.get("id") not in away_event_ids:
                continue
            ht = (e.get("homeTeam") or {}).get("name", "Home")
            at = (e.get("awayTeam") or {}).get("name", "Away")
            hs = (e.get("homeScore") or {}).get("current", "?")
            aws = (e.get("awayScore") or {}).get("current", "?")
            h2h_rows.append(f"{ht} {hs}-{aws} {at}")
            if len(h2h_rows) >= 10:
                break
    except Exception:
        pass

    return {
        "event_id": None,
        "home": team_data[0],
        "away": team_data[1],
        "h2h": h2h_rows,
        "event_name": f"{team_data[0]['name']} vs {team_data[1]['name']}",
        "status": "Team comparison link — no specific fixture selected",
        "start_timestamp": None,
        "home_recent": team_data[0]["recent"],
        "away_recent": team_data[1]["recent"],
        "source_type": "team_compare",
    }

def sofascore_recent_average(team_id, count=6):
    """Calculate goals for/against from the team's last completed matches."""
    body = sofascore_get(f"team/{int(team_id)}/events/last/0")
    events = body.get("events", [])
    rows = []
    for event in events:
        status = (event.get("status") or {}).get("type", "")
        if status != "finished":
            continue
        home = event.get("homeTeam") or {}
        away = event.get("awayTeam") or {}
        hs = (event.get("homeScore") or {}).get("current")
        aws = (event.get("awayScore") or {}).get("current")
        if hs is None or aws is None:
            continue
        if int(home.get("id", -1)) == int(team_id):
            rows.append((float(hs), float(aws)))
        elif int(away.get("id", -1)) == int(team_id):
            rows.append((float(aws), float(hs)))
        if len(rows) >= count:
            break
    if not rows:
        raise RuntimeError("SofaScore did not return completed recent matches for this team.")
    gf = sum(x[0] for x in rows) / len(rows)
    ga = sum(x[1] for x in rows) / len(rows)
    return round(gf, 3), round(ga, 3), len(rows), rows

def fetch_sofascore_match(url):
    """Fetch a SofaScore match URL or Team Compare URL."""
    compare_ids = sofascore_compare_team_ids(url)
    if compare_ids:
        return fetch_sofascore_team_compare(url)
    event_id = sofascore_event_id(url)
    body = sofascore_get(f"event/{event_id}")
    event = body.get("event") or {}
    home = event.get("homeTeam") or {}
    away = event.get("awayTeam") or {}
    if not home.get("name") or not away.get("name"):
        raise RuntimeError("The link did not return a recognizable football match.")
    hgf, hga, hn, hrows = sofascore_recent_average(home["id"])
    agf, aga, an, arows = sofascore_recent_average(away["id"])
    h2h_rows = []
    try:
        duel = sofascore_get(f"event/{event_id}/h2h")
        for item in (duel.get("events") or [])[:10]:
            ht = (item.get("homeTeam") or {}).get("name", "Home")
            at = (item.get("awayTeam") or {}).get("name", "Away")
            hs = (item.get("homeScore") or {}).get("current", "?")
            aws = (item.get("awayScore") or {}).get("current", "?")
            h2h_rows.append(f"{ht} {hs}-{aws} {at}")
    except Exception:
        pass
    return {
        "event_id": event_id,
        "home": {"name": home["name"], "gf": hgf, "ga": hga, "sample_size": hn, "elo": None, "avg_opp_elo": 1750.0, "absences": []},
        "away": {"name": away["name"], "gf": agf, "ga": aga, "sample_size": an, "elo": None, "avg_opp_elo": 1750.0, "absences": []},
        "h2h": h2h_rows,
        "event_name": f"{home['name']} vs {away['name']}",
        "status": (event.get("status") or {}).get("description", "Unknown"),
        "start_timestamp": event.get("startTimestamp"),
        "home_recent": hrows, "away_recent": arows, "source_type": "match",
    }

# ---------------- API-Football Live Data Engine ----------------
def get_api_key():
    if st.session_state.get('api_key'):
        return st.session_state['api_key']
    try:
        return st.secrets.get("API_FOOTBALL_KEY", "")
    except Exception:
        return ""

def api_get(path, params, key):
    if not key or not key.strip():
        raise RuntimeError("API-Football key is required for Automatic API Mode.")
    resp = requests.get(f"{API_BASE}/{path.lstrip('/')}", headers={'x-apisports-key': key.strip()}, params=params, timeout=20)
    resp.raise_for_status()
    body = resp.json()
    if body.get('errors') and str(body['errors']) not in ("[]", "{}"):
        raise RuntimeError(f"API-Football error: {body['errors']}")
    return body.get('response', [])

def is_senior_team(name):
    banned = (' u21', ' u19', ' u18', ' u23', ' youth', ' reserve', ' res.', ' ii', ' b team', ' women', ' w')
    low = f" {name.lower()} "
    return not any(b in low for b in banned)

def resolve_senior_team_api(name, key):
    rows = api_get('teams', {'search': name.strip()}, key)
    clean_target = re.sub(r'[^a-z0-9]', '', name.lower())
    cands = []
    for r in rows:
        t = r['team']
        if not is_senior_team(t['name']):
            continue
        clean_cand = re.sub(r'[^a-z0-9]', '', t['name'].lower())
        score = 10 if clean_cand == clean_target else (5 if clean_target in clean_cand else 0)
        cands.append((score, t))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]

def fetch_team_recent_stats_api(team_id, key, last_n=8):
    rows = api_get('fixtures', {'team': team_id, 'last': last_n, 'status': 'FT'}, key)
    if not rows:
        return None
    gf, ga, count = 0, 0, 0
    for m in rows:
        goals = m.get('goals', {})
        if goals.get('home') is None or goals.get('away') is None:
            continue
        if m['teams']['home']['id'] == team_id:
            gf += goals['home']
            ga += goals['away']
        else:
            gf += goals['away']
            ga += goals['home']
        count += 1
    if count == 0:
        return None
    return {'gf': round(gf / count, 2), 'ga': round(ga / count, 2), 'sample_size': count}

def fetch_team_injuries_api(fixture_id, home_id, away_id, key):
    if not fixture_id:
        return [], []
    rows = api_get('injuries', {'fixture': fixture_id}, key)
    h_abs, a_abs = [], []
    for item in rows:
        tid = item.get('team', {}).get('id')
        pname = item.get('player', {}).get('name', 'Unknown')
        ptype = item.get('player', {}).get('type', 'MID')
        entry = {'player': pname, 'position': ptype[:3].upper(), 'importance': 'starter', 'status': 'injured'}
        if tid == home_id:
            h_abs.append(entry)
        elif tid == away_id:
            a_abs.append(entry)
    return h_abs, a_abs

# ---------------- Helper to Run & Log Prediction ----------------
def execute_and_save_model(payload, source_mode, neutral_venue, home_adv, rho, elo_weight, unc, simulation_count=2500, fixture_id=None):
    lh, ma = calculate_expected_goals(payload, elo_weight=elo_weight)
    M = compute_score_matrix(lh, ma, rho=rho, home_adv=home_adv, unc=unc, n=simulation_count)
    res = extract_markets(M)
    res['lh_base'] = lh
    res['ma_base'] = ma
    st.session_state['match_v51'] = payload
    st.session_state['res_v51'] = res

    with db() as con:
        con.execute('''INSERT INTO predictions
            (created_at, source_mode, fixture_id, home, away, neutral_venue,
             p_home, p_draw, p_away, p_over15, p_over25, p_over35, p_btts,
             exp_home, exp_away, odd_h, odd_d, odd_a,
             home_attributes_json, away_attributes_json, home_players_json, away_players_json,
             match_context_json, home_form_json, away_form_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (datetime.now().strftime("%Y-%m-%d %H:%M"), source_mode, fixture_id,
             payload['home']['name'], payload['away']['name'], int(neutral_venue),
             res['p_home'], res['p_draw'], res['p_away'],
             res['p_over15'], res['p_over25'], res['p_over35'], res['p_btts'],
             res['exp_home'], res['exp_away'],
             payload['odds']['home'], payload['odds']['draw'], payload['odds']['away'],
             json.dumps(payload['home'].get('attributes', {}), ensure_ascii=False),
             json.dumps(payload['away'].get('attributes', {}), ensure_ascii=False),
             json.dumps(payload['home'].get('player_ratings', []), ensure_ascii=False),
             json.dumps(payload['away'].get('player_ratings', []), ensure_ascii=False),
             json.dumps(payload.get('match_context', {}), ensure_ascii=False),
             json.dumps({"recent_form": payload['home'].get('recent_form', []), "additional_stats": payload['home'].get('additional_stats', {}), "tactics": payload['home'].get('tactics', {}), "sample_scope": payload['home'].get('sample_scope'), "absences": payload['home'].get('absences', [])}, ensure_ascii=False),
             json.dumps({"recent_form": payload['away'].get('recent_form', []), "additional_stats": payload['away'].get('additional_stats', {}), "tactics": payload['away'].get('tactics', {}), "sample_scope": payload['away'].get('sample_scope'), "absences": payload['away'].get('absences', [])}, ensure_ascii=False))
        )

# ---------------- Sidebar Controls ----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/football-ball-1.png", width=64)
    st.title("Match Lab V6.1")

    api_key_input = st.text_input("API-Football Key (For Live API Mode)", type="password", value=get_api_key())
    st.session_state['api_key'] = api_key_input

    st.divider()
    st.subheader("Model Calibration")
    neutral_venue = st.checkbox("Neutral Venue (Disable Home Advantage)", value=False)
    home_adv_val = 1.00 if neutral_venue else st.slider("Home Advantage Multiplier", 1.00, 1.25, 1.08, 0.01)
    dixon_coles_rho = st.slider("Dixon-Coles Rho (Draw Tuning)", -0.30, 0.00, -0.13, 0.01)
    elo_blend = st.slider("Elo Weight (Prevents Double-Counting)", 0.0, 1.0, 0.35, 0.05)
    uncertainty_val = st.slider("Rate Uncertainty (Sigma)", 0.05, 0.30, 0.15, 0.01)
    simulation_count = st.selectbox(
        "Monte Carlo Simulation Runs",
        options=[2500, 5000, 10000, 25000, 50000],
        index=0,
        format_func=lambda value: f"{value:,} simulations"
    )
    st.caption("More runs reduce simulation noise but do not guarantee more accurate predictions.")

# ---------------- Main UI & Tabs ----------------
st.title("⚽ Football Match Lab V6.1 — Manual-First Match Simulation")
st.caption("Manual-first builder • Lineups & tactical context • Screenshot/API imports • Odds analysis • Historical tracking")

tabs = st.tabs([
    "📥 1. Match Input (Manual / Block / API)",
    "🎲 2. Prediction & Score Matrix",
    "📊 3. Market Odds & EV",
    "🗄️ 4. History & Accuracy"
])

# ---------------- TAB 0: UNIFIED INPUT ----------------
with tabs[0]:
    input_mode = st.radio(
        "Select Data Input System:",
        [
            "Structured JSON / Match Block Paste",
            "Manual Form Builder",
            "SofaScore Match URL Import",
            "API-Football Live Fetch",
            "Screenshot Import (OCR + Review)"
        ],
        horizontal=True
    )

    st.divider()

    # --- MODE A: STRUCTURED BLOCK / JSON PASTE ---
    if input_mode == "Structured JSON / Match Block Paste":
        st.subheader("Paste JSON or Flexible Match Block")
        st.caption("Accepts Home/Away GF, GA, Elo, odds and flexible absence rows such as Player | Position | Importance | Status. Missing core numeric data is still rejected.")

        template_json = """{
  "home": {
    "name": "Italy",
    "gf": 1.67,
    "ga": 1.15,
    "sample_size": 6,
    "elo": 1840,
    "avg_opp_elo": 1780,
    "absences": [
      {"player": "Giovanni Di Lorenzo", "position": "DEF", "importance": "starter"},
      {"player": "Moise Kean", "position": "FWD", "importance": "starter"}
    ]
  },
  "away": {
    "name": "France",
    "gf": 2.10,
    "ga": 0.75,
    "sample_size": 6,
    "elo": 1910,
    "avg_opp_elo": 1790,
    "absences": [
      {"player": "William Saliba", "position": "DEF", "importance": "star"},
      {"player": "Ousmane Dembélé", "position": "FWD", "importance": "star"}
    ]
  },
  "odds": {
    "home": 2.75,
    "draw": 3.20,
    "away": 2.45
  }
}"""
        raw_block = st.text_area("Match Data Block (JSON or Sectioned Text)", value="", placeholder=template_json, height=340)

        if st.button("🚀 Validate & Run Simulation", type="primary", use_container_width=True):
            if not raw_block.strip():
                st.error("Input box is empty. Paste a match JSON or structured block first.")
            else:
                parsed, errs = parse_structured_text_block(raw_block)
                if errs:
                    st.error("❌ Validation Failed — Please fix these fields:")
                    for e in errs:
                        st.write(f"- {e}")
                else:
                    execute_and_save_model(parsed, "Block/JSON", neutral_venue, home_adv_val, dixon_coles_rho, elo_blend, uncertainty_val, simulation_count=simulation_count)
                    st.success(f"✅ Validated & Simulated: **{parsed['home']['name']} vs {parsed['away']['name']}**! Open Tab 2 to view full breakdown.")

    # --- MODE B: MANUAL FORM BUILDER ---
    elif input_mode == "Manual Form Builder":
        st.subheader("V6 Manual Match Builder")
        st.caption("Manual-first workflow. Enter verified information; optional fields can stay blank. Tactical and player details are saved for review but are not converted into uncalibrated probability multipliers.")

        team_tabs = st.tabs(["🏠 Home Team", "✈️ Away Team", "⚔️ Match Context", "💰 Odds & Run"])
        team_payload = {}
        team_rating_payload = {}
        team_squad_payload = {}
        team_form_payload = {}
        team_absence_payload = {}

        for side, tab, label in [("home", team_tabs[0], "Home"), ("away", team_tabs[1], "Away")]:
            with tab:
                st.markdown(f"### {label} team profile")
                c1, c2 = st.columns(2)
                with c1:
                    name = st.text_input(f"{label} Team Name", key=f"v6_{side}_name")
                    gf = st.number_input(f"{label} Avg Goals Scored (GF)", min_value=0.0, max_value=10.0, value=None, step=0.05, placeholder="e.g. 1.65", key=f"v6_{side}_gf")
                    elo = st.number_input(f"{label} Elo Rating", min_value=800.0, max_value=2600.0, value=None, step=5.0, placeholder="e.g. 1850", key=f"v6_{side}_elo")
                    sample = st.number_input(f"{label} Matches in Statistical Sample", min_value=1, max_value=50, value=8, key=f"v6_{side}_sample")
                with c2:
                    ga = st.number_input(f"{label} Avg Goals Conceded (GA)", min_value=0.0, max_value=10.0, value=None, step=0.05, placeholder="e.g. 1.10", key=f"v6_{side}_ga")
                    opp_elo = st.number_input(f"{label} Average Opponent Elo", min_value=800.0, max_value=2600.0, value=1750.0, step=10.0, key=f"v6_{side}_opp_elo")
                    sample_scope = st.selectbox(f"{label} Statistical Sample", ["All competitions", "League only", "Home matches only", "Away matches only", "Last 5", "Last 10", "Last 15", "Custom sample"], key=f"v6_{side}_scope")

                with st.expander(f"{label}: Additional performance statistics (optional)", expanded=True):
                    st.caption("These are stored as context. The current goal model does not add them again on top of GF/GA, which could double-count the same performance.")
                    stat_cols = st.columns(4)
                    optional_stats = {}
                    for i, (key, title, maximum) in enumerate([
                        ("xg", "xG / match", 10.0), ("xga", "xGA / match", 10.0),
                        ("shots", "Shots / match", 60.0), ("shots_on_target", "Shots on target / match", 30.0),
                        ("big_chances", "Big chances / match", 30.0), ("big_chances_conceded", "Big chances conceded", 30.0),
                        ("possession", "Possession %", 100.0), ("clean_sheets_pct", "Clean sheets %", 100.0)
                    ]):
                        optional_stats[key] = stat_cols[i % 4].number_input(f"{label} {title}", min_value=0.0, max_value=maximum, value=None, step=0.1, key=f"v6_{side}_{key}")

                with st.expander(f"{label}: Recent match log", expanded=False):
                    st.caption("Paste all recent results together — no spreadsheet rows. One match per line. Required: date, opponent, venue, goals for and goals against. xG, xGA and opponent Elo are optional.")
                    st.code("Date | Opponent | Venue | GF | GA | xG | xGA | Opponent Elo\n2026-11-20 | Arsenal | Away | 1 | 2 | 0.85 | 1.60 | 1900\n2026-11-08 | Burnley | Home | 3 | 0 | 2.10 | 0.45 | 1500", language="text")
                    recent_text = st.text_area(
                        f"{label} recent matches — paste here",
                        placeholder="Paste 5–10 recent matches, one per line...",
                        height=180, key=f"v6_{side}_form_bulk"
                    )
                    form_rows, form_errors = parse_recent_match_rows(recent_text)
                    for error in form_errors:
                        st.warning(error)
                    team_form_payload[side] = form_rows
                    if form_rows:
                        st.success(f"Loaded {len(form_rows)} recent matches.")
                        st.dataframe(pd.DataFrame(form_rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("No recent matches entered. This section can be left blank.")

                with st.expander(f"{label}: Tactics & formation", expanded=False):
                    t1, t2, t3 = st.columns(3)
                    formation = t1.selectbox(f"{label} Formation", ["Unknown", "4-3-3", "4-2-3-1", "4-4-2", "4-1-4-1", "3-4-2-1", "3-5-2", "3-4-3", "5-3-2", "Other"], key=f"v6_{side}_formation")
                    style = t2.selectbox(f"{label} Playing Style", ["Unknown", "Possession", "High press", "Mid block", "Low block", "Counterattack", "Direct", "Balanced"], key=f"v6_{side}_style")
                    press = t3.selectbox(f"{label} Pressing", ["Unknown", "Low", "Medium", "High", "Very high"], key=f"v6_{side}_press")
                    t4, t5, t6 = st.columns(3)
                    line = t4.selectbox(f"{label} Defensive Line", ["Unknown", "Low", "Medium", "High"], key=f"v6_{side}_line")
                    set_piece = t5.selectbox(f"{label} Set-piece threat", ["Unknown", "Weak", "Average", "Strong"], key=f"v6_{side}_setpiece")
                    transition = t6.selectbox(f"{label} Transition threat", ["Unknown", "Weak", "Average", "Strong"], key=f"v6_{side}_transition")
                    tactics = {"formation": formation, "style": style, "pressing": press, "defensive_line": line, "set_piece": set_piece, "transition": transition}

                with st.expander(f"{label}: Starting XI & substitutes", expanded=False):
                    st.caption("Paste the whole lineup in one go. One player per line; use | between columns. You can paste directly from a text lineup and quickly add positions/roles. Ratings and fitness are optional.")
                    st.code("Player | Position | Role | Rating | Fitness\nGoalkeeper Name | GK | Starter | |\nDefender Name | CB | Starter | |\nMidfielder Name | CM | Starter | |\nForward Name | ST | Starter | |\nSubstitute Name | FWD | Substitute | |", language="text")
                    squad_text = st.text_area(
                        f"{label} full squad — paste lineup here",
                        placeholder="Paste 11 starters and substitutes here...",
                        height=220, key=f"v6_{side}_squad_bulk"
                    )
                    squad_rows, squad_errors = parse_pasted_rows(
                        squad_text,
                        ["player", "position", "role", "rating_0_100", "fitness_0_100"],
                        ["player", "position", "role"],
                        {"position": ["GK", "CB", "LB", "RB", "DEF", "DM", "CM", "AM", "LM", "RM", "MID", "LW", "RW", "ST", "FWD"],
                         "role": ["STARTER", "SUBSTITUTE", "DOUBTFUL"]}
                    )
                    if squad_errors:
                        for error in squad_errors:
                            st.warning(error)
                    team_squad_payload[side] = squad_rows
                    if squad_rows:
                        st.success(f"Loaded {sum(r['role'] == 'Starter' for r in squad_rows)} starters and {sum(r['role'] == 'Substitute' for r in squad_rows)} substitutes.")
                        st.dataframe(pd.DataFrame(squad_rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("No players entered yet. You can leave this section empty.")

                with st.expander(f"{label}: Injuries & suspensions", expanded=False):
                    st.caption("Paste all unavailable players together. One per line: Player | Position | Importance | Status")
                    st.code("Player | Position | Importance | Status\nPlayer Name | DEF | starter | injured\nPlayer Name | FWD | star | suspended", language="text")
                    absence_text = st.text_area(f"{label} unavailable players", height=110, key=f"v6_{side}_absences_bulk")
                    absence_rows, absence_errors = parse_pasted_rows(
                        absence_text, ["player", "position", "importance", "status"],
                        ["player", "position", "importance", "status"],
                        {"position": ["FWD", "MID", "DEF", "GK"],
                         "importance": ["STAR", "STARTER", "ROTATION"],
                         "status": ["INJURED", "SUSPENDED", "DOUBTFUL", "UNKNOWN", "AVAILABLE"]}
                    )
                    if absence_errors:
                        for error in absence_errors:
                            st.warning(error)
                    team_absence_payload[side] = [{**r, "importance": r["importance"].lower(), "status": r["status"].lower()} for r in absence_rows]
                    if absence_rows:
                        st.success(f"Loaded {len(absence_rows)} unavailable-player entries.")
                        st.dataframe(pd.DataFrame(team_absence_payload[side]), use_container_width=True, hide_index=True)

                team_rating_payload[side] = render_team_attribute_ratings(f"V6 {label}")
                team_payload[side] = {
                    "name": name.strip() if name else None, "gf": gf, "ga": ga, "elo": elo,
                    "sample_size": int(sample), "avg_opp_elo": float(opp_elo),
                    "sample_scope": sample_scope, "additional_stats": optional_stats,
                    "tactics": tactics, "absences": team_absence_payload.get(side, [])
                }

        with team_tabs[2]:
            st.markdown("### Match conditions")
            mc1, mc2, mc3 = st.columns(3)
            competition = mc1.text_input("Competition / League", key="v6_competition")
            match_date = mc2.date_input("Match date", value=None, key="v6_match_date")
            importance = mc3.selectbox("Match importance", ["Unknown", "Regular", "Derby", "Title race", "Relegation battle", "Cup knockout", "Final", "Must-win"])
            cc1, cc2, cc3 = st.columns(3)
            home_rest = cc1.number_input("Home rest days", min_value=0, max_value=30, value=7)
            away_rest = cc2.number_input("Away rest days", min_value=0, max_value=30, value=7)
            travel_km = cc3.number_input("Away travel distance (km, if known)", min_value=0, max_value=20000, value=None, step=50)
            weather = st.selectbox("Weather", ["Unknown", "Clear", "Rain", "Heavy rain", "Windy", "Hot", "Cold", "Snow"])
            venue_note = st.text_input("Venue / pitch notes (optional)")
            match_context = {
                "competition": competition, "match_date": str(match_date) if match_date else None,
                "importance": importance, "home_rest_days": int(home_rest), "away_rest_days": int(away_rest),
                "away_travel_km": travel_km, "weather": weather, "venue_notes": venue_note,
                "neutral_venue": bool(neutral_venue)
            }

        with team_tabs[3]:
            st.markdown("### Bookmaker 1X2 decimal odds")
            st.caption("Odds are required by the existing EV comparison. Use the same bookmaker and record them close to kickoff when possible.")
            oc1, oc2, oc3 = st.columns(3)
            m_odd_h = oc1.number_input("Home Odds (1)", min_value=1.01, max_value=100.0, value=None, step=0.05, placeholder="e.g. 2.10", key="v6_odd_h")
            m_odd_d = oc2.number_input("Draw Odds (X)", min_value=1.01, max_value=100.0, value=None, step=0.05, placeholder="e.g. 3.40", key="v6_odd_d")
            m_odd_a = oc3.number_input("Away Odds (2)", min_value=1.01, max_value=100.0, value=None, step=0.05, placeholder="e.g. 3.50", key="v6_odd_a")
            st.info("The current prediction engine remains a team-goal / Dixon–Coles model. Tactics, recent-match logs, weather and player ratings are saved with the prediction for audit, but are not given invented numerical effects. The existing GF/GA, Elo, team attributes and role-based absence logic remain the active inputs.")

            if st.button("🚀 Validate Manual Builder & Run Simulation", type="primary", use_container_width=True):
                manual_payload = {
                    "home": dict(team_payload["home"]),
                    "away": dict(team_payload["away"]),
                    "odds": {"home": m_odd_h, "draw": m_odd_d, "away": m_odd_a},
                    "match_context": match_context
                }
                for side in ("home", "away"):
                    manual_payload[side]["absences"] = team_absence_payload.get(side, [])
                    manual_payload[side]["recent_form"] = team_form_payload.get(side, [])
                    manual_payload[side]["squad"] = team_squad_payload.get(side, [])
                manual_payload = apply_attribute_ratings(
                    manual_payload,
                    team_rating_payload.get("home"), team_rating_payload.get("away"),
                    team_squad_payload.get("home", []), team_squad_payload.get("away", [])
                )
                valid_obj, errs = validate_match_payload(manual_payload)
                if errs:
                    st.error("❌ Missing or invalid required fields:")
                    for e in errs:
                        st.write(f"- {e}")
                else:
                    execute_and_save_model(valid_obj, "Manual V6 Builder", neutral_venue, home_adv_val, dixon_coles_rho, elo_blend, uncertainty_val, simulation_count=simulation_count)
                    st.success(f"✅ Simulation complete: **{valid_obj['home']['name']} vs {valid_obj['away']['name']}**. Open Prediction & Score Matrix for results.")

    # --- MODE C: SOFASCORE MATCH URL IMPORT ---
    elif input_mode == "SofaScore Match URL Import":
        st.subheader("Import Match Data from a SofaScore Link")
        st.caption("Automatically imports team names, recent completed-match GF/GA averages and H2H when available. Elo ratings and bookmaker odds are not reliably supplied by this import, so you must enter them yourself.")
        sofa_url = st.text_input("Paste the full SofaScore match URL", placeholder="https://www.sofascore.com/football/match/...")
        if st.button("🔎 Fetch Match Information from SofaScore", type="primary", use_container_width=True):
            try:
                st.session_state["sofascore_import"] = fetch_sofascore_match(sofa_url.strip())
                st.success("SofaScore match data retrieved. Review the information below.")
            except Exception as exc:
                st.error(f"Could not import this SofaScore link: {exc}")

        sofa_data = st.session_state.get("sofascore_import")
        if sofa_data:
            st.markdown(f"### {sofa_data['event_name']}")
            if sofa_data.get("event_id"):
                st.caption(f"Fixture status: {sofa_data['status']} • Match ID: {sofa_data['event_id']}")
            else:
                st.caption(f"Source: SofaScore Team Compare • Team IDs: {sofa_data['home']['id']} and {sofa_data['away']['id']}")
                st.info("This link compares two teams; it does not identify one specific upcoming match. Recent form and shared past meetings are imported, but fixture date/status are not available from this link.")
            s1, s2 = st.columns(2)
            with s1:
                st.markdown(f"**{sofa_data['home']['name']} — recent {sofa_data['home']['sample_size']} matches**")
                st.write(f"Goals scored per match: **{sofa_data['home']['gf']:.2f}**")
                st.write(f"Goals conceded per match: **{sofa_data['home']['ga']:.2f}**")
            with s2:
                st.markdown(f"**{sofa_data['away']['name']} — recent {sofa_data['away']['sample_size']} matches**")
                st.write(f"Goals scored per match: **{sofa_data['away']['gf']:.2f}**")
                st.write(f"Goals conceded per match: **{sofa_data['away']['ga']:.2f}**")
            if sofa_data["h2h"]:
                with st.expander("SofaScore H2H results"):
                    for item in sofa_data["h2h"]:
                        st.write("- " + item)
            else:
                st.info("H2H results were not available from this SofaScore endpoint.")

            st.warning("Please enter the teams' Elo ratings and decimal 1X2 odds. These are not guessed or silently filled by SofaScore import.")
            ec1, ec2 = st.columns(2)
            with ec1:
                sofa_home_elo = st.number_input(f"{sofa_data['home']['name']} Elo", min_value=800.0, max_value=2600.0, value=None, step=5.0, key="sofa_home_elo")
            with ec2:
                sofa_away_elo = st.number_input(f"{sofa_data['away']['name']} Elo", min_value=800.0, max_value=2600.0, value=None, step=5.0, key="sofa_away_elo")
            oo1, oo2, oo3 = st.columns(3)
            with oo1:
                sofa_odd_h = st.number_input("Home odds (1)", min_value=1.01, max_value=100.0, value=None, step=0.05, key="sofa_odd_h")
            with oo2:
                sofa_odd_d = st.number_input("Draw odds (X)", min_value=1.01, max_value=100.0, value=None, step=0.05, key="sofa_odd_d")
            with oo3:
                sofa_odd_a = st.number_input("Away odds (2)", min_value=1.01, max_value=100.0, value=None, step=0.05, key="sofa_odd_a")

            if st.button("⚽ Run Simulation with Imported Data", type="primary", use_container_width=True):
                payload = {
                    "home": dict(sofa_data["home"]),
                    "away": dict(sofa_data["away"]),
                    "odds": {"home": sofa_odd_h, "draw": sofa_odd_d, "away": sofa_odd_a},
                    "h2h": sofa_data["h2h"],
                }
                payload["home"]["elo"] = sofa_home_elo
                payload["away"]["elo"] = sofa_away_elo
                valid, errors = validate_match_payload(payload)
                if errors:
                    st.error("Please complete the missing required fields:")
                    for error in errors:
                        st.write("- " + error)
                else:
                    execute_and_save_model(valid, "SofaScore URL", neutral_venue, home_adv_val, dixon_coles_rho, elo_blend, uncertainty_val, simulation_count=simulation_count, fixture_id=sofa_data.get("event_id"))
                    st.success(f"Simulation completed: {sofa_data['event_name']}. Open Tab 2 for results.")


    # --- MODE E: SCREENSHOT IMPORT WITH OCR + HUMAN REVIEW ---
    elif input_mode == "Screenshot Import (OCR + Review)":
        st.subheader("Import Match Information from Screenshots")
        st.caption(
            "Upload screenshots from SofaScore or another source. OCR is only a first draft: "
            "check and correct every value before running the simulation."
        )
        uploaded_images = st.file_uploader(
            "Upload match screenshots",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="match_screenshot_uploads"
        )

        if st.button("🔍 Read Text from Screenshots", use_container_width=True):
            if not uploaded_images:
                st.warning("Upload at least one screenshot first.")
            else:
                extracted = []
                for image_file in uploaded_images:
                    try:
                        extracted.append(
                            f"--- {image_file.name} ---\\n{extract_screenshot_text(image_file)}"
                        )
                    except Exception as exc:
                        extracted.append(f"--- {image_file.name} ---\\nOCR unavailable: {exc}")
                st.session_state["screenshot_ocr_text"] = "\\n\\n".join(extracted)

        if uploaded_images:
            with st.expander("View uploaded screenshots"):
                for image_file in uploaded_images:
                    st.image(image_file, caption=image_file.name, use_container_width=True)

        ocr_review_text = st.text_area(
            "OCR text — review and edit this text, then make sure it follows the match block format",
            value=st.session_state.get("screenshot_ocr_text", ""),
            height=260,
            key="screenshot_review_text",
            placeholder=(
                "Home Team: ...\\nHome GF: ...\\nHome GA: ...\\nHome Elo: ...\\n"
                "Away Team: ...\\nAway GF: ...\\nAway GA: ...\\nAway Elo: ...\\n"
                "Odds 1: ...\\nOdds X: ...\\nOdds 2: ..."
            )
        )
        st.info(
            "OCR may misread team names, numbers and decimal odds. Correct the text manually. "
            "The app will not invent missing Elo ratings or bookmaker odds."
        )

        if st.button("Continue to Ratings and Validate Screenshot Data", type="primary", use_container_width=True):
            parsed, parse_errors = parse_structured_text_block(ocr_review_text)
            if parse_errors:
                st.error("The reviewed text is not complete yet. Fix these fields:")
                for err in parse_errors:
                    st.write("- " + err)
            else:
                st.session_state["screenshot_parsed_payload"] = parsed
                st.success("Text parsed. Add optional team/player ratings below, then run the simulation.")

        screenshot_payload = st.session_state.get("screenshot_parsed_payload")
        if screenshot_payload:
            st.markdown(
                f"### {screenshot_payload['home']['name']} vs {screenshot_payload['away']['name']}"
            )
            r1, r2 = st.columns(2)
            with r1:
                screenshot_home_ratings = render_team_attribute_ratings("Screenshot Home")
                screenshot_home_players = render_player_attribute_table("Screenshot Home")
            with r2:
                screenshot_away_ratings = render_team_attribute_ratings("Screenshot Away")
                screenshot_away_players = render_player_attribute_table("Screenshot Away")

            if st.button("⚽ Run Simulation from Reviewed Screenshot Data", type="primary", use_container_width=True):
                final_payload = apply_attribute_ratings(
                    dict(screenshot_payload),
                    screenshot_home_ratings, screenshot_away_ratings,
                    screenshot_home_players.dropna(subset=["player"]).to_dict(orient="records"),
                    screenshot_away_players.dropna(subset=["player"]).to_dict(orient="records")
                )
                execute_and_save_model(
                    final_payload, "Screenshot Import", neutral_venue,
                    home_adv_val, dixon_coles_rho, elo_blend, uncertainty_val, simulation_count=simulation_count
                )
                st.success(
                    f"Simulation completed for {final_payload['home']['name']} vs "
                    f"{final_payload['away']['name']}. Open Tab 2 for results."
                )

    # --- MODE D: API-FOOTBALL LIVE FETCH ---
    else:
        st.subheader("API-Football Automatic Senior-Team Pipeline")
        st.caption("Fetches real recent GF/GA and confirmed injury lists via API-Football, combined with your Elo & odds inputs.")

        ac1, ac2 = st.columns(2)
        with ac1:
            api_h_name = st.text_input("Home Senior Team Name", placeholder="e.g. Arsenal")
            api_h_elo = st.number_input("Home Elo Rating (from ClubElo / EloRatings)", min_value=800.0, max_value=2600.0, value=None, step=5.0)
        with ac2:
            api_a_name = st.text_input("Away Senior Team Name", placeholder="e.g. Chelsea")
            api_a_elo = st.number_input("Away Elo Rating (from ClubElo / EloRatings)", min_value=800.0, max_value=2600.0, value=None, step=5.0)

        ao1, ao2, ao3 = st.columns(3)
        api_odd_h = ao1.number_input("Bookmaker Home Odds (1)", min_value=1.01, max_value=100.0, value=None, step=0.05)
        api_odd_d = ao2.number_input("Bookmaker Draw Odds (X)", min_value=1.01, max_value=100.0, value=None, step=0.05)
        api_odd_a = ao3.number_input("Bookmaker Away Odds (2)", min_value=1.01, max_value=100.0, value=None, step=0.05)

        if st.button("🌐 Fetch Live API Stats & Run Simulation", type="primary", use_container_width=True):
            key = get_api_key()
            if not key:
                st.error("Enter your API-Football key in the left sidebar first.")
            elif not api_h_name.strip() or not api_a_name.strip():
                st.error("Enter both Home and Away team names.")
            else:
                try:
                    with st.spinner("Resolving senior teams, fetching recent fixtures & injury reports..."):
                        ht = resolve_senior_team_api(api_h_name, key)
                        at = resolve_senior_team_api(api_a_name, key)
                        if not ht or not at:
                            st.error("Could not resolve one or both senior teams on API-Football.")
                        else:
                            h_stats = fetch_team_recent_stats_api(ht['id'], key)
                            a_stats = fetch_team_recent_stats_api(at['id'], key)
                            if not h_stats or not a_stats:
                                st.error("Insufficient completed match history returned by API-Football.")
                            else:
                                # Check if an upcoming head-to-head fixture exists to pull live injury list
                                next_fix = api_get('fixtures', {'team': ht['id'], 'next': 10}, key)
                                fix_id = None
                                for fx in next_fix:
                                    if fx['teams']['home']['id'] == ht['id'] and fx['teams']['away']['id'] == at['id']:
                                        fix_id = fx['fixture']['id']
                                        break
                                h_abs, a_abs = fetch_team_injuries_api(fix_id, ht['id'], at['id'], key)

                                api_payload = {
                                    'home': {
                                        'name': ht['name'], 'gf': h_stats['gf'], 'ga': h_stats['ga'],
                                        'sample_size': h_stats['sample_size'], 'elo': api_h_elo,
                                        'avg_opp_elo': 1750.0, 'absences': h_abs
                                    },
                                    'away': {
                                        'name': at['name'], 'gf': a_stats['gf'], 'ga': a_stats['ga'],
                                        'sample_size': a_stats['sample_size'], 'elo': api_a_elo,
                                        'avg_opp_elo': 1750.0, 'absences': a_abs
                                    },
                                    'odds': {'home': api_odd_h, 'draw': api_odd_d, 'away': api_odd_a}
                                }
                                valid_obj, errs = validate_match_payload(api_payload)
                                if errs:
                                    st.error("❌ Missing or Invalid Fields:")
                                    for e in errs:
                                        st.write(f"- {e}")
                                else:
                                    execute_and_save_model(valid_obj, "API-Football", neutral_venue, home_adv_val, dixon_coles_rho, elo_blend, uncertainty_val, simulation_count=simulation_count, fixture_id=fix_id)
                                    st.success(f"✅ API Data Verified & Simulated: **{ht['name']} vs {at['name']}** (Fixture ID: {fix_id or 'Unscheduled'})")
                except Exception as exc:
                    st.error(f"API Fetch Error: {exc}")

# ---------------- TAB 1: PREDICTION & SCORE MATRIX ----------------
with tabs[1]:
    st.subheader("Vectorized Dixon-Coles Probabilities & Scorelines")
    data = st.session_state.get('match_v51')
    res = st.session_state.get('res_v51')

    if not data or not res:
        st.info("Validate and run a match in Tab 1 first.")
    else:
        h_name, a_name = data['home']['name'], data['away']['name']

        # Allow instant recalculation if sidebar sliders changed
        if st.button("🔄 Recalculate with Current Sidebar Settings"):
            lh, ma = calculate_expected_goals(data, elo_weight=elo_blend)
            M = compute_score_matrix(lh, ma, rho=dixon_coles_rho, home_adv=home_adv_val, unc=uncertainty_val, n=simulation_count)
            res = extract_markets(M)
            st.session_state['res_v51'] = res

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{h_name} Elo", f"{data['home']['elo']}")
        c2.metric(f"{a_name} Elo", f"{data['away']['elo']}")
        c3.metric("Venue Mode", "Neutral (1.00x)" if neutral_venue else f"Home Adv ({home_adv_val:.2f}x)")
        c4.metric("Absences Logged", f"{len(data['home'].get('absences', []))} H / {len(data['away'].get('absences', []))} A")

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric(f"🏠 {h_name} Win (1)", f"{res['p_home']:.1%}")
        m2.metric("🤝 Draw (X)", f"{res['p_draw']:.1%}")
        m3.metric(f"✈️ {a_name} Win (2)", f"{res['p_away']:.1%}")

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Over 1.5 Goals", f"{res['p_over15']:.1%}")
        g2.metric("Over 2.5 Goals", f"{res['p_over25']:.1%}")
        g3.metric("Over 3.5 Goals", f"{res['p_over35']:.1%}")
        g4.metric("BTTS Yes", f"{res['p_btts']:.1%}")

        st.markdown(f"**Model Expected Goals (xG):** {h_name} **{res['exp_home']:.2f}** — **{res['exp_away']:.2f}** {a_name}")
        ctx = data.get("match_context", {})
        if ctx:
            st.caption(f"Context: {ctx.get('competition') or 'Competition not supplied'} • Importance: {ctx.get('importance', 'Unknown')} • Rest days H/A: {ctx.get('home_rest_days', '—')}/{ctx.get('away_rest_days', '—')}")
        for side, team_name in [("home", h_name), ("away", a_name)]:
            squad = data.get(side, {}).get("squad", data.get(side, {}).get("player_ratings", []))
            if squad:
                starters = sum(1 for p in squad if str(p.get("role", "")).lower() == "starter")
                st.caption(f"{team_name}: {len(squad)} squad entries recorded ({starters} marked as starters). Squad entries are contextual only in this model version.")

        st.markdown("### Most Likely Exact Scorelines")
        st.dataframe(res['scores'].style.format({'Probability': '{:.2%}'}), hide_index=True, use_container_width=True)

# ---------------- TAB 2: MARKET ODDS & EV ----------------
with tabs[2]:
    st.subheader("Market Implied Probability, Edge & Expected Value (EV)")
    data = st.session_state.get('match_v51')
    res = st.session_state.get('res_v51')

    if not data or not res:
        st.info("Run a prediction first to compare against bookmaker odds.")
    else:
        oh, od, oa = data['odds']['home'], data['odds']['draw'], data['odds']['away']
        raw_h, raw_d, raw_a = 1.0 / oh, 1.0 / od, 1.0 / oa
        overround = raw_h + raw_d + raw_a
        vig_pct = (overround - 1.0) * 100.0

        imp_h, imp_d, imp_a = raw_h / overround, raw_d / overround, raw_a / overround
        st.write(f"**Bookmaker Margin (Overround):** `{vig_pct:.2f}%`")

        outcomes = [
            (f"{data['home']['name']} Win (1)", oh, res['p_home'], imp_h),
            ("Draw (X)", od, res['p_draw'], imp_d),
            (f"{data['away']['name']} Win (2)", oa, res['p_away'], imp_a),
        ]

        rows_out = []
        large_gap = False
        for label, odds, mod_p, mkt_p in outcomes:
            diff_pts = (mod_p - mkt_p) * 100.0
            ev_pct = ((mod_p * odds) - 1.0) * 100.0
            if abs(diff_pts) > 10.0:
                large_gap = True
                flag = "⚠️ >10% Gap (Audit Inputs)"
            elif ev_pct > 2.0:
                flag = "✅ Positive EV"
            else:
                flag = "— No Edge"

            rows_out.append({
                "Outcome": label,
                "Decimal Odds": f"{odds:.2f}",
                "Model Prob": f"{mod_p:.1%}",
                "No-Vig Market": f"{mkt_p:.1%}",
                "Edge (Diff)": f"{diff_pts:+.1f}%",
                "Expected Value (EV)": f"{ev_pct:+.1f}%",
                "Assessment": flag
            })

        st.dataframe(pd.DataFrame(rows_out), hide_index=True, use_container_width=True)

        if large_gap:
            st.warning("⚠️ **Model-vs-Market Divergence Alert:** One or more outcomes differ from the no-vig market by more than 10 percentage points. Check whether Elo ratings, neutral venue status, or key injury weights need adjustment.")

# ---------------- TAB 3: HISTORY & CALIBRATION ----------------
with tabs[3]:
    st.subheader("🗄️ Prediction Log & Brier Accuracy Tracking")
    with db() as con:
        rows = con.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()

    if not rows:
        st.info("No predictions recorded yet.")
    else:
        df_hist = pd.DataFrame([dict(r) for r in rows])
        st.dataframe(
            df_hist[['id', 'created_at', 'source_mode', 'home', 'away', 'p_home', 'p_draw', 'p_away', 'odd_h', 'odd_d', 'odd_a', 'actual_home', 'actual_away']],
            hide_index=True,
            use_container_width=True
        )

        st.divider()
        st.markdown("### Settle Completed Match Result")
        unsettled = [r for r in rows if r['actual_home'] is None]
        if unsettled:
            opts = {f"#{r['id']} | {r['home']} vs {r['away']} ({r['created_at']})": r['id'] for r in unsettled}
            choice = st.selectbox("Select Match to Settle", list(opts.keys()))
            sc1, sc2 = st.columns(2)
            fin_h = sc1.number_input("Final Home Goals", min_value=0, max_value=25, value=0)
            fin_a = sc2.number_input("Final Away Goals", min_value=0, max_value=25, value=0)

            if st.button("💾 Save Final Score & Update Accuracy"):
                settled_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                with db() as con:
                    con.execute(
                        "UPDATE predictions SET actual_home = ?, actual_away = ?, settled_at = ? WHERE id = ?",
                        (int(fin_h), int(fin_a), settled_time, opts[choice])
                    )
                settle_market_records(opts[choice], int(fin_h), int(fin_a), settled_time)
                st.success("Result and all three goal markets settled!")
                st.rerun()

        settled = [r for r in rows if r['actual_home'] is not None]
        if settled:
            brier_vals = []
            for r in settled:
                yh = 1.0 if r['actual_home'] > r['actual_away'] else 0.0
                yd = 1.0 if r['actual_home'] == r['actual_away'] else 0.0
                ya = 1.0 if r['actual_home'] < r['actual_away'] else 0.0
                brier_vals.append(((r['p_home'] - yh) ** 2 + (r['p_draw'] - yd) ** 2 + (r['p_away'] - ya) ** 2) / 3.0)
            st.metric(
                "Multi-Class Brier Score (Lower = Better | Random Baseline = 0.222)",
                f"{np.mean(brier_vals):.4f}",
                delta=f"{len(settled)} settled matches"
            )

        # ---------------- Individual Market Tracking ----------------
        st.divider()
        st.markdown("### Individual Market Performance")
        st.caption("Each settled match is scored separately for Over 1.5, Over 2.5 and BTTS. The selected side is the more likely side from the model (Yes or No).")
        with db() as con:
            market_rows = con.execute("SELECT * FROM market_settlements ORDER BY prediction_id DESC, market").fetchall()
        if market_rows:
            market_df = pd.DataFrame([dict(r) for r in market_rows])
            summary_rows = []
            for market_name in ["Over 1.5", "Over 2.5", "BTTS"]:
                part = market_df[market_df["market"] == market_name].copy()
                if part.empty:
                    continue
                wins = int((part["result"] == "WIN").sum())
                losses = int((part["result"] == "LOSS").sum())
                # Binary Brier score uses the model's probability of the Yes outcome.
                brier = float(np.mean((part["predicted_yes_probability"] - part["actual_yes"]) ** 2))
                summary_rows.append({
                    "Market": market_name,
                    "Settled": len(part),
                    "Wins": wins,
                    "Losses": losses,
                    "Selection Win Rate": f"{wins / len(part):.1%}",
                    "Binary Brier (lower is better)": f"{brier:.4f}"
                })
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

            st.markdown("#### Confidence Calibration by Market")
            calibration_rows = []
            for market_name in ["Over 1.5", "Over 2.5", "BTTS"]:
                part = market_df[market_df["market"] == market_name].copy()
                if part.empty:
                    continue
                part["confidence_band"] = pd.cut(
                    part["selected_probability"],
                    bins=[0.499999, 0.60, 0.70, 0.80, 0.90, 1.000001],
                    labels=["50–60%", "60–70%", "70–80%", "80–90%", "90–100%"],
                    include_lowest=True
                )
                for band, group in part.groupby("confidence_band", observed=False):
                    if len(group) == 0:
                        continue
                    calibration_rows.append({
                        "Market": market_name,
                        "Confidence Band": str(band),
                        "Predictions": len(group),
                        "Average Predicted Confidence": f"{group['selected_probability'].mean():.1%}",
                        "Actual Win Rate": f"{(group['result'] == 'WIN').mean():.1%}"
                    })
            if calibration_rows:
                st.dataframe(pd.DataFrame(calibration_rows), hide_index=True, use_container_width=True)

            st.markdown("#### Individual Market Betting History")
            display_market = market_df.merge(
                df_hist[["id", "created_at", "home", "away"]],
                left_on="prediction_id", right_on="id", how="left"
            )
            st.dataframe(
                display_market[["prediction_id", "created_at", "home", "away", "market", "selection", "selected_probability", "actual_yes", "result"]].rename(columns={
                    "prediction_id": "Prediction ID",
                    "created_at": "Date",
                    "home": "Home",
                    "away": "Away",
                    "market": "Market",
                    "selection": "Selection",
                    "selected_probability": "Confidence",
                    "actual_yes": "Actual Yes (1/0)",
                    "result": "Result"
                }).style.format({"Confidence": "{:.1%}"}),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No settled market records yet. Settle a completed match above to start market-by-market tracking.")

