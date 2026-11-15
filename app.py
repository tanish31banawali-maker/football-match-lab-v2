import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import math
import os
import re
import urllib.parse
import requests
from datetime import datetime

# ---------------- App & Database Configuration ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, 'football_lab_v5_1.db')
API_BASE = 'https://v3.football.api-sports.io'

st.set_page_config(
    page_title='Football Match Lab V5.1',
    page_icon='⚽',
    layout='wide',
    initial_sidebar_state='expanded'
)

def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

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
            settled_at TEXT
        )''')

init_db()

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

    # Role-based & importance-weighted absence impact
    imp_weights = {'star': 0.06, 'starter': 0.035, 'rotation': 0.015}

    def calc_absence_effects(absences):
        att_pen, def_leak = 0.0, 0.0
        for ab in absences:
            w = imp_weights.get(str(ab.get('importance', 'starter')).lower(), 0.035)
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

def parse_structured_text_block(raw_text):
    """
    Strict parser for sectioned match blocks. Starts every field as None,
    reads strictly after colons, uses word boundaries, and resets sections cleanly.
    """
    text = raw_text.strip()
    if text.startswith('{'):
        try:
            obj = json.loads(text)
            return validate_match_payload(obj)
        except json.JSONDecodeError as e:
            return None, [f"JSON syntax error: {e}"]

    payload = {
        'home': {'name': None, 'gf': None, 'ga': None, 'elo': None, 'sample_size': 6, 'avg_opp_elo': 1750.0, 'absences': []},
        'away': {'name': None, 'gf': None, 'ga': None, 'elo': None, 'sample_size': 6, 'avg_opp_elo': 1750.0, 'absences': []},
        'odds': {'home': None, 'draw': None, 'away': None},
        'h2h': []
    }

    section = None
    active_side = None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for ln in lines:
        low = ln.lower()

        # Reset section headers explicitly
        if re.match(r'^(1\.|2\.|3\.|4\.|5\.|6\.)', low) or low.startswith('---'):
            section = None
            active_side = None
            if 'base goal' in low or 'performance' in low or 'ratings' in low:
                section = 'stats'
            elif 'lineup' in low or 'formation' in low:
                section = 'lineups'
            elif 'absence' in low or 'injur' in low:
                section = 'absences'
            elif 'head-to-head' in low or re.search(r'\bh2h\b', low):
                section = 'h2h'
            elif 'odds' in low or 'market' in low:
                section = 'odds'
            continue

        # Explicit team declarations
        if re.match(r'^home\s*team\s*:', low):
            payload['home']['name'] = ln.split(':', 1)[1].strip()
            active_side = 'home'
            continue
        if re.match(r'^away\s*team\s*:', low):
            payload['away']['name'] = ln.split(':', 1)[1].strip()
            active_side = 'away'
            continue

        # Detect standalone team subheaders inside stats or absences
        if ':' not in ln and not ln.startswith(('-', '*')) and section in ('stats', 'absences', None):
            cand = ln.strip()
            if payload['home']['name'] is None:
                payload['home']['name'] = cand
                active_side = 'home'
                continue
            elif payload['away']['name'] is None and cand.lower() != payload['home']['name'].lower():
                payload['away']['name'] = cand
                active_side = 'away'
                continue
            elif payload['home']['name'] and cand.lower() == payload['home']['name'].lower():
                active_side = 'home'
                continue
            elif payload['away']['name'] and cand.lower() == payload['away']['name'].lower():
                active_side = 'away'
                continue

        # Parse Stats (strictly after colon + word boundaries)
        if active_side in ('home', 'away') and ':' in ln and section in ('stats', None):
            if re.search(r'\b(gf|goals\s+scored)\b', low):
                val = parse_number_after_colon(ln)
                if val is not None:
                    payload[active_side]['gf'] = val
                continue
            if re.search(r'\b(ga|goals\s+conceded)\b', low):
                val = parse_number_after_colon(ln)
                if val is not None:
                    payload[active_side]['ga'] = val
                continue
            if re.search(r'\belo\b', low):
                val = parse_number_after_colon(ln)
                if val is not None:
                    payload[active_side]['elo'] = val
                continue

        # Parse Absences (only while inside absences section and active_side is known)
        if section == 'absences' and active_side in ('home', 'away') and ln.startswith(('-', '*')):
            item_txt = ln.lstrip('-* ').strip()
            pos = 'MID'
            if re.search(r'\b(fwd|forward|striker|st|att)\b', item_txt, re.I):
                pos = 'FWD'
            elif re.search(r'\b(def|defender|gk|goalkeeper)\b', item_txt, re.I):
                pos = 'DEF'
            imp = 'star' if re.search(r'\bstar\b', item_txt, re.I) else 'starter'
            player_name = re.split(r'\(|–|-', item_txt)[0].strip()
            payload[active_side]['absences'].append({'player': player_name, 'position': pos, 'importance': imp})
            continue

        # Parse H2H
        if section == 'h2h' and ln.startswith(('-', '*')):
            payload['h2h'].append(ln.lstrip('-* ').strip())
            continue

        # Parse Explicit Odds Labels
        if ':' in ln and (section == 'odds' or re.search(r'\bodds\b', low)):
            val = parse_number_after_colon(ln)
            if val is not None:
                if re.search(r'\b(home\s*odds|odds\s*1)\b', low):
                    payload['odds']['home'] = val
                elif re.search(r'\b(draw\s*odds|odds\s*x)\b', low):
                    payload['odds']['draw'] = val
                elif re.search(r'\b(away\s*odds|odds\s*2)\b', low):
                    payload['odds']['away'] = val

    return validate_match_payload(payload)

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
        entry = {'player': pname, 'position': ptype[:3].upper(), 'importance': 'starter'}
        if tid == home_id:
            h_abs.append(entry)
        elif tid == away_id:
            a_abs.append(entry)
    return h_abs, a_abs

# ---------------- Helper to Run & Log Prediction ----------------
def execute_and_save_model(payload, source_mode, neutral_venue, home_adv, rho, elo_weight, unc, fixture_id=None):
    lh, ma = calculate_expected_goals(payload, elo_weight=elo_weight)
    M = compute_score_matrix(lh, ma, rho=rho, home_adv=home_adv, unc=unc)
    res = extract_markets(M)
    res['lh_base'] = lh
    res['ma_base'] = ma
    st.session_state['match_v51'] = payload
    st.session_state['res_v51'] = res

    with db() as con:
        con.execute('''INSERT INTO predictions
            (created_at, source_mode, fixture_id, home, away, neutral_venue,
             p_home, p_draw, p_away, p_over15, p_over25, p_over35, p_btts,
             exp_home, exp_away, odd_h, odd_d, odd_a)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (datetime.now().strftime("%Y-%m-%d %H:%M"), source_mode, fixture_id,
             payload['home']['name'], payload['away']['name'], int(neutral_venue),
             res['p_home'], res['p_draw'], res['p_away'],
             res['p_over15'], res['p_over25'], res['p_over35'], res['p_btts'],
             res['exp_home'], res['exp_away'],
             payload['odds']['home'], payload['odds']['draw'], payload['odds']['away'])
        )

# ---------------- Sidebar Controls ----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/football-ball-1.png", width=64)
    st.title("Match Lab V5.1")

    api_key_input = st.text_input("API-Football Key (For Live API Mode)", type="password", value=get_api_key())
    st.session_state['api_key'] = api_key_input

    st.divider()
    st.subheader("Model Calibration")
    neutral_venue = st.checkbox("Neutral Venue (Disable Home Advantage)", value=False)
    home_adv_val = 1.00 if neutral_venue else st.slider("Home Advantage Multiplier", 1.00, 1.25, 1.08, 0.01)
    dixon_coles_rho = st.slider("Dixon-Coles Rho (Draw Tuning)", -0.30, 0.00, -0.13, 0.01)
    elo_blend = st.slider("Elo Weight (Prevents Double-Counting)", 0.0, 1.0, 0.35, 0.05)
    uncertainty_val = st.slider("Rate Uncertainty (Sigma)", 0.05, 0.30, 0.15, 0.01)

# ---------------- Main UI & Tabs ----------------
st.title("⚽ Football Match Lab V5.1 — Unified Calibrated Engine")
st.caption("Strict Validation • Manual / Structured Block / Live API Modes • Vectorized Dixon-Coles • Brier Calibration")

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
            "API-Football Live Fetch"
        ],
        horizontal=True
    )

    st.divider()

    # --- MODE A: STRUCTURED BLOCK / JSON PASTE ---
    if input_mode == "Structured JSON / Match Block Paste":
        st.subheader("Paste JSON or Strict Sectioned Match Block")
        st.caption("No hidden defaults. Missing or 'Unverified' fields will be rejected with an exact error list, and valid inputs will trigger the simulation automatically.")

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
                    execute_and_save_model(parsed, "Block/JSON", neutral_venue, home_adv_val, dixon_coles_rho, elo_blend, uncertainty_val)
                    st.success(f"✅ Validated & Simulated: **{parsed['home']['name']} vs {parsed['away']['name']}**! Open Tab 2 to view full breakdown.")

    # --- MODE B: MANUAL FORM BUILDER ---
    elif input_mode == "Manual Form Builder":
        st.subheader("Manual Match Entry Form")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🏠 Home Team")
            m_h_name = st.text_input("Home Team Name", value="")
            m_h_gf = st.number_input("Home Avg Goals Scored (GF)", min_value=0.0, max_value=10.0, value=None, step=0.05, placeholder="e.g. 1.80")
            m_h_ga = st.number_input("Home Avg Goals Conceded (GA)", min_value=0.0, max_value=10.0, value=None, step=0.05, placeholder="e.g. 1.05")
            m_h_elo = st.number_input("Home Elo Rating", min_value=800.0, max_value=2600.0, value=None, step=5.0, placeholder="e.g. 1850")
            m_h_sample = st.number_input("Home Matches Sample Size", min_value=1, max_value=50, value=6)
            m_h_opp_elo = st.number_input("Home Avg Opponent Elo", min_value=800.0, max_value=2600.0, value=1750.0, step=10.0)

        with c2:
            st.markdown("### ✈️ Away Team")
            m_a_name = st.text_input("Away Team Name", value="")
            m_a_gf = st.number_input("Away Avg Goals Scored (GF)", min_value=0.0, max_value=10.0, value=None, step=0.05, placeholder="e.g. 1.50")
            m_a_ga = st.number_input("Away Avg Goals Conceded (GA)", min_value=0.0, max_value=10.0, value=None, step=0.05, placeholder="e.g. 1.20")
            m_a_elo = st.number_input("Away Elo Rating", min_value=800.0, max_value=2600.0, value=None, step=5.0, placeholder="e.g. 1810")
            m_a_sample = st.number_input("Away Matches Sample Size", min_value=1, max_value=50, value=6)
            m_a_opp_elo = st.number_input("Away Avg Opponent Elo", min_value=800.0, max_value=2600.0, value=1750.0, step=10.0)

        st.markdown("### 🏥 Role-Based Key Absences")
        ac1, ac2 = st.columns(2)
        with ac1:
            st.caption("Home Absences")
            h_abs_df = st.data_editor(
                pd.DataFrame(columns=["player", "position", "importance"]),
                column_config={
                    "position": st.column_config.SelectboxColumn("Position", options=["FWD", "MID", "DEF", "GK"], required=True),
                    "importance": st.column_config.SelectboxColumn("Importance", options=["star", "starter", "rotation"], required=True),
                },
                num_rows="dynamic",
                key="manual_h_abs",
                use_container_width=True
            )
        with ac2:
            st.caption("Away Absences")
            a_abs_df = st.data_editor(
                pd.DataFrame(columns=["player", "position", "importance"]),
                column_config={
                    "position": st.column_config.SelectboxColumn("Position", options=["FWD", "MID", "DEF", "GK"], required=True),
                    "importance": st.column_config.SelectboxColumn("Importance", options=["star", "starter", "rotation"], required=True),
                },
                num_rows="dynamic",
                key="manual_a_abs",
                use_container_width=True
            )

        st.markdown("### 💰 Bookmaker 1X2 Decimal Odds")
        oc1, oc2, oc3 = st.columns(3)
        m_odd_h = oc1.number_input("Home Odds (1)", min_value=1.01, max_value=100.0, value=None, step=0.05, placeholder="e.g. 2.10")
        m_odd_d = oc2.number_input("Draw Odds (X)", min_value=1.01, max_value=100.0, value=None, step=0.05, placeholder="e.g. 3.40")
        m_odd_a = oc3.number_input("Away Odds (2)", min_value=1.01, max_value=100.0, value=None, step=0.05, placeholder="e.g. 3.50")

        if st.button("🚀 Validate Manual Form & Run Simulation", type="primary", use_container_width=True):
            manual_payload = {
                'home': {
                    'name': m_h_name.strip() if m_h_name else None,
                    'gf': m_h_gf, 'ga': m_h_ga, 'elo': m_h_elo,
                    'sample_size': int(m_h_sample), 'avg_opp_elo': float(m_h_opp_elo),
                    'absences': h_abs_df.dropna(subset=['player']).to_dict(orient='records')
                },
                'away': {
                    'name': m_a_name.strip() if m_a_name else None,
                    'gf': m_a_gf, 'ga': m_a_ga, 'elo': m_a_elo,
                    'sample_size': int(m_a_sample), 'avg_opp_elo': float(m_a_opp_elo),
                    'absences': a_abs_df.dropna(subset=['player']).to_dict(orient='records')
                },
                'odds': {'home': m_odd_h, 'draw': m_odd_d, 'away': m_odd_a}
            }
            valid_obj, errs = validate_match_payload(manual_payload)
            if errs:
                st.error("❌ Missing or Invalid Fields:")
                for e in errs:
                    st.write(f"- {e}")
            else:
                execute_and_save_model(valid_obj, "Manual Form", neutral_venue, home_adv_val, dixon_coles_rho, elo_blend, uncertainty_val)
                st.success(f"✅ Simulation Complete for **{valid_obj['home']['name']} vs {valid_obj['away']['name']}**! Open Tab 2 for full results.")

    # --- MODE C: API-FOOTBALL LIVE FETCH ---
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
                                    execute_and_save_model(valid_obj, "API-Football", neutral_venue, home_adv_val, dixon_coles_rho, elo_blend, uncertainty_val, fixture_id=fix_id)
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
            M = compute_score_matrix(lh, ma, rho=dixon_coles_rho, home_adv=home_adv_val, unc=uncertainty_val)
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
                with db() as con:
                    con.execute(
                        "UPDATE predictions SET actual_home = ?, actual_away = ?, settled_at = ? WHERE id = ?",
                        (int(fin_h), int(fin_a), datetime.now().strftime("%Y-%m-%d %H:%M"), opts[choice])
                    )
                st.success("Result settled!")
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
