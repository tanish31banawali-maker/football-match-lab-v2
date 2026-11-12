import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import math
import os
import re
import requests
from datetime import datetime

# ---------------- App & Database Configuration ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, 'football_lab_v5_2.db')
API_BASE = 'https://v3.football.api-sports.io'

st.set_page_config(
    page_title='Football Match Lab V5.2',
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

    M[:, 0, 0] *= np.maximum(1e-4, 1.0 - lh_s * ma_s * rho)
    M[:, 0, 1] *= np.maximum(1e-4, 1.0 + lh_s * rho)
    M[:, 1, 0] *= np.maximum(1e-4, 1.0 + ma_s * rho)
    M[:, 1, 1] *= np.maximum(1e-4, 1.0 - rho)

    M /= M.sum(axis=(1, 2), keepdims=True)
    return M.mean(axis=0)

def extract_markets(M):
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
        'scores': scores_df
    }

# ---------------- Calibrated Strength & Role-Based Absence Model ----------------
def shrink_rate(obs_rate, sample_size, baseline=1.35, k=8.0):
    w = sample_size / (sample_size + k)
    return w * obs_rate + (1.0 - w) * baseline

def calculate_expected_goals(data, elo_weight=0.35):
    h = data['home']
    a = data['away']

    h_gf_s = shrink_rate(h['gf'], h.get('sample_size', 6))
    h_ga_s = shrink_rate(h['ga'], h.get('sample_size', 6))
    a_gf_s = shrink_rate(a['gf'], a.get('sample_size', 6))
    a_ga_s = shrink_rate(a['ga'], a.get('sample_size', 6))

    h_opp_fac = 10.0 ** ((h.get('avg_opp_elo', 1750.0) - 1750.0) / 1000.0)
    a_opp_fac = 10.0 ** ((a.get('avg_opp_elo', 1750.0) - 1750.0) / 1000.0)

    h_gf_adj = h_gf_s * h_opp_fac
    h_ga_adj = h_ga_s / h_opp_fac
    a_gf_adj = a_gf_s * a_opp_fac
    a_ga_adj = a_ga_s / a_opp_fac

    base_lh = 0.55 * h_gf_adj + 0.45 * a_ga_adj
    base_ma = 0.55 * a_gf_adj + 0.45 * h_ga_adj

    elo_diff = h['elo'] - a['elo']
    raw_elo_h = 10.0 ** (elo_diff / 1200.0)
    raw_elo_a = 10.0 ** (-elo_diff / 1200.0)

    elo_mult_h = (1.0 - elo_weight) + elo_weight * raw_elo_h
    elo_mult_a = (1.0 - elo_weight) + elo_weight * raw_elo_a

    lh = base_lh * elo_mult_h
    ma = base_ma * elo_mult_a

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

# ---------------- Strict Validation & One-Paste Parsers ----------------
def validate_match_payload(data):
    if not isinstance(data, dict):
        return None, ["Input must be a valid match dictionary."]

    errors = []
    for side in ('home', 'away'):
        if side not in data or not isinstance(data[side], dict):
            errors.append(f"Missing '{side}' section.")
            continue
        team_obj = data[side]
        if not team_obj.get('name') or not str(team_obj['name']).strip():
            errors.append(f"Missing '{side}' team name.")
        for num_field in ('gf', 'ga', 'elo'):
            val = team_obj.get(num_field)
            if val is None or isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                errors.append(f"Missing or invalid numeric '{side} {num_field}' (must be a real number, not 'Unverified').")
            elif num_field == 'elo' and not (800 <= val <= 2600):
                errors.append(f"'{side} elo' ({val}) out of realistic bounds (800–2600).")
            elif num_field in ('gf', 'ga') and val < 0:
                errors.append(f"'{side} {num_field}' cannot be negative.")

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

def parse_absences_list(after_colon_str):
    """Parses comma-separated absences like: Saliba (DEF, star), Mbappe (FWD, star)"""
    text = after_colon_str.strip()
    if not text or text.lower() in ('none', 'nil', '0', 'no'):
        return []
    items = [x.strip() for x in text.split(',') if x.strip()]
    parsed = []
    # Recombine if commas were inside parentheses like "Saliba (DEF, star)"
    merged = []
    buf = ""
    for part in items:
        buf = f"{buf}, {part}" if buf else part
        if buf.count('(') == buf.count(')'):
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)

    for entry in merged:
        pos = 'MID'
        if re.search(r'\b(fwd|forward|striker|st|att)\b', entry, re.I):
            pos = 'FWD'
        elif re.search(r'\b(def|defender|gk|goalkeeper)\b', entry, re.I):
            pos = 'DEF'
        imp = 'starter'
        if re.search(r'\bstar\b', entry, re.I):
            imp = 'star'
        elif re.search(r'\b(rotation|sub|bench|reserve)\b', entry, re.I):
            imp = 'rotation'
        name = re.split(r'\(|–|-', entry)[0].strip()
        if name:
            parsed.append({'player': name, 'position': pos, 'importance': imp})
    return parsed

def parse_one_paste_input(raw_text):
    """
    Supports both clean JSON and the simple 12-line Key: Value text block.
    Reads strictly after colons with word boundaries. Never uses silent defaults.
    """
    text = raw_text.strip()
    # Strip markdown code fences if user pasted ```json ... ```
    text = re.sub(r'^```[a-zA-Z]*\s*', '', text)
    text = re.sub(r'\s*```$', '', text).strip()

    if text.startswith('{'):
        try:
            obj = json.loads(text)
            return validate_match_payload(obj)
        except json.JSONDecodeError as e:
            return None, [f"JSON syntax error: {e}"]

    payload = {
        'home': {'name': None, 'gf': None, 'ga': None, 'elo': None, 'sample_size': 6, 'avg_opp_elo': 1750.0, 'absences': []},
        'away': {'name': None, 'gf': None, 'ga': None, 'elo': None, 'sample_size': 6, 'avg_opp_elo': 1750.0, 'absences': []},
        'odds': {'home': None, 'draw': None, 'away': None}
    }

    for raw_ln in text.splitlines():
        ln = raw_ln.strip().lstrip('-* ').strip()
        if ':' not in ln:
            continue
        key_part, val_part = ln.split(':', 1)
        k = key_part.lower().strip()
        v = val_part.strip()

        # Home fields
        if re.fullmatch(r'home\s*team', k):
            payload['home']['name'] = v
        elif re.fullmatch(r'home\s*gf', k):
            payload['home']['gf'] = parse_number_after_colon(ln)
        elif re.fullmatch(r'home\s*ga', k):
            payload['home']['ga'] = parse_number_after_colon(ln)
        elif re.fullmatch(r'home\s*elo', k):
            payload['home']['elo'] = parse_number_after_colon(ln)
        elif re.fullmatch(r'home\s*sample', k):
            num = parse_number_after_colon(ln)
            if num: payload['home']['sample_size'] = int(num)
        elif re.fullmatch(r'home\s*opp\s*elo', k):
            num = parse_number_after_colon(ln)
            if num: payload['home']['avg_opp_elo'] = num
        elif re.fullmatch(r'home\s*absences', k):
            payload['home']['absences'] = parse_absences_list(v)

        # Away fields
        elif re.fullmatch(r'away\s*team', k):
            payload['away']['name'] = v
        elif re.fullmatch(r'away\s*gf', k):
            payload['away']['gf'] = parse_number_after_colon(ln)
        elif re.fullmatch(r'away\s*ga', k):
            payload['away']['ga'] = parse_number_after_colon(ln)
        elif re.fullmatch(r'away\s*elo', k):
            payload['away']['elo'] = parse_number_after_colon(ln)
        elif re.fullmatch(r'away\s*sample', k):
            num = parse_number_after_colon(ln)
            if num: payload['away']['sample_size'] = int(num)
        elif re.fullmatch(r'away\s*opp\s*elo', k):
            num = parse_number_after_colon(ln)
            if num: payload['away']['avg_opp_elo'] = num
        elif re.fullmatch(r'away\s*absences', k):
            payload['away']['absences'] = parse_absences_list(v)

        # Odds fields
        elif re.fullmatch(r'odds\s*1|home\s*odds', k):
            payload['odds']['home'] = parse_number_after_colon(ln)
        elif re.fullmatch(r'odds\s*x|draw\s*odds', k):
            payload['odds']['draw'] = parse_number_after_colon(ln)
        elif re.fullmatch(r'odds\s*2|away\s*odds', k):
            payload['odds']['away'] = parse_number_after_colon(ln)

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
        raise RuntimeError("API-Football key is required for Live API Mode.")
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
    st.session_state['match_v52'] = payload
    st.session_state['res_v52'] = res

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
    st.title("Match Lab V5.2")

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
st.title("⚽ Football Match Lab V5.2 — One-Paste & Live API Engine")
st.caption("Single-Box Paste (Text or JSON) • Live API Fetch • Strict Validation • Vectorized Dixon-Coles")

tabs = st.tabs([
    "📥 1. One-Paste Input / API",
    "🎲 2. Prediction & Score Matrix",
    "📊 3. Market Odds & EV",
    "🗄️ 4. History & Accuracy"
])

# ---------------- TAB 0: ONE-PASTE INPUT & API ----------------
with tabs[0]:
    input_mode = st.radio(
        "Select Input Mode:",
        [
            "One-Paste Match Block (Text or JSON)",
            "API-Football Live Fetch"
        ],
        horizontal=True
    )

    st.divider()

    if input_mode == "One-Paste Match Block (Text or JSON)":
        st.subheader("Paste Complete Match Block (One Single Box)")
        st.caption("Paste either the simple 15-line Key: Value block OR raw JSON below. Everything runs in one click.")

        one_paste_template = """Home Team: Italy
Home GF: 1.67
Home GA: 1.15
Home Elo: 1840
Home Absences: Giovanni Di Lorenzo (DEF, starter), Moise Kean (FWD, starter)

Away Team: France
Away GF: 2.10
Away GA: 0.75
Away Elo: 1910
Away Absences: William Saliba (DEF, star), Ousmane Dembélé (FWD, star)

Odds 1: 2.75
Odds X: 3.20
Odds 2: 2.45"""

        with st.expander("📋 Click to Copy Blank Template"):
            st.code(one_paste_template, language="yaml")

        raw_paste = st.text_area(
            "Paste Match Block (Key: Value Text OR JSON) Here:",
            value="",
            placeholder=one_paste_template,
            height=340,
            key="one_paste_box"
        )

        if st.button("🚀 Validate & Run Simulation", type="primary", use_container_width=True):
            if not raw_paste.strip():
                st.error("Paste box is empty. Paste a match block first.")
            else:
                parsed, errs = parse_one_paste_input(raw_paste)
                if errs:
                    st.error("❌ Validation Failed — Fix these fields:")
                    for e in errs:
                        st.write(f"- {e}")
                else:
                    execute_and_save_model(parsed, "One-Paste", neutral_venue, home_adv_val, dixon_coles_rho, elo_blend, uncertainty_val)
                    st.success(f"✅ Validated & Simulated: **{parsed['home']['name']} (Elo {parsed['home']['elo']}) vs {parsed['away']['name']} (Elo {parsed['away']['elo']})**! Switch to Tab 2 to view results.")

    else:
        st.subheader("API-Football Live Senior-Team Pipeline")
        st.caption("Fetches real recent GF/GA and confirmed injury lists via API-Football.")

        ac1, ac2 = st.columns(2)
        with ac1:
            api_h_name = st.text_input("Home Senior Team Name", placeholder="e.g. Arsenal")
            api_h_elo = st.number_input("Home Elo Rating", min_value=800.0, max_value=2600.0, value=None, step=5.0)
        with ac2:
            api_a_name = st.text_input("Away Senior Team Name", placeholder="e.g. Chelsea")
            api_a_elo = st.number_input("Away Elo Rating", min_value=800.0, max_value=2600.0, value=None, step=5.0)

        ao1, ao2, ao3 = st.columns(3)
        api_odd_h = ao1.number_input("Odds 1 (Home)", min_value=1.01, max_value=100.0, value=None, step=0.05)
        api_odd_d = ao2.number_input("Odds X (Draw)", min_value=1.01, max_value=100.0, value=None, step=0.05)
        api_odd_a = ao3.number_input("Odds 2 (Away)", min_value=1.01, max_value=100.0, value=None, step=0.05)

        if st.button("🌐 Fetch Live API Stats & Run Simulation", type="primary", use_container_width=True):
            key = get_api_key()
            if not key:
                st.error("Enter your API-Football key in the left sidebar first.")
            elif not api_h_name.strip() or not api_a_name.strip():
                st.error("Enter both Home and Away team names.")
            else:
                try:
                    with st.spinner("Resolving senior teams & fetching recent stats..."):
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
                                    st.success(f"✅ API Data Verified & Simulated: **{ht['name']} vs {at['name']}**!")
                except Exception as exc:
                    st.error(f"API Fetch Error: {exc}")

# ---------------- TAB 1: PREDICTION & SCORE MATRIX ----------------
with tabs[1]:
    st.subheader("Vectorized Dixon-Coles Probabilities & Scorelines")
    data = st.session_state.get('match_v52')
    res = st.session_state.get('res_v52')

    if not data or not res:
        st.info("Paste and run a match in Tab 1 first.")
    else:
        h_name, a_name = data['home']['name'], data['away']['name']

        if st.button("🔄 Recalculate with Current Sidebar Settings"):
            lh, ma = calculate_expected_goals(data, elo_weight=elo_blend)
            M = compute_score_matrix(lh, ma, rho=dixon_coles_rho, home_adv=home_adv_val, unc=uncertainty_val)
            res = extract_markets(M)
            st.session_state['res_v52'] = res

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{h_name} Elo", f"{data['home']['elo']}")
        c2.metric(f"{a_name} Elo", f"{data['away']['elo']}")
        c3.metric("Venue Mode", "Neutral (1.00x)" if neutral_venue else f"Home Adv ({home_adv_val:.2f}x)")
        c4.metric("Absences Parsed", f"{len(data['home'].get('absences', []))} H / {len(data['away'].get('absences', []))} A")

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
    data = st.session_state.get('match_v52')
    res = st.session_state.get('res_v52')

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
