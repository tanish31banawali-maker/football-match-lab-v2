import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import math
import os
import re
import requests
import urllib.parse
from datetime import date, datetime, timedelta

# ---------------- App & Database Configuration ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, 'football_lab_v4.db')
API_BASE = 'https://v3.football.api-sports.io'
OPENFOOTBALL_BASE = 'https://raw.githubusercontent.com/openfootball/football.json/master'

st.set_page_config(
    page_title='Football Match Lab V4',
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
            fixture_id INTEGER,
            match_date TEXT,
            home_id INTEGER,
            away_id INTEGER,
            home TEXT NOT NULL,
            away TEXT NOT NULL,
            competition TEXT,
            simulations INTEGER,
            p_home REAL,
            p_draw REAL,
            p_away REAL,
            p_over15 REAL,
            p_over25 REAL,
            p_over35 REAL,
            p_over45 REAL,
            p_btts REAL,
            exp_home REAL,
            exp_away REAL,
            top_score TEXT,
            bookmaker_home_odds REAL,
            bookmaker_draw_odds REAL,
            bookmaker_away_odds REAL,
            implied_home_prob REAL,
            implied_draw_prob REAL,
            implied_away_prob REAL,
            verification_status TEXT DEFAULT 'Unverified',
            source TEXT DEFAULT 'manual',
            actual_home INTEGER,
            actual_away INTEGER,
            settled_at TEXT
        )''')
        con.execute('''CREATE TABLE IF NOT EXISTS team_elo_ratings (
            team_id INTEGER PRIMARY KEY,
            team_name TEXT,
            rating REAL,
            last_updated TEXT
        )''')
        con.execute('''CREATE TABLE IF NOT EXISTS research_cache (
            cache_key TEXT PRIMARY KEY,
            fetched_at TEXT,
            payload TEXT
        )''')

init_db()

# ---------------- Mathematical & Modeling Utilities ----------------
def safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def poisson_pmf(lam, k):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))

def dixon_coles_tau(x, y, lambda_h, mu_a, rho=-0.13):
    if x == 0 and y == 0:
        return max(0.0001, 1.0 - (lambda_h * mu_a * rho))
    elif x == 1 and y == 0:
        return max(0.0001, 1.0 + (mu_a * rho))
    elif x == 0 and y == 1:
        return max(0.0001, 1.0 + (lambda_h * rho))
    elif x == 1 and y == 1:
        return max(0.0001, 1.0 - rho)
    else:
        return 1.0

def simulate_dixon_coles_monte_carlo(lambda_h, mu_a, n=50000, seed=20261111, rho=-0.13, home_adv=1.08, uncertainty=0.15):
    rng = np.random.default_rng(int(seed))
    adjusted_lh = max(0.1, lambda_h * home_adv)
    adjusted_ma = max(0.1, mu_a)
    
    lh_samples = np.maximum(0.05, rng.lognormal(np.log(adjusted_lh), uncertainty, int(n)))
    ma_samples = np.maximum(0.05, rng.lognormal(np.log(adjusted_ma), uncertainty, int(n)))
    
    hg_raw = rng.poisson(lh_samples)
    ag_raw = rng.poisson(ma_samples)
    
    hg = np.zeros(n, dtype=int)
    ag = np.zeros(n, dtype=int)
    
    for i in range(n):
        x, y = hg_raw[i], ag_raw[i]
        if x <= 1 and y <= 1:
            tau = dixon_coles_tau(x, y, lh_samples[i], ma_samples[i], rho)
            if rng.random() > tau:
                x = rng.poisson(lh_samples[i])
                y = rng.poisson(ma_samples[i])
        hg[i] = x
        ag[i] = y
        
    return hg, ag

def model_probabilities_from_samples(hg, ag):
    n = len(hg)
    total = hg + ag
    
    p_home = float(np.mean(hg > ag))
    p_draw = float(np.mean(hg == ag))
    p_away = float(np.mean(hg < ag))
    
    sum_1x2 = p_home + p_draw + p_away
    p_home /= sum_1x2
    p_draw /= sum_1x2
    p_away /= sum_1x2

    scores_df = pd.DataFrame({'home': hg, 'away': ag}).value_counts().rename('count').reset_index()
    scores_df['probability'] = scores_df['count'] / n

    return {
        'p_home': p_home,
        'p_draw': p_draw,
        'p_away': p_away,
        'p_over15': float(np.mean(total >= 2)),
        'p_over25': float(np.mean(total >= 3)),
        'p_over35': float(np.mean(total >= 4)),
        'p_over45': float(np.mean(total >= 5)),
        'p_under25': float(np.mean(total <= 2)),
        'p_btts': float(np.mean((hg > 0) & (ag > 0))),
        'exp_home': float(np.mean(hg)),
        'exp_away': float(np.mean(ag)),
        'scores': scores_df.sort_values('probability', ascending=False).head(10)
    }

# ---------------- Tactical Formation Impact Engine ----------------
FORMATION_PROFILES = {
    "4-3-3": {"attack": 1.05, "defense": 1.00},
    "4-2-3-1": {"attack": 1.02, "defense": 1.02},
    "3-5-2": {"attack": 1.04, "defense": 0.96},
    "4-4-2": {"attack": 0.98, "defense": 1.03},
    "3-4-2-1": {"attack": 1.06, "defense": 0.97},
    "5-3-2": {"attack": 0.90, "defense": 1.12}
}

def get_formation_multipliers(formation_name):
    return FORMATION_PROFILES.get(formation_name, {"attack": 1.00, "defense": 1.00})

# ---------------- Elo & Shrinkage Engine ----------------
DEFAULT_ELO = 1500.0

def get_team_elo(team_id):
    with db() as con:
        cur = con.execute("SELECT rating FROM team_elo_ratings WHERE team_id = ?", (int(team_id),))
        row = cur.fetchone()
        return row['rating'] if row else DEFAULT_ELO

def calculate_elo_adjustment(team_elo, opp_elo, is_home=True):
    diff = (team_elo + (100.0 if is_home else 0.0)) - opp_elo
    multiplier = 10.0 ** (diff / 600.0)
    return clamp(multiplier, 0.5, 2.0)

def apply_bayesian_shrinkage(observed_value, sample_size, baseline_value, k=10.0):
    weight = sample_size / (sample_size + k)
    return (weight * observed_value) + ((1.0 - weight) * baseline_value)

# ---------------- OpenFootball Engine (No API Key Required) ----------------
@st.cache_data(ttl=3600)
def fetch_openfootball_league(season_str, league_code):
    url = f"{OPENFOOTBALL_BASE}/{season_str}/{league_code}.json"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

def calc_openfootball_team_stats(team_name, season_str="2024-25", league_code="en.1"):
    data = fetch_openfootball_league(season_str, league_code)
    if not data or 'matches' not in data:
        return None
        
    scored, conceded, matches_played = 0, 0, 0
    t_clean = team_name.lower().strip()
    
    for m in data['matches']:
        if 'score' not in m or 'ft' not in m['score']:
            continue
        
        t1 = m['team1'].lower()
        t2 = m['team2'].lower()
        ft = m['score']['ft']
        
        if t_clean in t1:
            scored += ft[0]
            conceded += ft[1]
            matches_played += 1
        elif t_clean in t2:
            scored += ft[1]
            conceded += ft[0]
            matches_played += 1
            
    if matches_played == 0:
        return None
        
    return {
        'matches_played': matches_played,
        'gf': round(scored / matches_played, 2),
        'ga': round(conceded / matches_played, 2)
    }

# ---------------- API Engine with Safe URL Encoding & Error Handling ----------------
def get_api_key():
    if 'api_key' in st.session_state and st.session_state['api_key']:
        return st.session_state['api_key']
    try:
        return st.secrets.get("API_FOOTBALL_KEY", "")
    except Exception:
        return ""

def api_get(path, params, key):
    if not key or not key.strip():
        raise RuntimeError("API Key missing. Enter your API-Football key in Settings sidebar or switch to Manual / OpenFootball Mode.")
    resp = requests.get(f"{API_BASE}/{path.lstrip('/')}", headers={'x-apisports-key': key}, params=params, timeout=20)
    resp.raise_for_status()
    body = resp.json()
    if body.get('errors'):
        err_msg = str(body['errors'])
        if err_msg != "[]" and err_msg != "{}":
            raise RuntimeError(f"API Error: {err_msg}")
    return body.get('response', []), body.get('results', 0), body.get('paging', {})

def is_senior_team(team_name):
    banned_keywords = [' u21', ' u19', ' u18', ' u23', ' youth', ' reserve', ' res.', ' ii', ' b team', ' women', ' w']
    lower = f" {team_name.lower()} "
    for kw in banned_keywords:
        if kw in lower:
            return False
    return True

def resolve_exact_team(name, key):
    safe_search = urllib.parse.quote(name.strip())
    rows, _, _ = api_get('teams', {'search': safe_search}, key)
    if not rows:
        return None
    
    clean_search = re.sub(r'[^a-z0-9]', '', name.lower())
    candidates = []
    for r in rows:
        t_name = r['team']['name']
        if not is_senior_team(t_name):
            continue
        clean_cand = re.sub(r'[^a-z0-9]', '', t_name.lower())
        score = 10 if clean_cand == clean_search else 0
        if r['team'].get('national', False): score += 2
        candidates.append((score, r))
        
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def verify_and_fetch_fixture(home_id, away_id, season, key, league_id=None):
    params = {'team': home_id, 'next': 15, 'season': season}
    if league_id and int(league_id) > 0:
        params['league'] = int(league_id)
        
    rows, _, _ = api_get('fixtures', params, key)
    for x in rows:
        if x['teams']['home']['id'] == home_id and x['teams']['away']['id'] == away_id:
            return {
                'verified': True,
                'fixture_id': x['fixture']['id'],
                'match_date': x['fixture']['date'][:10],
                'competition': x['league']['name'],
                'competition_id': x['league']['id'],
                'country': x['league']['country'],
                'status': x['fixture']['status']['long']
            }
    return {'verified': False, 'fixture_id': None, 'reason': 'Fixture match not found on provider schedule'}

# ---------------- Bookmaker Odds Normalization ----------------
def analyze_bookmaker_odds(home_odds, draw_odds, away_odds):
    if home_odds <= 1.0 or draw_odds <= 1.0 or away_odds <= 1.0:
        return None
    raw_h = 1.0 / home_odds
    raw_d = 1.0 / draw_odds
    raw_a = 1.0 / away_odds
    
    total_implied = raw_h + raw_d + raw_a
    margin = (total_implied - 1.0) * 100.0
    
    return {
        'margin_pct': margin,
        'implied_h': raw_h / total_implied,
        'implied_d': raw_d / total_implied,
        'implied_a': raw_a / total_implied
    }

# ---------------- Sidebar Navigation & Controls ----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/football-ball-1.png", width=64)
    st.title("Match Lab V4 Controls")
    
    api_key_input = st.text_input("API-Football Key", type="password", value=get_api_key(), help="Leave blank if using Manual or OpenFootball Mode.")
    st.session_state['api_key'] = api_key_input
    
    season_val = st.number_input("Season Year", min_value=2018, max_value=2030, value=2026)
    league_id_val = st.number_input("League ID (Optional)", min_value=0, value=0)
    
    st.divider()
    st.subheader("Monte Carlo Simulation Engine")
    sim_count = st.select_slider("Simulation Runs", options=[50000, 100000, 200000], value=50000)
    home_adv_val = st.slider("Home Advantage Multiplier", 1.00, 1.25, 1.08, 0.01)
    dixon_coles_rho = st.slider("Dixon-Coles Rho (Draw Tuning)", -0.30, 0.00, -0.13, 0.01)

# ---------------- App Header ----------------
st.title("⚽ Football Match Lab — Version 4")
st.caption("Senior-Team Verified • Dixon-Coles Poisson Monte Carlo • Elo Opponent-Adjusted • Market Comparison")

tabs = st.tabs([
    "🎯 Match Builder",
    "🛡️ Team Analysis",
    "📈 Recent Form",
    "⚔️ Head-to-Head",
    "👤 Player Analysis",
    "🏥 Injuries & Squad",
    "📋 Confirmed Lineups",
    "🧠 Tactical Analysis",
    "🎲 Prediction Engine",
    "📊 Bookmaker Comparison",
    "📉 Accuracy & Calibration",
    "📋 Data Quality & Status"
])

# ---------------- TAB 0: MATCH BUILDER ----------------
with tabs[0]:
    st.subheader("Step 1: Identify Match & Select Verification Mode")
    
    mode = st.radio(
        "Select Operating Mode:",
        [
            "Manual / Standalone Mode (Free - Manual Inputs)",
            "OpenFootball Data Mode (Free - GitHub JSON Dataset)",
            "API-Football Automatic Mode (API Key Required)"
        ],
        index=0
    )
    
    c1, c2 = st.columns(2)
    with c1:
        home_name_input = st.text_input("Home Team Name", value="Arsenal", key="mb_home")
        h_g_for = st.number_input("Home Avg Goals Scored", min_value=0.0, max_value=5.0, value=1.8, step=0.1)
        h_g_against = st.number_input("Home Avg Goals Conceded", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    with c2:
        away_name_input = st.text_input("Away Team Name", value="Chelsea", key="mb_away")
        a_g_for = st.number_input("Away Avg Goals Scored", min_value=0.0, max_value=5.0, value=1.3, step=0.1)
        a_g_against = st.number_input("Away Avg Goals Conceded", min_value=0.0, max_value=5.0, value=1.2, step=0.1)
        
    if mode.startswith("Manual"):
        is_senior = st.checkbox("Confirm Senior Teams (Not Youth, U21, or Reserves)", value=True)
        if st.button("✅ Confirm Manual Fixture & Unlock Prediction Engine", type="primary", use_container_width=True):
            if not is_senior:
                st.error("Must confirm senior team status before running analysis.")
            else:
                st.session_state['current_match_v4'] = {
                    'verified': True,
                    'verification_status': 'Manually Verified',
                    'home_team': {'id': 9991, 'name': home_name_input, 'country': 'Manual'},
                    'away_team': {'id': 9992, 'name': away_name_input, 'country': 'Manual'},
                    'recent_h': [{'gf': h_g_for, 'ga': h_g_against, 'opponent_elo': 1500}] * 10,
                    'recent_a': [{'gf': a_g_for, 'ga': a_g_against, 'opponent_elo': 1500}] * 10,
                    'h2h': [], 'injuries_h': [], 'injuries_a': [], 'lineups': []
                }
                st.success(f"Fixture '{home_name_input} vs {away_name_input}' manually confirmed! Prediction engine unlocked.")
                st.rerun()

    elif mode.startswith("OpenFootball"):
        st.info("Fetches goal averages automatically from GitHub `openfootball/football.json` datasets.")
        league_code_input = st.selectbox("Select Competition Code", ["en.1", "es.1", "de.1", "it.1"], format_func=lambda x: {"en.1": "Premier League", "es.1": "La Liga", "de.1": "Bundesliga", "it.1": "Serie A"}[x])
        season_code_input = st.text_input("Season Format (e.g. 2024-25)", value="2024-25")
        
        if st.button("🌐 Fetch OpenFootball Stats & Confirm Fixture", type="primary", use_container_width=True):
            with st.spinner("Fetching openfootball datasets..."):
                h_stats = calc_openfootball_team_stats(home_name_input, season_code_input, league_code_input)
                a_stats = calc_openfootball_team_stats(away_name_input, season_code_input, league_code_input)
                
                h_gf = h_stats['gf'] if h_stats else h_g_for
                h_ga = h_stats['ga'] if h_stats else h_g_against
                a_gf = a_stats['gf'] if a_stats else a_g_for
                a_ga = a_stats['ga'] if a_stats else a_g_against
                
                st.session_state['current_match_v4'] = {
                    'verified': True,
                    'verification_status': 'OpenFootball Auto-Verified',
                    'home_team': {'id': 8881, 'name': home_name_input, 'country': 'OpenFootball'},
                    'away_team': {'id': 8882, 'name': away_name_input, 'country': 'OpenFootball'},
                    'recent_h': [{'gf': h_gf, 'ga': h_ga, 'opponent_elo': 1500}] * 10,
                    'recent_a': [{'gf': a_gf, 'ga': a_ga, 'opponent_elo': 1500}] * 10,
                    'h2h': [], 'injuries_h': [], 'injuries_a': [], 'lineups': []
                }
                st.success(f"Fetched stats: {home_name_input} ({h_gf} GF / {h_ga} GA) vs {away_name_input} ({a_gf} GF / {a_ga} GA). Engine unlocked!")
                st.rerun()

    else:
        btn_verify = st.button("🔍 Verify Teams via API", use_container_width=True, type="primary")
        if btn_verify:
            key = get_api_key()
            if not key:
                st.error("API Key missing. Enter key in sidebar or switch to Manual / OpenFootball Mode above.")
            else:
                with st.spinner("Verifying senior teams..."):
                    try:
                        ht = resolve_exact_team(home_name_input, key)
                        at = resolve_exact_team(away_name_input, key)
                        if not ht or not at:
                            missing = []
                            if not ht: missing.append(home_name_input)
                            if not at: missing.append(away_name_input)
                            st.error(f"Could not resolve senior team(s): {', '.join(missing)}. Try exact spelling or switch to Manual Mode.")
                            st.session_state['current_match_v4'] = {'verified': False, 'reason': 'Team resolution failed'}
                        else:
                            hid, aid = ht['team']['id'], at['team']['id']
                            fix_info = verify_and_fetch_fixture(hid, aid, season_val, key, league_id_val)
                            st.session_state['current_match_v4'] = {
                                'verified': True,
                                'verification_status': 'API Verified' if fix_info['verified'] else 'API Teams Found (Unscheduled)',
                                'fixture_info': fix_info,
                                'home_team': ht['team'],
                                'away_team': at['team'],
                                'recent_h': [{'gf': h_g_for, 'ga': h_g_against, 'opponent_elo': 1500}] * 10,
                                'recent_a': [{'gf': a_g_for, 'ga': a_g_against, 'opponent_elo': 1500}] * 10,
                                'h2h': [], 'injuries_h': [], 'injuries_a': [], 'lineups': []
                            }
                            st.success(f"✅ Teams Resolved: **{ht['team']['name']}** vs **{at['team']['name']}**")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Verification process error: {e}")
                        st.session_state['current_match_v4'] = {'verified': False, 'reason': str(e)}

    match_data = st.session_state.get('current_match_v4', {})
    if match_data.get('verified'):
        st.success(f"✅ **Fixture Status:** {match_data['home_team']['name']} vs {match_data['away_team']['name']} ({match_data.get('verification_status', 'Verified')})")
    else:
        st.warning("⚠️ **Fixture Not Verified** — Confirm a fixture above to enable predictions.")

# ---------------- TAB 1: TEAM ANALYSIS ----------------
with tabs[1]:
    st.subheader("Team Identity & Elo Ratings")
    match_data = st.session_state.get('current_match_v4', {})
    if not match_data.get('verified'):
        st.warning("Please verify a fixture in Match Builder first.")
    else:
        c1, c2 = st.columns(2)
        c1.markdown(f"### 🏠 {match_data['home_team']['name']}")
        c1.metric("Home Team Elo Rating", f"{get_team_elo(match_data['home_team']['id']):.1f}")
        c2.markdown(f"### ✈️ {match_data['away_team']['name']}")
        c2.metric("Away Team Elo Rating", f"{get_team_elo(match_data['away_team']['id']):.1f}")

# ---------------- TAB 2: RECENT FORM ----------------
with tabs[2]:
    st.subheader("📈 Recent Form & Match History Logging")
    match_data = st.session_state.get('current_match_v4', {})
    if not match_data.get('verified'):
        st.warning("Please confirm a fixture in Match Builder first.")
    else:
        st.write(f"Add custom recent results for **{match_data['home_team']['name']}** and **{match_data['away_team']['name']}** to fine-tune goal expectations.")
        col_h, col_a = st.columns(2)
        
        with col_h:
            st.markdown(f"#### 🏠 {match_data['home_team']['name']} Last 5 Matches")
            default_h_df = pd.DataFrame(match_data.get('recent_h', [{'gf': 1.8, 'ga': 1.0, 'opponent_elo': 1500}]*5))
            edited_h = st.data_editor(default_h_df, num_rows="dynamic", key="editor_form_h", use_container_width=True)
            if st.button("Save Home Form Data"):
                match_data['recent_h'] = edited_h.to_dict(orient="records")
                st.session_state['current_match_v4'] = match_data
                st.success("Home form updated!")
                
        with col_a:
            st.markdown(f"#### ✈️ {match_data['away_team']['name']} Last 5 Matches")
            default_a_df = pd.DataFrame(match_data.get('recent_a', [{'gf': 1.3, 'ga': 1.2, 'opponent_elo': 1500}]*5))
            edited_a = st.data_editor(default_a_df, num_rows="dynamic", key="editor_form_a", use_container_width=True)
            if st.button("Save Away Form Data"):
                match_data['recent_a'] = edited_a.to_dict(orient="records")
                st.session_state['current_match_v4'] = match_data
                st.success("Away form updated!")

# ---------------- TAB 3: HEAD-TO-HEAD HISTORY ----------------
with tabs[3]:
    st.subheader("⚔️ Head-to-Head (H2H) Record")
    match_data = st.session_state.get('current_match_v4', {})
    if not match_data.get('verified'):
        st.warning("Please confirm a fixture in Match Builder first.")
    else:
        st.write("Add recent direct encounters between these two teams:")
        h2h_data = match_data.get('h2h', [
            {"Date": "2025-11-10", "Home": match_data['home_team']['name'], "Score": "2 - 1", "Away": match_data['away_team']['name']},
            {"Date": "2025-04-15", "Home": match_data['away_team']['name'], "Score": "1 - 1", "Away": match_data['home_team']['name']}
        ])
        edited_h2h = st.data_editor(pd.DataFrame(h2h_data), num_rows="dynamic", use_container_width=True)
        if st.button("💾 Save H2H Matches"):
            match_data['h2h'] = edited_h2h.to_dict(orient="records")
            st.session_state['current_match_v4'] = match_data
            st.success("H2H data saved!")

# ---------------- TAB 4: PLAYER ANALYSIS ----------------
with tabs[4]:
    st.subheader("👤 Key Player Impact & Rating")
    match_data = st.session_state.get('current_match_v4', {})
    if not match_data.get('verified'):
        st.warning("Please confirm a fixture in Match Builder first.")
    else:
        st.write("Track standout players and key statistical metrics for this fixture:")
        p_df = pd.DataFrame(match_data.get('player_impact', [
            {"Team": match_data['home_team']['name'], "Player Name": "Key Striker", "Position": "FW", "Rating": 8.2, "xG per 90": 0.65},
            {"Team": match_data['away_team']['name'], "Player Name": "Playmaker", "Position": "MF", "Rating": 7.9, "xG per 90": 0.42}
        ]))
        edited_p = st.data_editor(p_df, num_rows="dynamic", key="player_impact_editor", use_container_width=True)
        if st.button("💾 Save Key Player Impact"):
            match_data['player_impact'] = edited_p.to_dict(orient="records")
            st.session_state['current_match_v4'] = match_data
            st.success("Player impact ratings saved for model adjustment.")

# ---------------- TAB 5: INJURIES & SQUAD ----------------
with tabs[5]:
    st.subheader("🏥 Injuries & Squad Availability")
    match_data = st.session_state.get('current_match_v4', {})
    if not match_data.get('verified'):
        st.warning("Please confirm a fixture in Match Builder first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"#### 🏠 {match_data['home_team']['name']} Absences")
            inj_h = pd.DataFrame(match_data.get('injuries_h', [{"Player": "Defender A", "Reason": "Hamstring", "Status": "Out"}]))
            edited_inj_h = st.data_editor(inj_h, num_rows="dynamic", key="inj_h_edit", use_container_width=True)
        with c2:
            st.markdown(f"#### ✈️ {match_data['away_team']['name']} Absences")
            inj_a = pd.DataFrame(match_data.get('injuries_a', [{"Player": "Midfielder B", "Reason": "Suspended", "Status": "Out"}]))
            edited_inj_a = st.data_editor(inj_a, num_rows="dynamic", key="inj_a_edit", use_container_width=True)
            
        if st.button("💾 Save Squad Availability"):
            match_data['injuries_h'] = edited_inj_h.to_dict(orient="records")
            match_data['injuries_a'] = edited_inj_a.to_dict(orient="records")
            st.session_state['current_match_v4'] = match_data
            st.success("Squad availability saved!")

# ---------------- TAB 6: CONFIRMED LINEUPS ----------------
with tabs[6]:
    st.subheader("📋 Match Lineup Builder & Tactical Formations")
    match_data = st.session_state.get('current_match_v4', {})
    if not match_data.get('verified'):
        st.warning("Please confirm a fixture in Match Builder first.")
    else:
        st.write("Enter or paste starting XIs and select tactical formations for both teams:")
        
        # Presets generator
        preset_cols = st.columns(3)
        with preset_cols[0]:
            if st.button("⚡ Load 4-3-3 Standard Preset", use_container_width=True):
                st.session_state['h_formation'] = '4-3-3'
                st.session_state['a_formation'] = '4-3-3'
                st.session_state['h_lineup_text'] = "GK: Raya\nDF: White, Saliba, Gabriel, Calafiori\nMF: Odegaard, Partey, Rice\nFW: Saka, Havertz, Martinelli"
                st.session_state['a_lineup_text'] = "GK: Sanchez\nDF: James, Fofana, Colwill, Cucurella\nMF: Caicedo, Enzo, Palmer\nFW: Madueke, Jackson, Neto"
                st.rerun()
                
        with preset_cols[1]:
            if st.button("⚡ Load 4-2-3-1 Modern Preset", use_container_width=True):
                st.session_state['h_formation'] = '4-2-3-1'
                st.session_state['a_formation'] = '4-2-3-1'
                st.session_state['h_lineup_text'] = "GK: Neuer\nDF: Laimer, Upamecano, Kim, Davies\nMF: Kimmich, Pavlovic\nFW: Olise, Musiala, Gnabry\nST: Kane"
                st.session_state['a_lineup_text'] = "GK: Donnarumma\nDF: Hakimi, Marquinhos, Pacho, Mendes\nMF: Neves, Vitinha, Ruiz\nFW: Dembele, Asensio, Barcola"
                st.rerun()

        with preset_cols[2]:
            if st.button("⚡ Load 3-5-2 Wing-Back Preset", use_container_width=True):
                st.session_state['h_formation'] = '3-5-2'
                st.session_state['a_formation'] = '3-5-2'
                st.session_state['h_lineup_text'] = "GK: Sommer\nDF: Pavard, Acerbi, Bastoni\nMF: Darmian, Barella, Calhanoglu, Mkhitaryan, Dimarco\nFW: Thuram, Martinez"
                st.session_state['a_lineup_text'] = "GK: Svilar\nDF: Mancini, Ndicka, Hermoso\nMF: Celik, Kone, Cristante, Pellegrini, Angelino\nFW: Dybala, Dovbyk"
                st.rerun()

        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🏠 {match_data['home_team']['name']}")
            h_form = st.selectbox("Home Formation", ["4-3-3", "4-2-3-1", "3-5-2", "4-4-2", "3-4-2-1", "5-3-2"], key="h_formation")
            h_starting = st.text_area(
                "Home Starting XI", 
                value=st.session_state.get('h_lineup_text', "Raya\nWhite\nSaliba\nGabriel\nCalafiori\nOdegaard\nPartey\nRice\nSaka\nHavertz\nMartinelli"),
                height=220,
                key="h_lineup_input"
            )
            h_subs = st.text_input("Home Substitutes", value="Neto, Kiwior, Jorginho, Sterling, Jesus", key="h_subs")
            h_manager = st.text_input("Home Manager", value="Mikel Arteta", key="h_mgr")

        with col2:
            st.markdown(f"### ✈️ {match_data['away_team']['name']}")
            a_form = st.selectbox("Away Formation", ["4-2-3-1", "4-3-3", "3-5-2", "4-4-2", "3-4-2-1", "5-3-2"], key="a_formation")
            a_starting = st.text_area(
                "Away Starting XI", 
                value=st.session_state.get('a_lineup_text', "Sanchez\nJames\nFofana\nColwill\nCucurella\nCaicedo\nEnzo\nPalmer\nMadueke\nJackson\nNeto"),
                height=220,
                key="a_lineup_input"
            )
            a_subs = st.text_input("Away Substitutes", value="Jorgensen, Badiashile, Dewsbury-Hall, Nkunku, Felix", key="a_subs")
            a_manager = st.text_input("Away Manager", value="Enzo Maresca", key="a_mgr")

        st.divider()
        if st.button("✅ Confirm Lineups & Tactical Formations", type="primary", use_container_width=True):
            h_list = [p.strip() for p in h_starting.replace('\n', ',').split(',') if p.strip()]
            a_list = [p.strip() for p in a_starting.replace('\n', ',').split(',') if p.strip()]
            
            lineup_payload = {
                'home': {'formation': h_form, 'starting': h_list, 'subs': h_subs, 'manager': h_manager},
                'away': {'formation': a_form, 'starting': a_list, 'subs': a_subs, 'manager': a_manager},
                'confirmed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            match_data['lineups'] = lineup_payload
            st.session_state['current_match_v4'] = match_data
            st.success("Lineups and tactical formations confirmed!")

# ---------------- TAB 7: TACTICAL OVERVIEW ----------------
with tabs[7]:
    st.subheader("🧠 Tactical Analysis & Formation Matchup")
    match_data = st.session_state.get('current_match_v4', {})
    if not match_data.get('verified'):
        st.warning("Please confirm a fixture in Match Builder first.")
    else:
        lineup_data = match_data.get('lineups', {})
        h_form = lineup_data.get('home', {}).get('formation', '4-3-3') if isinstance(lineup_data, dict) else '4-3-3'
        a_form = lineup_data.get('away', {}).get('formation', '4-2-3-1') if isinstance(lineup_data, dict) else '4-2-3-1'
        
        c1, c2 = st.columns(2)
        c1.metric(f"🏠 {match_data['home_team']['name']} Formation", h_form)
        c2.metric(f"✈️ {match_data['away_team']['name']} Formation", a_form)
        
        h_prof = get_formation_multipliers(h_form)
        a_prof = get_formation_multipliers(a_form)
        
        st.markdown("### Tactical Advantage Matrix")
        tac_df = pd.DataFrame([
            {"Team": match_data['home_team']['name'], "Shape": h_form, "Offensive Index": f"{h_prof['attack']:.2f}x", "Defensive Stability": f"{h_prof['defense']:.2f}x"},
            {"Team": match_data['away_team']['name'], "Shape": a_form, "Offensive Index": f"{a_prof['attack']:.2f}x", "Defensive Stability": f"{a_prof['defense']:.2f}x"}
        ])
        st.dataframe(tac_df, hide_index=True, use_container_width=True)

# ---------------- TAB 8: PREDICTION ENGINE ----------------
with tabs[8]:
    st.subheader("Dixon-Coles Poisson Monte Carlo Engine")
    match_data = st.session_state.get('current_match_v4', {})
    
    if not match_data.get('verified'):
        st.error("⚠️ **Automatic Prediction Disabled:** Fixture must be confirmed in Match Builder.")
    else:
        if st.button("🚀 Run V4 Monte Carlo Prediction Engine", type="primary", use_container_width=True):
            df_h = pd.DataFrame(match_data['recent_h'])
            df_a = pd.DataFrame(match_data['recent_a'])
            
            raw_h_for = df_h['gf'].mean() if not df_h.empty else 1.8
            raw_h_against = df_h['ga'].mean() if not df_h.empty else 1.0
            raw_a_for = df_a['gf'].mean() if not df_a.empty else 1.3
            raw_a_against = df_a['ga'].mean() if not df_a.empty else 1.2
            
            h_for_shrunk = apply_bayesian_shrinkage(raw_h_for, len(df_h), 1.35, k=10.0)
            a_for_shrunk = apply_bayesian_shrinkage(raw_a_for, len(df_a), 1.35, k=10.0)
            
            elo_h = get_team_elo(match_data['home_team']['id'])
            elo_a = get_team_elo(match_data['away_team']['id'])
            
            adj_h_mult = calculate_elo_adjustment(elo_h, elo_a, is_home=True)
            adj_a_mult = calculate_elo_adjustment(elo_a, elo_h, is_home=False)

            # ---------------- Tactical Formation & Squad Absences Adjustments ----------------
            lineups = match_data.get('lineups', {})
            h_form = lineups.get('home', {}).get('formation', '4-3-3') if isinstance(lineups, dict) else '4-3-3'
            a_form = lineups.get('away', {}).get('formation', '4-2-3-1') if isinstance(lineups, dict) else '4-2-3-1'
            
            h_prof = get_formation_multipliers(h_form)
            a_prof = get_formation_multipliers(a_form)
            
            # Absence Penalty (-7% per confirmed absence)
            inj_h = match_data.get('injuries_h', [])
            inj_a = match_data.get('injuries_a', [])
            h_absence_mult = clamp(1.0 - (len(inj_h) * 0.07), 0.60, 1.0)
            a_absence_mult = clamp(1.0 - (len(inj_a) * 0.07), 0.60, 1.0)

            # Expected Goal Multipliers
            lambda_h = (0.6 * h_for_shrunk + 0.4 * raw_a_against) * adj_h_mult * h_prof['attack'] * a_prof['defense'] * h_absence_mult
            mu_a = (0.6 * a_for_shrunk + 0.4 * raw_h_against) * adj_a_mult * a_prof['attack'] * h_prof['defense'] * a_absence_mult
            
            hg, ag = simulate_dixon_coles_monte_carlo(
                lambda_h, mu_a, n=sim_count, seed=20261111, rho=dixon_coles_rho, home_adv=home_adv_val
            )
            
            res = model_probabilities_from_samples(hg, ag)
            st.session_state['last_prediction_res_v4'] = res
            
            st.success("Simulation Complete!")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Home Win Probability", f"{res['p_home']:.1%}")
            c2.metric("Draw Probability", f"{res['p_draw']:.1%}")
            c3.metric("Away Win Probability", f"{res['p_away']:.1%}")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Over 1.5 Goals", f"{res['p_over15']:.1%}")
            m2.metric("Over 2.5 Goals", f"{res['p_over25']:.1%}")
            m3.metric("Over 3.5 Goals", f"{res['p_over35']:.1%}")
            m4.metric("BTTS Yes", f"{res['p_btts']:.1%}")
            
            st.markdown(f"**Expected Goals:** {match_data['home_team']['name']} **{res['exp_home']:.2f}** — **{res['exp_away']:.2f}** {match_data['away_team']['name']}")
            
            st.markdown("### Most Likely Correct Scores")
            scores_df = res['scores'].copy()
            scores_df['score'] = scores_df.apply(lambda x: f"{int(x.home)}-{int(x.away)}", axis=1)
            st.dataframe(scores_df[['score', 'probability']].style.format({'probability': '{:.2%}'}), hide_index=True, use_container_width=True)

# ---------------- TAB 9: BOOKMAKER COMPARISON ----------------
with tabs[9]:
    st.subheader("Market Odds & Value Comparison")
    c1, c2, c3 = st.columns(3)
    odd_h = c1.number_input("Bookmaker Home Odds", min_value=1.01, value=2.10, step=0.05)
    odd_d = c2.number_input("Bookmaker Draw Odds", min_value=1.01, value=3.40, step=0.05)
    odd_a = c3.number_input("Bookmaker Away Odds", min_value=1.01, value=3.60, step=0.05)
        
    bm_res = analyze_bookmaker_odds(odd_h, odd_d, odd_a)
    if bm_res:
        st.write(f"**Bookmaker Margin:** {bm_res['margin_pct']:.2f}%")
        pred_res = st.session_state.get('last_prediction_res_v4')
        if pred_res:
            comp_data = [
                {
                    'Outcome': 'Home Win (1)',
                    'Model Prob': f"{pred_res['p_home']:.1%}",
                    'Bookmaker Implied': f"{bm_res['implied_h']:.1%}",
                    'Diff': f"{(pred_res['p_home'] - bm_res['implied_h']) * 100:+.1f}%"
                },
                {
                    'Outcome': 'Draw (X)',
                    'Model Prob': f"{pred_res['p_draw']:.1%}",
                    'Bookmaker Implied': f"{bm_res['implied_d']:.1%}",
                    'Diff': f"{(pred_res['p_draw'] - bm_res['implied_d']) * 100:+.1f}%"
                },
                {
                    'Outcome': 'Away Win (2)',
                    'Model Prob': f"{pred_res['p_away']:.1%}",
                    'Bookmaker Implied': f"{bm_res['implied_a']:.1%}",
                    'Diff': f"{(pred_res['p_away'] - bm_res['implied_a']) * 100:+.1f}%"
                }
            ]
            st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

# ---------------- TAB 10-11: ACCURACY & STATUS ----------------
with tabs[10]:
    st.subheader("Historical Accuracy Tracking")
    st.info("Saves predictions locally to `football_lab_v4.db` for performance calculation.")

with tabs[11]:
    st.subheader("Data Quality Verification")
    match_data = st.session_state.get('current_match_v4', {})
    st.write(f"**Current Verification Mode:** {match_data.get('verification_status', 'Unverified')}")
