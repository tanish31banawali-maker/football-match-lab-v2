import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import math
import os
import re
import requests
from datetime import date, datetime, timedelta

# ---------------- App & Database Configuration ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, 'football_lab_v4.db')
API_BASE = 'https://v3.football.api-sports.io'
OPENFOOTBALL_BASE = 'https://raw.githubusercontent.com/openfootball/football.json/master'

st.set_page_config(
    page_title='Football Match Lab V4.1',
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

def generate_dixon_coles_matrix(lambda_h, mu_a, rho=-0.13, max_goals=10):
    """Builds a normalized bivariate score matrix applying Dixon-Coles tau adjustment."""
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for x in range(max_goals + 1):
        p_x = poisson_pmf(lambda_h, x)
        for y in range(max_goals + 1):
            p_y = poisson_pmf(mu_a, y)
            tau = dixon_coles_tau(x, y, lambda_h, mu_a, rho) if (x <= 1 and y <= 1) else 1.0
            matrix[x, y] = p_x * p_y * tau
            
    total_p = matrix.sum()
    if total_p > 0:
        matrix /= total_p
    return matrix

def simulate_dixon_coles_monte_carlo(lambda_h, mu_a, n=50000, seed=20261111, rho=-0.13, home_adv=1.08, uncertainty=0.15):
    """
    Simulates match outcomes by generating Dixon-Coles score matrices 
    over lognormal uncertainty samples of team ratings.
    """
    rng = np.random.default_rng(int(seed))
    
    # Apply home advantage directly to expected home goals (avoiding Elo double-counting)
    adj_lh_base = max(0.05, lambda_h * home_adv)
    adj_ma_base = max(0.05, mu_a)
    
    # Corrected lognormal location parameterization (mean-preserving)
    sigma2 = uncertainty ** 2
    mu_lh = np.log(adj_lh_base) - (sigma2 / 2.0)
    mu_ma = np.log(adj_ma_base) - (sigma2 / 2.0)
    
    lh_samples = np.maximum(0.05, rng.lognormal(mu_lh, uncertainty, int(n)))
    ma_samples = np.maximum(0.05, rng.lognormal(mu_ma, uncertainty, int(n)))
    
    hg = np.zeros(n, dtype=int)
    ag = np.zeros(n, dtype=int)
    
    max_g = 10
    flat_indices = np.arange((max_g + 1) * (max_g + 1))
    
    # Vectorized sample processing using score distribution matrix
    for i in range(n):
        mat = generate_dixon_coles_matrix(lh_samples[i], ma_samples[i], rho=rho, max_goals=max_g)
        idx = rng.choice(flat_indices, p=mat.ravel())
        hg[i] = idx // (max_g + 1)
        ag[i] = idx % (max_g + 1)
        
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

# ---------------- Elo Rating & Learning Engine ----------------
DEFAULT_ELO = 1500.0

def get_team_elo(team_id):
    if not team_id or team_id >= 8880: # Manual or unregistered IDs return baseline
        return DEFAULT_ELO
    with db() as con:
        cur = con.execute("SELECT rating FROM team_elo_ratings WHERE team_id = ?", (int(team_id),))
        row = cur.fetchone()
        return row['rating'] if row else DEFAULT_ELO

def set_team_elo(team_id, team_name, rating):
    with db() as con:
        con.execute('''INSERT INTO team_elo_ratings (team_id, team_name, rating, last_updated)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(team_id) DO UPDATE SET
                       rating = excluded.rating,
                       last_updated = excluded.last_updated''',
                    (int(team_id), team_name, float(rating), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

def calculate_elo_adjustment(team_elo, opp_elo):
    """Calculates relative opponent rating adjustment multiplier without home bias double-counting."""
    diff = team_elo - opp_elo
    multiplier = 10.0 ** (diff / 600.0)
    return clamp(multiplier, 0.5, 2.0)

def update_elo_ratings(home_id, home_name, away_id, away_name, actual_home, actual_away, k_factor=32.0):
    """Updates team Elo ratings dynamically based on match outcomes."""
    if home_id >= 8880 or away_id >= 8880:
        return
        
    r_home = get_team_elo(home_id)
    r_away = get_team_elo(away_id)
    
    # Expected outcome (incorporating neutral baseline)
    e_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home) / 400.0))
    e_away = 1.0 - e_home
    
    if actual_home > actual_away:
        s_home, s_away = 1.0, 0.0
    elif actual_home == actual_away:
        s_home, s_away = 0.5, 0.5
    else:
        s_home, s_away = 0.0, 1.0
        
    # Margin multiplier
    goal_diff = abs(actual_home - actual_away)
    margin_mult = math.sqrt(max(1, goal_diff))
    
    new_r_home = r_home + k_factor * margin_mult * (s_home - e_home)
    new_r_away = r_away + k_factor * margin_mult * (s_away - e_away)
    
    set_team_elo(home_id, home_name, new_r_home)
    set_team_elo(away_id, away_name, new_r_away)

def apply_bayesian_shrinkage(observed_value, sample_size, baseline_value, k=10.0):
    weight = sample_size / (sample_size + k)
    return (weight * observed_value) + ((1.0 - weight) * baseline_value)

# ---------------- OpenFootball Engine ----------------
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
    history = []
    
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
            history.append({'gf': ft[0], 'ga': ft[1], 'opponent_elo': 1500})
        elif t_clean in t2:
            scored += ft[1]
            conceded += ft[0]
            matches_played += 1
            history.append({'gf': ft[1], 'ga': ft[0], 'opponent_elo': 1500})
            
    if matches_played == 0:
        return None
        
    return {
        'matches_played': matches_played,
        'gf': round(scored / matches_played, 2),
        'ga': round(conceded / matches_played, 2),
        'history': history[-10:]
    }

# ---------------- API Engine with Strict Verification ----------------
def get_api_key():
    if 'api_key' in st.session_state and st.session_state['api_key']:
        return st.session_state['api_key']
    try:
        return st.secrets.get("API_FOOTBALL_KEY", "")
    except Exception:
        return ""

def api_get(path, params, key):
    if not key or not key.strip():
        raise RuntimeError("API Key missing. Enter your key in the sidebar or select Manual Mode.")
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
    rows, _, _ = api_get('teams', {'search': name.strip()}, key)
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
    params = {'team': home_id, 'next': 20, 'season': season}
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
    return {'verified': False, 'fixture_id': None, 'reason': 'Fixture not scheduled in upcoming match list'}

def fetch_recent_team_fixtures(team_id, season, key, last_n=10):
    params = {'team': team_id, 'last': last_n, 'season': season, 'status': 'FT'}
    rows, _, _ = api_get('fixtures', params, key)
    history = []
    for x in rows:
        is_home = (x['teams']['home']['id'] == team_id)
        gf = x['goals']['home'] if is_home else x['goals']['away']
        ga = x['goals']['away'] if is_home else x['goals']['home']
        opp_id = x['teams']['away']['id'] if is_home else x['teams']['home']['id']
        opp_elo = get_team_elo(opp_id)
        if gf is not None and ga is not None:
            history.append({'gf': gf, 'ga': ga, 'opponent_elo': opp_elo})
    return history

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

# ---------------- Sidebar Controls ----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/football-ball-1.png", width=64)
    st.title("Match Lab V4.1 Controls")
    
    api_key_input = st.text_input("API-Football Key", type="password", value=get_api_key(), help="Enter key for live fixture & form data.")
    st.session_state['api_key'] = api_key_input
    
    season_val = st.number_input("Season Year", min_value=2018, max_value=2030, value=2026)
    league_id_val = st.number_input("League ID (Optional)", min_value=0, value=0)
    
    st.divider()
    st.subheader("Monte Carlo Engine Parameters")
    sim_count = st.select_slider("Simulation Runs", options=[10000, 50000, 100000, 200000], value=50000)
    home_adv_val = st.slider("Home Goal Advantage Multiplier", 1.00, 1.25, 1.08, 0.01)
    dixon_coles_rho = st.slider("Dixon-Coles Rho (Draw Tuning)", -0.30, 0.00, -0.13, 0.01)

# ---------------- App Header ----------------
st.title("⚽ Football Match Lab — Version 4.1 (Production Baseline)")
st.caption("Strict Fixture Verification • Dixon-Coles Probability Matrix • Dynamic Elo Ratings • Zero Fabrication")

tabs = st.tabs([
    "🎯 Match Builder",
    "🛡️ Team Analysis",
    "📈 Recent Form",
    "⚔️ Head-to-Head",
    "👤 Player Analysis",
    "🏥 Squad Absences",
    "📋 Confirmed Lineups",
    "🧠 Tactical Overview",
    "🎲 Prediction Engine",
    "📊 Bookmaker Comparison",
    "📉 Accuracy & Settlement",
    "📋 Data Quality & Status"
])

# ---------------- TAB 0: MATCH BUILDER ----------------
with tabs[0]:
    st.subheader("Step 1: Identify Match & Select Mode")
    
    mode = st.radio(
        "Select Operating Mode:",
        [
            "Manual / Standalone Mode (Unverified API)",
            "OpenFootball Data Mode (GitHub JSON Summary)",
            "API-Football Automatic Mode (API Verified)"
        ],
        index=0
    )
    
    c1, c2 = st.columns(2)
    with c1:
        home_name_input = st.text_input("Home Team Name", value="Türkiye", key="mb_home")
        h_g_for = st.number_input("Home Avg Goals Scored", min_value=0.0, max_value=5.0, value=1.1, step=0.1)
        h_g_against = st.number_input("Home Avg Goals Conceded", min_value=0.0, max_value=5.0, value=1.9, step=0.1)
    with c2:
        away_name_input = st.text_input("Away Team Name", value="Belgium", key="mb_away")
        a_g_for = st.number_input("Away Avg Goals Scored", min_value=0.0, max_value=5.0, value=1.75, step=0.1)
        a_g_against = st.number_input("Away Avg Goals Conceded", min_value=0.0, max_value=5.0, value=1.1, step=0.1)
        
    if mode.startswith("Manual"):
        is_senior = st.checkbox("Confirm Senior Teams (Not Youth, U21, or Reserves)", value=True)
        if st.button("✅ Confirm Manual Inputs & Unlock Prediction Engine", type="primary", use_container_width=True):
            if not is_senior:
                st.error("Must confirm senior team status before running analysis.")
            else:
                st.session_state['current_match_v4'] = {
                    'verified': False,  # MUST NEVER BE True FOR MANUAL
                    'verification_status': 'Manual Mode (Unverified API)',
                    'home_team': {'id': 9991, 'name': home_name_input, 'country': 'Manual'},
                    'away_team': {'id': 9992, 'name': away_name_input, 'country': 'Manual'},
                    'recent_h': [{'gf': h_g_for, 'ga': h_g_against, 'opponent_elo': 1500}] * 5,
                    'recent_a': [{'gf': a_g_for, 'ga': a_g_against, 'opponent_elo': 1500}] * 5,
                    'h2h': [], 'injuries_h': [], 'injuries_a': [], 'lineups': []
                }
                st.success(f"Manual fixture '{home_name_input} vs {away_name_input}' set.")
                st.rerun()

    elif mode.startswith("OpenFootball"):
        st.info("Fetches goal averages automatically from open-source GitHub datasets.")
        league_code_input = st.selectbox("Select Competition Code", ["en.1", "es.1", "de.1", "it.1"], format_func=lambda x: {"en.1": "Premier League", "es.1": "La Liga", "de.1": "Bundesliga", "it.1": "Serie A"}[x])
        season_code_input = st.text_input("Season Format (e.g. 2024-25)", value="2024-25")
        
        if st.button("🌐 Fetch OpenFootball Dataset Stats", type="primary", use_container_width=True):
            with st.spinner("Fetching dataset..."):
                h_stats = calc_openfootball_team_stats(home_name_input, season_code_input, league_code_input)
                a_stats = calc_openfootball_team_stats(away_name_input, season_code_input, league_code_input)
                
                h_gf = h_stats['gf'] if h_stats else h_g_for
                h_ga = h_stats['ga'] if h_stats else h_g_against
                a_gf = a_stats['gf'] if a_stats else a_g_for
                a_ga = a_stats['ga'] if a_stats else a_g_against
                
                st.session_state['current_match_v4'] = {
                    'verified': False, # NEVER MARKED API VERIFIED
                    'verification_status': 'OpenFootball Summary Data (Unverified Fixture)',
                    'home_team': {'id': 8881, 'name': home_name_input, 'country': 'OpenFootball'},
                    'away_team': {'id': 8882, 'name': away_name_input, 'country': 'OpenFootball'},
                    'recent_h': h_stats['history'] if h_stats else [{'gf': h_gf, 'ga': h_ga, 'opponent_elo': 1500}] * 5,
                    'recent_a': a_stats['history'] if a_stats else [{'gf': a_gf, 'ga': a_ga, 'opponent_elo': 1500}] * 5,
                    'h2h': [], 'injuries_h': [], 'injuries_a': [], 'lineups': []
                }
                st.success(f"OpenFootball Stats Loaded: {home_name_input} ({h_gf} GF) vs {away_name_input} ({a_gf} GF).")
                st.rerun()

    else:
        btn_verify = st.button("🔍 Verify Fixture & Fetch Real Data via API", use_container_width=True, type="primary")
        if btn_verify:
            key = get_api_key()
            if not key:
                st.error("API Key missing. Enter key in sidebar or switch to Manual Mode.")
            else:
                with st.spinner("Searching API & verifying fixture..."):
                    try:
                        ht = resolve_exact_team(home_name_input, key)
                        at = resolve_exact_team(away_name_input, key)
                        if not ht or not at:
                            st.error("Could not resolve one or both senior teams.")
                            st.session_state['current_match_v4'] = {'verified': False, 'verification_status': 'Team Resolution Failed'}
                        else:
                            hid, aid = ht['team']['id'], at['team']['id']
                            fix_info = verify_and_fetch_fixture(hid, aid, season_val, key, league_id_val)
                            
                            # Fetch real historical match records
                            h_hist = fetch_recent_team_fixtures(hid, season_val, key, last_n=10)
                            a_hist = fetch_recent_team_fixtures(aid, season_val, key, last_n=10)
                            
                            # CRITICAL FIX 1: STRICT VERIFICATION ASSIGNMENT
                            is_fixture_verified = bool(fix_info.get('verified', False))
                            v_status = 'API Verified' if is_fixture_verified else 'Fixture Not Verified (Unscheduled API Match)'
                            
                            st.session_state['current_match_v4'] = {
                                'verified': is_fixture_verified,
                                'verification_status': v_status,
                                'fixture_info': fix_info,
                                'home_team': ht['team'],
                                'away_team': at['team'],
                                'recent_h': h_hist if h_hist else [{'gf': h_g_for, 'ga': h_g_against, 'opponent_elo': 1500}] * 5,
                                'recent_a': a_hist if a_hist else [{'gf': a_g_for, 'ga': a_g_against, 'opponent_elo': 1500}] * 5,
                                'h2h': [], 'injuries_h': [], 'injuries_a': [], 'lineups': []
                            }
                            if is_fixture_verified:
                                st.success(f"✅ Fixture Confirmed: **{ht['team']['name']}** vs **{at['team']['name']}** ({fix_info['match_date']})")
                            else:
                                st.warning(f"⚠️ Teams found, but exact scheduled match not confirmed in upcoming API list.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"API Verification error: {e}")
                        st.session_state['current_match_v4'] = {'verified': False, 'verification_status': f'Error: {str(e)}'}

    match_data = st.session_state.get('current_match_v4', {})
    v_status = match_data.get('verification_status', 'Unverified')
    if match_data.get('verified'):
        st.success(f"✅ **Fixture Status:** {match_data['home_team']['name']} vs {match_data['away_team']['name']} ({v_status})")
    else:
        st.info(f"ℹ️ **Current Mode / Status:** {v_status}")

# ---------------- TAB 1: TEAM ANALYSIS ----------------
with tabs[1]:
    st.subheader("Team Identity & Elo Ratings")
    match_data = st.session_state.get('current_match_v4', {})
    if 'home_team' not in match_data:
        st.warning("Please configure a fixture in Match Builder first.")
    else:
        c1, c2 = st.columns(2)
        h_elo = get_team_elo(match_data['home_team']['id'])
        a_elo = get_team_elo(match_data['away_team']['id'])
        
        c1.markdown(f"### 🏠 {match_data['home_team']['name']}")
        c1.metric("Current Elo Rating", f"{h_elo:.1f}")
        
        c2.markdown(f"### ✈️ {match_data['away_team']['name']}")
        c2.metric("Current Elo Rating", f"{a_elo:.1f}")
        
        st.divider()
        st.subheader("Update Elo Manual Override")
        e_c1, e_c2 = st.columns(2)
        new_h_elo = e_c1.number_input(f"{match_data['home_team']['name']} Elo", value=float(h_elo))
        new_a_elo = e_c2.number_input(f"{match_data['away_team']['name']} Elo", value=float(a_elo))
        if st.button("Save Manual Elo Ratings"):
            set_team_elo(match_data['home_team']['id'], match_data['home_team']['name'], new_h_elo)
            set_team_elo(match_data['away_team']['id'], match_data['away_team']['name'], new_a_elo)
            st.success("Elo ratings updated!")

# ---------------- TAB 2: RECENT FORM ----------------
with tabs[2]:
    st.subheader("📈 Opponent-Adjusted Match History & Form")
    match_data = st.session_state.get('current_match_v4', {})
    if 'home_team' not in match_data:
        st.warning("Please configure a fixture in Match Builder first.")
    else:
        st.write("Match history records used to calculate base goal expectations:")
        col_h, col_a = st.columns(2)
        
        with col_h:
            st.markdown(f"#### 🏠 {match_data['home_team']['name']} Recent Matches")
            h_df = pd.DataFrame(match_data.get('recent_h', []))
            if not h_df.empty:
                edited_h = st.data_editor(h_df, num_rows="dynamic", key="form_h_edit", use_container_width=True)
                match_data['recent_h'] = edited_h.to_dict(orient="records")
            else:
                st.info("No match history loaded.")
                
        with col_a:
            st.markdown(f"#### ✈️ {match_data['away_team']['name']} Recent Matches")
            a_df = pd.DataFrame(match_data.get('recent_a', []))
            if not a_df.empty:
                edited_a = st.data_editor(a_df, num_rows="dynamic", key="form_a_edit", use_container_width=True)
                match_data['recent_a'] = edited_a.to_dict(orient="records")
            else:
                st.info("No match history loaded.")

# ---------------- TAB 3: HEAD-TO-HEAD HISTORY ----------------
with tabs[3]:
    st.subheader("⚔️ Head-to-Head (H2H) Record")
    match_data = st.session_state.get('current_match_v4', {})
    if 'home_team' not in match_data:
        st.warning("Please configure a fixture in Match Builder first.")
    else:
        # CRITICAL FIX 2: REMOVED FABRICATED DEFAULTS
        h2h_list = match_data.get('h2h', [])
        if not h2h_list:
            st.info("ℹ️ No verified H2H records fetched. You may manually enter past meetings below:")
            h2h_df = pd.DataFrame(columns=["Date", "Home", "Score", "Away"])
        else:
            h2h_df = pd.DataFrame(h2h_list)
            
        edited_h2h = st.data_editor(h2h_df, num_rows="dynamic", key="h2h_editor", use_container_width=True)
        if st.button("💾 Save Manual H2H Data"):
            match_data['h2h'] = edited_h2h.to_dict(orient="records")
            st.session_state['current_match_v4'] = match_data
            st.success("H2H data saved.")

# ---------------- TAB 4: PLAYER ANALYSIS ----------------
with tabs[4]:
    st.subheader("👤 Player Analysis")
    # CRITICAL FIX 2: REMOVED FAKE PLAYERS
    st.info("ℹ️ Verified individual player performance statistics not fetched. Add custom metrics below if desired:")
    p_df = pd.DataFrame(columns=["Team", "Player Name", "Position", "Rating", "xG_90"])
    st.data_editor(p_df, num_rows="dynamic", key="p_analysis_edit", use_container_width=True)

# ---------------- TAB 5: SQUAD ABSENCES ----------------
with tabs[5]:
    st.subheader("🏥 Squad Absences & Injuries")
    match_data = st.session_state.get('current_match_v4', {})
    # CRITICAL FIX 2 & 5: REMOVED GUESSED PENALTIES & FAKE INJURIES
    st.info("ℹ️ Verified injury feeds require live team news. Enter confirmed absences for display below:")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🏠 Home Absences")
        inj_h_df = pd.DataFrame(match_data.get('injuries_h', []), columns=["Player", "Reason", "Status"])
        edited_inj_h = st.data_editor(inj_h_df, num_rows="dynamic", key="inj_h_edit", use_container_width=True)
    with c2:
        st.markdown("#### ✈️ Away Absences")
        inj_a_df = pd.DataFrame(match_data.get('injuries_a', []), columns=["Player", "Reason", "Status"])
        edited_inj_a = st.data_editor(inj_a_df, num_rows="dynamic", key="inj_a_edit", use_container_width=True)
        
    if st.button("💾 Save Absences"):
        match_data['injuries_h'] = edited_inj_h.to_dict(orient="records")
        match_data['injuries_a'] = edited_inj_a.to_dict(orient="records")
        st.session_state['current_match_v4'] = match_data
        st.success("Absences saved for display.")

# ---------------- TAB 6: CONFIRMED LINEUPS ----------------
with tabs[6]:
    st.subheader("📋 Lineups & Formations")
    match_data = st.session_state.get('current_match_v4', {})
    if 'home_team' not in match_data:
        st.warning("Please configure a fixture in Match Builder first.")
    else:
        # CRITICAL FIX 2: REMOVED UNVERIFIED PRESETS FROM BEING LABELED CONFIRMED
        st.write("Enter starting XIs. Only API-retrieved XIs should be considered official confirmed lineups.")
        col1, col2 = st.columns(2)
        with col1:
            h_form = st.text_input("Home Formation", value="4-2-3-1", key="h_form_val")
            h_lineup = st.text_area("Home Starting XI", value="", placeholder="Enter 11 starting players...", key="h_lineup_val")
        with col2:
            a_form = st.text_input("Away Formation", value="4-2-3-1", key="a_form_val")
            a_lineup = st.text_area("Away Starting XI", value="", placeholder="Enter 11 starting players...", key="a_lineup_val")
            
        if st.button("Save Lineups"):
            match_data['lineups'] = {
                'home': {'formation': h_form, 'starting': h_lineup.split('\n')},
                'away': {'formation': a_form, 'starting': a_lineup.split('\n')}
            }
            st.session_state['current_match_v4'] = match_data
            st.success("Lineup notes saved.")

# ---------------- TAB 7: TACTICAL OVERVIEW ----------------
with tabs[7]:
    st.subheader("🧠 Tactical Overview")
    st.info("ℹ️ Tactical adjustments are held neutral (1.0x) to avoid unvalidated bias multipliers in Poisson calculations.")

# ---------------- TAB 8: PREDICTION ENGINE ----------------
with tabs[8]:
    st.subheader("Dixon-Coles Bivariate Poisson Monte Carlo Engine")
    match_data = st.session_state.get('current_match_v4', {})
    
    if 'home_team' not in match_data:
        st.error("⚠️ **Engine Locked:** Configure a match in Match Builder first.")
    else:
        if st.button("🚀 Run Monte Carlo Simulation", type="primary", use_container_width=True):
            df_h = pd.DataFrame(match_data.get('recent_h', []))
            df_a = pd.DataFrame(match_data.get('recent_a', []))
            
            raw_h_for = df_h['gf'].mean() if not df_h.empty else 1.1
            raw_h_against = df_h['ga'].mean() if not df_h.empty else 1.9
            raw_a_for = df_a['gf'].mean() if not df_a.empty else 1.75
            raw_a_against = df_a['ga'].mean() if not df_a.empty else 1.1
            
            # Bayesian shrinkage towards league average (~1.35 goals)
            h_for_shrunk = apply_bayesian_shrinkage(raw_h_for, len(df_h), 1.35, k=10.0)
            a_for_shrunk = apply_bayesian_shrinkage(raw_a_for, len(df_a), 1.35, k=10.0)
            
            # Opponent Elo Rating Adjustments
            elo_h = get_team_elo(match_data['home_team']['id'])
            elo_a = get_team_elo(match_data['away_team']['id'])
            
            adj_h_mult = calculate_elo_adjustment(elo_h, elo_a)
            adj_a_mult = calculate_elo_adjustment(elo_a, elo_h)

            # CRITICAL FIX 5: REMOVED ARBITRARY INJURY PENALTY MULTIPLIER
            # Expected Goals calculation based purely on performance statistics
            lambda_h = (0.6 * h_for_shrunk + 0.4 * raw_a_against) * adj_h_mult
            mu_a = (0.6 * a_for_shrunk + 0.4 * raw_h_against) * adj_a_mult
            
            # CRITICAL FIX 3: MATHEMATICALLY CORRECT DIXON-COLES SAMPLING
            hg, ag = simulate_dixon_coles_monte_carlo(
                lambda_h, mu_a, n=sim_count, seed=20261111, rho=dixon_coles_rho, home_adv=home_adv_val
            )
            
            res = model_probabilities_from_samples(hg, ag)
            st.session_state['last_prediction_res_v4'] = res
            
            st.success("Simulation Completed Successfully!")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Home Win Probability", f"{res['p_home']:.1%}")
            c2.metric("Draw Probability", f"{res['p_draw']:.1%}")
            c3.metric("Away Win Probability", f"{res['p_away']:.1%}")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Over 1.5 Goals", f"{res['p_over15']:.1%}")
            m2.metric("Over 2.5 Goals", f"{res['p_over25']:.1%}")
            m3.metric("Over 3.5 Goals", f"{res['p_over35']:.1%}")
            m4.metric("BTTS Yes", f"{res['p_btts']:.1%}")
            
            st.markdown(f"**Expected Goals (Mean):** {match_data['home_team']['name']} **{res['exp_home']:.2f}** — **{res['exp_away']:.2f}** {match_data['away_team']['name']}")
            
            st.markdown("### Most Likely Correct Scores")
            scores_df = res['scores'].copy()
            scores_df['score'] = scores_df.apply(lambda x: f"{int(x.home)}-{int(x.away)}", axis=1)
            st.dataframe(scores_df[['score', 'probability']].style.format({'probability': '{:.2%}'}), hide_index=True, use_container_width=True)

            # Option to save prediction to DB
            if st.button("💾 Save Prediction to Database"):
                with db() as con:
                    con.execute('''INSERT INTO predictions (
                        created_at, home_id, away_id, home, away, competition, simulations,
                        p_home, p_draw, p_away, p_over15, p_over25, p_over35, p_over45, p_btts,
                        exp_home, exp_away, verification_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        match_data['home_team']['id'], match_data['away_team']['id'],
                        match_data['home_team']['name'], match_data['away_team']['name'],
                        match_data.get('fixture_info', {}).get('competition', 'N/A'),
                        sim_count, res['p_home'], res['p_draw'], res['p_away'],
                        res['p_over15'], res['p_over25'], res['p_over35'], res['p_over45'],
                        res['p_btts'], res['exp_home'], res['exp_away'],
                        match_data.get('verification_status', 'Unverified')
                    ))
                st.success("Prediction logged in database.")

# ---------------- TAB 9: BOOKMAKER COMPARISON ----------------
with tabs[9]:
    st.subheader("Market Odds & Value Comparison")
    c1, c2, c3 = st.columns(3)
    odd_h = c1.number_input("Bookmaker Home Odds", min_value=1.01, value=3.95, step=0.05)
    odd_d = c2.number_input("Bookmaker Draw Odds", min_value=1.01, value=3.40, step=0.05)
    odd_a = c3.number_input("Bookmaker Away Odds", min_value=1.01, value=1.85, step=0.05)
        
    bm_res = analyze_bookmaker_odds(odd_h, odd_d, odd_a)
    if bm_res:
        st.write(f"**Bookmaker Overround Margin:** {bm_res['margin_pct']:.2f}%")
        pred_res = st.session_state.get('last_prediction_res_v4')
        if pred_res:
            comp_data = [
                {
                    'Outcome': 'Home Win (1)',
                    'Model Prob': f"{pred_res['p_home']:.1%}",
                    'Bookmaker Implied': f"{bm_res['implied_h']:.1%}",
                    'Edge (Diff)': f"{(pred_res['p_home'] - bm_res['implied_h']) * 100:+.1f}%"
                },
                {
                    'Outcome': 'Draw (X)',
                    'Model Prob': f"{pred_res['p_draw']:.1%}",
                    'Bookmaker Implied': f"{bm_res['implied_d']:.1%}",
                    'Edge (Diff)': f"{(pred_res['p_draw'] - bm_res['implied_d']) * 100:+.1f}%"
                },
                {
                    'Outcome': 'Away Win (2)',
                    'Model Prob': f"{pred_res['p_away']:.1%}",
                    'Bookmaker Implied': f"{bm_res['implied_a']:.1%}",
                    'Edge (Diff)': f"{(pred_res['p_away'] - bm_res['implied_a']) * 100:+.1f}%"
                }
            ]
            st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

# ---------------- TAB 10: ACCURACY TRACKER & SETTLEMENT ----------------
with tabs[10]:
    st.subheader("Historical Predictions & Score Settlement")
    
    with db() as con:
        df_preds = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", con)
        
    if df_preds.empty:
        st.info("No saved predictions in local database.")
    else:
        st.dataframe(df_preds[['id', 'created_at', 'home', 'away', 'p_home', 'p_draw', 'p_away', 'actual_home', 'actual_away', 'verification_status']], use_container_width=True)
        
        st.divider()
        st.subheader("Settle Match Score & Update Elo")
        pred_id = st.number_input("Prediction ID to Settle", min_value=1, step=1)
        sc1, sc2 = st.columns(2)
        act_h = sc1.number_input("Actual Home Goals", min_value=0, step=1, value=0)
        act_a = sc2.number_input("Actual Away Goals", min_value=0, step=1, value=0)
        
        if st.button("Settle Prediction & Learn Elo"):
            target_row = df_preds[df_preds['id'] == pred_id]
            if not target_row.empty:
                r = target_row.iloc[0]
                with db() as con:
                    con.execute('''UPDATE predictions SET 
                                    actual_home = ?, actual_away = ?, settled_at = ?
                                    WHERE id = ?''', (act_h, act_a, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pred_id))
                                    
                # CRITICAL FIX 4: ELO LEARNING ENGINE TRIGGER
                update_elo_ratings(r['home_id'], r['home'], r['away_id'], r['away'], act_h, act_a)
                st.success(f"Match settled! Updated Elo ratings for {r['home']} and {r['away']}.")
                st.rerun()

# ---------------- TAB 11: DATA QUALITY ----------------
with tabs[11]:
    st.subheader("Data Quality & System Status")
    match_data = st.session_state.get('current_match_v4', {})
    st.write(f"**Active Mode:** {match_data.get('verification_status', 'Unverified')}")
    st.write(f"**API Verified Status:** {match_data.get('verified', False)}")
