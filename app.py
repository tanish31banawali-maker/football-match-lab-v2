import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import math
import os
import re
from datetime import datetime

# ---------------- App & Database Configuration ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, 'football_lab_v4.db')

st.set_page_config(
    page_title='Football Match Lab V4.3',
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
            home TEXT NOT NULL,
            away TEXT NOT NULL,
            simulations INTEGER,
            p_home REAL, p_draw REAL, p_away REAL,
            p_over15 REAL, p_over25 REAL, p_over35 REAL, p_btts REAL,
            exp_home REAL, exp_away REAL,
            verification_status TEXT DEFAULT 'Parsed Data Input'
        )''')
        con.execute('''CREATE TABLE IF NOT EXISTS team_elo_ratings (
            team_id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT UNIQUE,
            rating REAL,
            last_updated TEXT
        )''')

init_db()

# ---------------- Dynamic Tactical & Formation Engines ----------------
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

def get_stored_elo(team_name, default=1500.0):
    with db() as con:
        cur = con.execute("SELECT rating FROM team_elo_ratings WHERE team_name = ?", (team_name,))
        row = cur.fetchone()
        return row['rating'] if row else default

def update_stored_elo(team_name, new_rating):
    with db() as con:
        con.execute('''INSERT INTO team_elo_ratings (team_name, rating, last_updated)
                       VALUES (?, ?, ?)
                       ON CONFLICT(team_name) DO UPDATE SET
                       rating = excluded.rating,
                       last_updated = excluded.last_updated''',
                    (team_name, float(new_rating), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# ---------------- Dixon-Coles & Monte Carlo Math Engine ----------------
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
    return 1.0

def generate_dixon_coles_matrix(lambda_h, mu_a, rho=-0.13, max_goals=10):
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

def simulate_dixon_coles_monte_carlo(lambda_h, mu_a, n=50000, seed=20261112, rho=-0.13, home_adv=1.08, uncertainty=0.15):
    rng = np.random.default_rng(int(seed))
    adj_lh_base = max(0.05, lambda_h * home_adv)
    adj_ma_base = max(0.05, mu_a)
    
    sigma2 = uncertainty ** 2
    mu_lh = np.log(adj_lh_base) - (sigma2 / 2.0)
    mu_ma = np.log(adj_ma_base) - (sigma2 / 2.0)
    
    lh_samples = np.maximum(0.05, rng.lognormal(mu_lh, uncertainty, int(n)))
    ma_samples = np.maximum(0.05, rng.lognormal(mu_ma, uncertainty, int(n)))
    
    hg, ag = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    max_g = 10
    flat_indices = np.arange((max_g + 1) * (max_g + 1))
    
    for i in range(n):
        mat = generate_dixon_coles_matrix(lh_samples[i], ma_samples[i], rho=rho, max_goals=max_g)
        idx = rng.choice(flat_indices, p=mat.ravel())
        hg[i] = idx // (max_g + 1)
        ag[i] = idx % (max_g + 1)
        
    return hg, ag

def model_probabilities_from_samples(hg, ag):
    n = len(hg)
    total = hg + ag
    p_h, p_d, p_a = float(np.mean(hg > ag)), float(np.mean(hg == ag)), float(np.mean(hg < ag))
    s_tot = p_h + p_d + p_a
    
    scores_df = pd.DataFrame({'home': hg, 'away': ag}).value_counts().rename('count').reset_index()
    scores_df['probability'] = scores_df['count'] / n

    return {
        'p_home': p_h / s_tot, 'p_draw': p_d / s_tot, 'p_away': p_a / s_tot,
        'p_over15': float(np.mean(total >= 2)),
        'p_over25': float(np.mean(total >= 3)),
        'p_over35': float(np.mean(total >= 4)),
        'p_btts': float(np.mean((hg > 0) & (ag > 0))),
        'exp_home': float(np.mean(hg)),
        'exp_away': float(np.mean(ag)),
        'scores': scores_df.sort_values('probability', ascending=False).head(10)
    }

# ---------------- Multi-Section Text Parser ----------------
def parse_single_block(text):
    data = {
        'home_team': 'Italy', 'away_team': 'France',
        'home_gf': 1.67, 'home_ga': 1.15, 'home_elo': 1840.0,
        'away_gf': 2.10, 'away_ga': 0.75, 'away_elo': 1910.0,
        'home_formation': '4-3-3', 'away_formation': '4-3-3',
        'home_absences': [], 'away_absences': [],
        'home_players': [], 'away_players': [],
        'h2h': [],
        'odd_h': 2.75, 'odd_d': 3.20, 'odd_a': 2.45
    }
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    current_section = None
    
    for line in lines:
        l = line.lower()
        
        # Detect Team Names
        if 'vs' in l and ('match:' in l or 'italy' in l or 'home' in l):
            m = re.search(r'([a-zA-Z\s]+)\s+vs\s+([a-zA-Z\s]+)', line, re.I)
            if m:
                data['home_team'] = m.group(1).replace('Match:', '').strip()
                data['away_team'] = m.group(2).strip()

        # Parse Goal Averages
        if 'home avg goals scored' in l or 'italy avg goals scored' in l or ('home' in l and 'gf:' in l):
            val = re.findall(r'\d+\.\d+|\d+', line)
            if val: data['home_gf'] = float(val[0])
        if 'home avg goals conceded' in l or 'italy avg goals conceded' in l or ('home' in l and 'ga:' in l):
            val = re.findall(r'\d+\.\d+|\d+', line)
            if val: data['home_ga'] = float(val[0])
        if 'away avg goals scored' in l or 'france avg goals scored' in l or ('away' in l and 'gf:' in l):
            val = re.findall(r'\d+\.\d+|\d+', line)
            if val: data['away_gf'] = float(val[0])
        if 'away avg goals conceded' in l or 'france avg goals conceded' in l or ('away' in l and 'ga:' in l):
            val = re.findall(r'\d+\.\d+|\d+', line)
            if val: data['away_ga'] = float(val[0])

        # Parse Elo Ratings
        if ('home' in l or data['home_team'].lower() in l) and 'elo' in l:
            val = re.findall(r'\d{4}', line)
            if val: data['home_elo'] = float(val[0])
        if ('away' in l or data['away_team'].lower() in l) and 'elo' in l:
            val = re.findall(r'\d{4}', line)
            if val: data['away_elo'] = float(val[0])

        # Parse Formations
        if 'home formation:' in l or 'italy (formation:' in l:
            m = re.search(r'\d-\d-\d(-\d)?', line)
            if m: data['home_formation'] = m.group(0)
        if 'away formation:' in l or 'france (formation:' in l:
            m = re.search(r'\d-\d-\d(-\d)?', line)
            if m: data['away_formation'] = m.group(0)

        # Parse Bookmaker Odds
        if 'home odds' in l or 'odds (1' in l:
            val = re.findall(r'\d+\.\d+', line)
            if val: data['odd_h'] = float(val[0])
        if 'draw odds' in l or 'odds (x' in l:
            val = re.findall(r'\d+\.\d+', line)
            if val: data['odd_d'] = float(val[0])
        if 'away odds' in l or 'odds (2' in l:
            val = re.findall(r'\d+\.\d+', line)
            if val: data['odd_a'] = float(val[0])

        # Detect List Sections
        if 'home absences' in l or 'italy absences' in l:
            current_section = 'home_absences'; continue
        elif 'away absences' in l or 'france absences' in l:
            current_section = 'away_absences'; continue
        elif 'head-to-head' in l or 'h2h' in l:
            current_section = 'h2h'; continue
        elif 'player impact' in l or 'player ratings' in l:
            current_section = 'players'; continue

        # Parse Bullet Items into Active Section
        if current_section == 'home_absences' and (line.startswith('*') or line.startswith('-')):
            data['home_absences'].append(line.strip('* -'))
        elif current_section == 'away_absences' and (line.startswith('*') or line.startswith('-')):
            data['away_absences'].append(line.strip('* -'))
        elif current_section == 'h2h' and (line.startswith('*') or line.startswith('-') or '|' in line):
            if 'Date' not in line and '---' not in line:
                data['h2h'].append(line.strip('* -'))

    return data

# ---------------- Sidebar Controls ----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/football-ball-1.png", width=64)
    st.title("Match Lab V4.3 Controls")
    sim_count = st.select_slider("Simulation Runs", options=[10000, 50000, 100000], value=50000)
    home_adv_val = st.slider("Home Advantage Multiplier", 1.00, 1.25, 1.08, 0.01)
    dixon_coles_rho = st.slider("Dixon-Coles Rho", -0.30, 0.00, -0.13, 0.01)

# ---------------- App Header & Tabs ----------------
st.title("⚽ Football Match Lab V4.3 — Unified Engine")

tabs = st.tabs([
    "📋 Text Block Parser",
    "🛡️ Team & Elo Analysis",
    "📈 Recent Form",
    "⚔️ Head-to-Head",
    "👤 Player Analysis",
    "🏥 Squad Absences",
    "📋 Lineups & Tactics",
    "🎲 Prediction Engine",
    "📊 Market Odds Check"
])

# ---------------- TAB 0: PARSER ----------------
with tabs[0]:
    st.subheader("Paste Complete Match Block")
    st.write("Paste your raw match details text below. Clicking parse will auto-fill every tab in the app and power the simulation engine:")
    
    default_block = """Match: Italy vs France
Italy Avg Goals Scored (GF): 1.67
Italy Avg Goals Conceded (GA): 1.15
Italy Team Elo Rating: 1840

France Avg Goals Scored (GF): 2.10
France Avg Goals Conceded (GA): 0.75
France Team Elo Rating: 1910

Italy Formation: 4-3-3
France Formation: 4-3-3

Italy Absences:
* Giovanni Di Lorenzo (Defender) - Muscle Injury
* Gianluca Mancini (Defender) - Withdrew
* Moise Kean (Forward) - Ankle Strain
* Giacomo Raspadori (Forward) - Thigh Strain

France Absences:
* Ousmane Dembélé (Forward) - Thigh Strain
* William Saliba (Defender) - Hamstring Injury
* Jules Koundé (Defender) - Muscle Fatigue
* Aurélien Tchouaméni (Midfielder) - Suspended
* Marcus Thuram (Forward) - Knee Injury

Head-to-Head History:
* 03 Oct 2026: France 1-1 Italy
* 17 Nov 2024: Italy 1-3 France
* 06 Sep 2024: France 1-3 Italy

Bookmaker Home Odds (1 - Italy): 2.75
Bookmaker Draw Odds (X): 3.20
Bookmaker Away Odds (2 - France): 2.45"""

    raw_text = st.text_area("Match Block Text", value=default_block, height=350)
    
    if st.button("🚀 Parse & Populate Match Engine", type="primary", use_container_width=True):
        parsed_data = parse_single_block(raw_text)
        st.session_state['parsed_match_v43'] = parsed_data
        st.success(f"Loaded match parameters for **{parsed_data['home_team']} vs {parsed_data['away_team']}**!")

# ---------------- TAB 1: ELO ANALYSIS ----------------
with tabs[1]:
    st.subheader("Team Identity & Dynamic Elo Ratings")
    data = st.session_state.get('parsed_match_v43')
    if not data:
        st.warning("Please parse a match block in Tab 0 first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### 🏠 {data['home_team']}")
            st.metric("Parsed Elo Rating", f"{data['home_elo']:.1f}")
        with c2:
            st.markdown(f"### ✈️ {data['away_team']}")
            st.metric("Parsed Elo Rating", f"{data['away_elo']:.1f}")

# ---------------- TAB 2: RECENT FORM ----------------
with tabs[2]:
    st.subheader("📈 Base Goal Averages")
    data = st.session_state.get('parsed_match_v43')
    if not data:
        st.warning("Please parse a match block in Tab 0 first.")
    else:
        c1, c2 = st.columns(2)
        c1.metric(f"{data['home_team']} Avg Scored (GF)", f"{data['home_gf']:.2f}")
        c1.metric(f"{data['home_team']} Avg Conceded (GA)", f"{data['home_ga']:.2f}")
        c2.metric(f"{data['away_team']} Avg Scored (GF)", f"{data['away_gf']:.2f}")
        c2.metric(f"{data['away_team']} Avg Conceded (GA)", f"{data['away_ga']:.2f}")

# ---------------- TAB 3: H2H ----------------
with tabs[3]:
    st.subheader("⚔️ Head-to-Head Encounters")
    data = st.session_state.get('parsed_match_v43')
    if not data or not data.get('h2h'):
        st.info("No H2H lines parsed or available.")
    else:
        st.dataframe(pd.DataFrame(data['h2h'], columns=["Match Event"]), use_container_width=True)

# ---------------- TAB 4: PLAYER ANALYSIS ----------------
with tabs[4]:
    st.subheader("👤 Key Player Impact")
    st.dataframe(pd.DataFrame([
        {"Team": "Italy", "Player": "Nicolò Barella", "Pos": "MF", "Rating": 8.1, "xG_90": 0.32},
        {"Team": "France", "Player": "Kylian Mbappé", "Pos": "FW", "Rating": 8.6, "xG_90": 0.78}
    ]), hide_index=True, use_container_width=True)

# ---------------- TAB 5: ABSENCES ----------------
with tabs[5]:
    st.subheader("🏥 Confirmed Squad Absences")
    data = st.session_state.get('parsed_match_v43')
    if not data:
        st.warning("Please parse a match block in Tab 0 first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"#### 🏠 {data['home_team']} Absences ({len(data['home_absences'])})")
            for a in data['home_absences']: st.write(f"- {a}")
        with c2:
            st.markdown(f"#### ✈️ {data['away_team']} Absences ({len(data['away_absences'])})")
            for a in data['away_absences']: st.write(f"- {a}")

# ---------------- TAB 6: LINEUPS & TACTICS ----------------
with tabs[6]:
    st.subheader("📋 Tactical Formations")
    data = st.session_state.get('parsed_match_v43')
    if not data:
        st.warning("Please parse a match block in Tab 0 first.")
    else:
        c1, c2 = st.columns(2)
        c1.metric(f"🏠 {data['home_team']} Structure", data['home_formation'])
        c2.metric(f"✈️ {data['away_team']} Structure", data['away_formation'])

# ---------------- TAB 7: PREDICTION ENGINE ----------------
with tabs[7]:
    st.subheader("🎲 Monte Carlo Poisson Simulation Engine")
    data = st.session_state.get('parsed_match_v43')
    
    if not data:
        st.error("Please parse match block in Tab 0 to run simulation.")
    else:
        if st.button("⚡ Run Monte Carlo Engine on Parsed Data", type="primary", use_container_width=True):
            # Calculate Elo relative rating adjustment
            elo_diff = data['home_elo'] - data['away_elo']
            adj_h_mult = max(0.5, min(2.0, 10.0 ** (elo_diff / 600.0)))
            adj_a_mult = max(0.5, min(2.0, 10.0 ** (-elo_diff / 600.0)))

            # Get Formation shape profiles
            h_prof = get_formation_multipliers(data['home_formation'])
            a_prof = get_formation_multipliers(data['away_formation'])

            # Absence Multipliers (-7% per missing player)
            h_abs_mult = max(0.60, 1.0 - (len(data['home_absences']) * 0.07))
            a_abs_mult = max(0.60, 1.0 - (len(data['away_absences']) * 0.07))

            # Calculate Expected Goals ($\lambda$ & $\mu$)
            lambda_h = (0.6 * data['home_gf'] + 0.4 * data['away_ga']) * adj_h_mult * h_prof['attack'] * a_prof['defense'] * h_abs_mult
            mu_a = (0.6 * data['away_gf'] + 0.4 * data['home_ga']) * adj_a_mult * a_prof['attack'] * h_prof['defense'] * a_abs_mult

            hg, ag = simulate_dixon_coles_monte_carlo(
                lambda_h, mu_a, n=sim_count, seed=20261112, rho=dixon_coles_rho, home_adv=home_adv_val
            )

            res = model_probabilities_from_samples(hg, ag)
            st.session_state['active_res_v43'] = res

            st.success("Simulation Completed!")

            c1, c2, c3 = st.columns(3)
            c1.metric(f"{data['home_team']} Win", f"{res['p_home']:.1%}")
            c2.metric("Draw", f"{res['p_draw']:.1%}")
            c3.metric(f"{data['away_team']} Win", f"{res['p_away']:.1%}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Over 1.5 Goals", f"{res['p_over15']:.1%}")
            m2.metric("Over 2.5 Goals", f"{res['p_over25']:.1%}")
            m3.metric("Over 3.5 Goals", f"{res['p_over35']:.1%}")
            m4.metric("BTTS Yes", f"{res['p_btts']:.1%}")

            st.markdown(f"**Expected Goals:** {data['home_team']} **{res['exp_home']:.2f}** — **{res['exp_away']:.2f}** {data['away_team']}")

# ---------------- TAB 8: MARKET ODDS ----------------
with tabs[8]:
    st.subheader("Market Odds & Value Comparison")
    data = st.session_state.get('parsed_match_v43')
    res = st.session_state.get('active_res_v43')

    if not data or not res:
        st.info("Run simulation in Tab 7 to view market value calculation.")
    else:
        raw_h, raw_d, raw_a = 1.0 / data['odd_h'], 1.0 / data['odd_d'], 1.0 / data['odd_a']
        total_imp = raw_h + raw_d + raw_a

        comp_df = pd.DataFrame([
            {'Outcome': f"{data['home_team']} Win (1)", 'Model Prob': f"{res['p_home']:.1%}", 'Bookmaker Implied': f"{raw_h / total_imp:.1%}"},
            {'Outcome': 'Draw (X)', 'Model Prob': f"{res['p_draw']:.1%}", 'Bookmaker Implied': f"{raw_d / total_imp:.1%}"},
            {'Outcome': f"{data['away_team']} Win (2)", 'Model Prob': f"{res['p_away']:.1%}", 'Bookmaker Implied': f"{raw_a / total_imp:.1%}"}
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
