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
DB_PATH = os.path.join(APP_DIR, 'football_lab_v4.db')

st.set_page_config(
    page_title='Football Match Lab V4.2',
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

init_db()

# ---------------- Dixon-Coles & Math Engine ----------------
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

# ---------------- Text Parsing Helper ----------------
def parse_match_text_block(text):
    data = {
        'home_team': 'Home Team', 'away_team': 'Away Team',
        'home_gf': 1.5, 'home_ga': 1.0, 'home_elo': 1500.0,
        'away_gf': 1.5, 'away_ga': 1.0, 'away_elo': 1500.0,
        'home_formation': '4-3-3', 'away_formation': '4-3-3',
        'home_absences': 0, 'away_absences': 0,
        'odd_h': 2.50, 'odd_d': 3.20, 'odd_a': 2.80,
        'raw_text': text
    }
    
    lines = text.split('\n')
    for line in lines:
        l = line.lower()
        if 'home team' in l or 'match:' in l or 'vs' in l:
            m = re.search(r'([a-zA-Z\s]+)\s+vs\s+([a-zA-Z\s]+)', line, re.I)
            if m:
                data['home_team'] = m.group(1).strip()
                data['away_team'] = m.group(2).strip()
        
        # Parse Goals Scored & Conceded
        if 'italy avg goals scored' in l or 'home avg goals scored' in l or ('home' in l and 'gf' in l):
            val = re.findall(r'\d+\.\d+|\d+', line)
            if val: data['home_gf'] = float(val[0])
        if 'italy avg goals conceded' in l or 'home avg goals conceded' in l or ('home' in l and 'ga' in l):
            val = re.findall(r'\d+\.\d+|\d+', line)
            if val: data['home_ga'] = float(val[0])
        if 'france avg goals scored' in l or 'away avg goals scored' in l or ('away' in l and 'gf' in l):
            val = re.findall(r'\d+\.\d+|\d+', line)
            if val: data['away_gf'] = float(val[0])
        if 'france avg goals conceded' in l or 'away avg goals conceded' in l or ('away' in l and 'ga' in l):
            val = re.findall(r'\d+\.\d+|\d+', line)
            if val: data['away_ga'] = float(val[0])
            
        # Parse Elo Ratings
        if 'home' in l and 'elo' in l:
            val = re.findall(r'\d{4}', line)
            if val: data['home_elo'] = float(val[0])
        if 'away' in l and 'elo' in l:
            val = re.findall(r'\d{4}', line)
            if val: data['away_elo'] = float(val[0])
            
        # Parse Odds
        if 'odds (1' in l or 'home odds' in l:
            val = re.findall(r'\d+\.\d+', line)
            if val: data['odd_h'] = float(val[0])
        if 'odds (x' in l or 'draw odds' in l:
            val = re.findall(r'\d+\.\d+', line)
            if val: data['odd_d'] = float(val[0])
        if 'odds (2' in l or 'away odds' in l:
            val = re.findall(r'\d+\.\d+', line)
            if val: data['odd_a'] = float(val[0])
            
    return data

# ---------------- Sidebar ----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/football-ball-1.png", width=64)
    st.title("Match Lab V4.2")
    sim_count = st.select_slider("Simulation Runs", options=[10000, 50000, 100000], value=50000)
    home_adv_val = st.slider("Home Advantage Multiplier", 1.00, 1.25, 1.08, 0.01)
    dixon_coles_rho = st.slider("Dixon-Coles Rho", -0.30, 0.00, -0.13, 0.01)

# ---------------- UI & Tabs ----------------
st.title("⚽ Football Match Lab V4.2 — Single-Block Data Input Engine")

tabs = st.tabs([
    "📋 Single-Block Input",
    "🛡️ Loaded Data Overview",
    "🎲 Prediction Engine",
    "📊 Market Odds Check"
])

# ---------------- TAB 0: SINGLE-BLOCK DATA PASTE ----------------
with tabs[0]:
    st.subheader("Paste Match Information Block")
    st.write("Paste your raw match details below at once. The engine will parse the stats for simulation:")
    
    default_text = """Match: Italy vs France
Italy Avg Goals Scored (GF): 1.67
Italy Avg Goals Conceded (GA): 1.15
Italy Team Elo Rating: 1840

France Avg Goals Scored (GF): 2.10
France Avg Goals Conceded (GA): 0.75
France Team Elo Rating: 1910

Bookmaker Home Odds (1 - Italy): 2.75
Bookmaker Draw Odds (X): 3.20
Bookmaker Away Odds (2 - France): 2.45"""

    raw_input = st.text_area("Match Block Text", value=default_text, height=300)
    
    if st.button("🚀 Parse & Load Match Data into Engine", type="primary", use_container_width=True):
        parsed = parse_match_text_block(raw_input)
        st.session_state['parsed_match_data'] = parsed
        st.success(f"Successfully loaded data for **{parsed['home_team']} vs {parsed['away_team']}**!")

# ---------------- TAB 1: LOADED DATA OVERVIEW ----------------
with tabs[1]:
    st.subheader("Loaded Parameters Summary")
    data = st.session_state.get('parsed_match_data')
    if not data:
        st.warning("Please paste and parse a match block in Tab 0 first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### 🏠 {data['home_team']}")
            st.write(f"**Avg Goals Scored:** {data['home_gf']}")
            st.write(f"**Avg Goals Conceded:** {data['home_ga']}")
            st.write(f"**Elo Rating:** {data['home_elo']}")
        with col2:
            st.markdown(f"### ✈️ {data['away_team']}")
            st.write(f"**Avg Goals Scored:** {data['away_gf']}")
            st.write(f"**Avg Goals Conceded:** {data['away_ga']}")
            st.write(f"**Elo Rating:** {data['away_elo']}")

# ---------------- TAB 2: PREDICTION ENGINE ----------------
with tabs[2]:
    st.subheader("Run Dixon-Coles Monte Carlo Engine")
    data = st.session_state.get('parsed_match_data')
    
    if not data:
        st.error("Please paste and parse match data in Tab 0 to run simulation.")
    else:
        if st.button("⚡ Run Simulation on Parsed Data", type="primary", use_container_width=True):
            # Calculate Elo Multipliers
            elo_diff = data['home_elo'] - data['away_elo']
            adj_h_mult = max(0.5, min(2.0, 10.0 ** (elo_diff / 600.0)))
            adj_a_mult = max(0.5, min(2.0, 10.0 ** (-elo_diff / 600.0)))
            
            # Expected Goals ($\lambda$ and $\mu$)
            lambda_h = (0.6 * data['home_gf'] + 0.4 * data['away_ga']) * adj_h_mult
            mu_a = (0.6 * data['away_gf'] + 0.4 * data['home_ga']) * adj_a_mult
            
            hg, ag = simulate_dixon_coles_monte_carlo(
                lambda_h, mu_a, n=sim_count, seed=20261112, rho=dixon_coles_rho, home_adv=home_adv_val
            )
            
            res = model_probabilities_from_samples(hg, ag)
            st.session_state['active_res'] = res
            
            st.success("Simulation Complete!")
            
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{data['home_team']} Win", f"{res['p_home']:.1%}")
            c2.metric("Draw", f"{res['p_draw']:.1%}")
            c3.metric(f"{data['away_team']} Win", f"{res['p_away']:.1%}")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Over 1.5", f"{res['p_over15']:.1%}")
            m2.metric("Over 2.5", f"{res['p_over25']:.1%}")
            m3.metric("Over 3.5", f"{res['p_over35']:.1%}")
            m4.metric("BTTS Yes", f"{res['p_btts']:.1%}")
            
            st.markdown(f"**Expected Goals:** {data['home_team']} **{res['exp_home']:.2f}** — **{res['exp_away']:.2f}** {data['away_team']}")

# ---------------- TAB 3: MARKET ODDS CHECK ----------------
with tabs[3]:
    st.subheader("Market Odds & Value Check")
    data = st.session_state.get('parsed_match_data')
    res = st.session_state.get('active_res')
    
    if not data or not res:
        st.info("Run simulation in Tab 2 to view market value calculation.")
    else:
        raw_h, raw_d, raw_a = 1.0 / data['odd_h'], 1.0 / data['odd_d'], 1.0 / data['odd_a']
        total_imp = raw_h + raw_d + raw_a
        
        comp_df = pd.DataFrame([
            {'Outcome': f"{data['home_team']} Win (1)", 'Model Prob': f"{res['p_home']:.1%}", 'Bookmaker Implied': f"{raw_h / total_imp:.1%}"},
            {'Outcome': 'Draw (X)', 'Model Prob': f"{res['p_draw']:.1%}", 'Bookmaker Implied': f"{raw_d / total_imp:.1%}"},
            {'Outcome': f"{data['away_team']} Win (2)", 'Model Prob': f"{res['p_away']:.1%}", 'Bookmaker Implied': f"{raw_a / total_imp:.1%}"}
        ])
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
