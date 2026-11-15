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
    Strict parser for sectioned match blocks. Starts every field as
