import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, json, math, os, re, requests
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from collections import Counter

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, 'football_lab_v2.db')
API_BASE = 'https://v3.football.api-sports.io'

st.set_page_config(page_title='Football Match Lab V2', page_icon='⚽', layout='wide')
st.title('⚽ Football Match Lab — Version 2')
st.caption('Research-assisted match analysis • persistent prediction tracking • configurable Monte Carlo simulation')

# ---------------- Database ----------------
def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with db() as con:
        con.execute('''CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, match_date TEXT,
            home TEXT NOT NULL, away TEXT NOT NULL, simulations INTEGER,
            p_home REAL, p_draw REAL, p_away REAL, p_over15 REAL, p_over25 REAL,
            p_under25 REAL, p_btts REAL, exp_home REAL, exp_away REAL,
            top_score TEXT, source TEXT DEFAULT 'manual', actual_home INTEGER,
            actual_away INTEGER, settled_at TEXT)''')
        con.execute('''CREATE TABLE IF NOT EXISTS research_cache (
            cache_key TEXT PRIMARY KEY, fetched_at TEXT, payload TEXT)''')

def save_prediction(r):
    with db() as con:
        cur = con.execute('''INSERT INTO predictions
        (created_at,match_date,home,away,simulations,p_home,p_draw,p_away,p_over15,p_over25,p_under25,p_btts,exp_home,exp_away,top_score,source)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (datetime.now().isoformat(timespec='seconds'),r['match_date'],r['home'],r['away'],r['simulations'],r['p_home'],r['p_draw'],r['p_away'],r['p_over15'],r['p_over25'],r['p_under25'],r['p_btts'],r['exp_home'],r['exp_away'],r['top_score'],r.get('source','manual')))
        return cur.lastrowid

def get_predictions():
    with db() as con:
        return pd.read_sql_query('SELECT * FROM predictions ORDER BY id DESC', con)

def settle_prediction(pid, hg, ag):
    with db() as con:
        con.execute('UPDATE predictions SET actual_home=?, actual_away=?, settled_at=? WHERE id=?',
                    (int(hg),int(ag),datetime.now().isoformat(timespec='seconds'),int(pid)))

init_db()

# ---------------- Model ----------------
def clamp(x, lo, hi): return max(lo, min(hi, x))
def safe_float(x, default=0.0):
    try:
        v=float(x)
        return v if np.isfinite(v) else default
    except Exception: return default

def form_strength(form):
    pts={'W':1.0,'D':0.5,'L':0.0}
    vals=[pts[c] for c in str(form).upper() if c in pts][:10]
    return float(np.mean(vals)) if vals else .5

def simulate(home_rate, away_rate, n=50000, seed=20261111, home_adv=1.08, uncertainty=.18):
    rng=np.random.default_rng(int(seed))
    hr=np.maximum(.05,rng.lognormal(np.log(max(.05,home_rate)),uncertainty,int(n))*home_adv)
    ar=np.maximum(.05,rng.lognormal(np.log(max(.05,away_rate)),uncertainty,int(n)))
    hg=rng.poisson(hr); ag=rng.poisson(ar)
    return hg,ag,hg+ag

def poisson_pmf(lam, k): return math.exp(-lam + k*math.log(lam) - math.lgamma(k+1)) if lam>0 else (1.0 if k==0 else 0.0)

def model_probabilities(hg, ag):
    n=len(hg); total=hg+ag
    scores=pd.DataFrame({'home':hg,'away':ag}).value_counts().rename('count').reset_index()
    scores['probability']=scores['count']/n
    return {
      'p_home':float(np.mean(hg>ag)), 'p_draw':float(np.mean(hg==ag)), 'p_away':float(np.mean(hg<ag)),
      'p_over15':float(np.mean(total>=2)), 'p_over25':float(np.mean(total>=3)),
      'p_under25':float(np.mean(total<=2)), 'p_btts':float(np.mean((hg>0)&(ag>0))),
      'exp_home':float(np.mean(hg)), 'exp_away':float(np.mean(ag)),
      'scores':scores.sort_values('probability',ascending=False).head(8)
    }

# ---------------- API-Football ----------------
def api_get(path, params, key):
    if not key: raise RuntimeError('Add your API-Football key in Settings first.')
    resp=requests.get(f'{API_BASE}/{path.lstrip("/")}',headers={'x-apisports-key':key},params=params,timeout=25)
    resp.raise_for_status()
    body=resp.json()
    if body.get('errors'):
        raise RuntimeError(str(body['errors']))
    return body.get('response',[]), body.get('results',0), body.get('paging',{})

def resolve_team(name,key):
    rows,_,_=api_get('teams',{'search':name},key)
    if not rows: return None
    target=re.sub(r'[^a-z0-9]','',name.lower())
    rows.sort(key=lambda x:(re.sub(r'[^a-z0-9]','',x['team']['name'].lower())!=target, x['team']['name']))
    return rows[0]

def get_recent_fixtures(team_id,season,key,venue=None,last=10):
    params={'team':team_id,'last':last,'season':season}
    if venue in ('home','away'): params['venue']=venue
    rows,_,_=api_get('fixtures',params,key)
    out=[]
    for x in rows:
        stt=x.get('fixture',{}).get('status',{}).get('short')
        if stt not in ('FT','AET','PEN'): continue
        is_home=x.get('teams',{}).get('home',{}).get('id')==team_id
        gf=x.get('goals',{}).get('home' if is_home else 'away')
        ga=x.get('goals',{}).get('away' if is_home else 'home')
        if gf is None or ga is None: continue
        out.append({'date':x['fixture']['date'][:10], 'opponent':x['teams']['away' if is_home else 'home']['name'],
                    'venue':'H' if is_home else 'A','gf':int(gf),'ga':int(ga),
                    'result':'W' if gf>ga else 'D' if gf==ga else 'L','fixture_id':x['fixture']['id']})
    return out

def get_h2h(home_id,away_id,key,last=10):
    rows,_,_=api_get('fixtures',{'h2h':f'{home_id}-{away_id}','last':last},key)
    out=[]
    for x in rows:
        if x.get('fixture',{}).get('status',{}).get('short') not in ('FT','AET','PEN'): continue
        out.append({'date':x['fixture']['date'][:10], 'home':x['teams']['home']['name'], 'away':x['teams']['away']['name'],
                    'home_goals':x['goals']['home'],'away_goals':x['goals']['away'],
                    'winner':x.get('teams',{}).get('home',{}).get('winner')})
    return out

def find_next_fixture(home_id,away_id,key,season):
    rows,_,_=api_get('fixtures',{'team':home_id,'next':20,'season':season},key)
    for x in rows:
        if {x['teams']['home']['id'],x['teams']['away']['id']}=={home_id,away_id}:
            return x
    return None

def get_injuries(team_id,season,key,fixture_id=None):
    params={'team':team_id,'season':season}
    if fixture_id: params['fixture']=fixture_id
    rows,_,_=api_get('injuries',params,key)
    out=[]
    for x in rows:
        p=x.get('player',{}); fx=x.get('fixture',{})
        out.append({'player':p.get('name','Unknown'),'type':p.get('type',''),'reason':p.get('reason','Not specified'),
                    'fixture_id':fx.get('id'),'date':fx.get('date','')[:10] if fx.get('date') else ''})
    return out

def get_lineups(fixture_id,key):
    rows,_,_=api_get('fixtures/lineups',{'fixture':fixture_id},key)
    out=[]
    for x in rows:
        team=x.get('team',{})
        out.append({'team':team.get('name','Unknown'),'formation':x.get('formation') or 'Not supplied',
                    'coach':(x.get('coach') or {}).get('name','Not supplied'),
                    'startXI':[p.get('player',{}).get('name') for p in x.get('startXI',[])],
                    'substitutes':[p.get('player',{}).get('name') for p in x.get('substitutes',[])]})
    return out

def get_coach(team_id,key):
    rows,_,_=api_get('coachs',{'team':team_id},key)
    return rows[0] if rows else None

def get_news(query):
    url='https://news.google.com/rss/search'
    r=requests.get(url,params={'q':query+' football injury lineup team news','hl':'en','gl':'GB','ceid':'GB:en'},timeout=18)
    r.raise_for_status()
    import xml.etree.ElementTree as ET
    root=ET.fromstring(r.content); out=[]
    for item in root.findall('.//item')[:8]:
        out.append({'title':item.findtext('title',''),'link':item.findtext('link',''),'date':item.findtext('pubDate','')})
    return out

def tactical_summary(team_id,season,key):
    coach=get_coach(team_id,key)
    recent=get_recent_fixtures(team_id,season,key,last=5)
    formations=[]; observations=[]
    # Historical lineup and match-stat endpoints are optional and may be restricted by API plan.
    for match in recent[:3]:
        try:
            lineup_rows,_,_=api_get('fixtures/lineups',{'fixture':match['fixture_id']},key)
            for item in lineup_rows:
                if item.get('team',{}).get('id')==team_id and item.get('formation'):
                    formations.append(item['formation'])
        except Exception: pass
        try:
            stat_rows,_,_=api_get('fixtures/statistics',{'fixture':match['fixture_id']},key)
            for item in stat_rows:
                if item.get('team',{}).get('id')==team_id:
                    vals={x.get('type'):x.get('value') for x in item.get('statistics',[])}
                    observations.append(vals)
        except Exception: pass
    def numeric_average(key_name):
        vals=[]
        for row in observations:
            value=row.get(key_name)
            if isinstance(value,str) and value.endswith('%'):
                try: value=float(value[:-1])
                except Exception: value=None
            if isinstance(value,(int,float)): vals.append(float(value))
        return round(float(np.mean(vals)),1) if vals else None
    possession=numeric_average('Ball Possession')
    shots=numeric_average('Total Shots')
    shots_on=numeric_average('Shots on Goal')
    passes=numeric_average('Total passes')
    style=[]
    if possession is not None:
        style.append('possession-oriented sample' if possession>=55 else 'lower-possession sample' if possession<=45 else 'mid-range possession sample')
    if shots is not None: style.append(('higher' if shots>=13 else 'lower' if shots<=8 else 'moderate')+' shot volume in sample')
    if shots_on is not None: style.append(f'{shots_on:.1f} shots on target per match in sample')
    return {'coach':(coach or {}).get('name','Not returned by provider'),
            'nationality':(coach or {}).get('nationality',''), 'formations':formations,
            'possession':possession,'shots':shots,'shots_on_target':shots_on,'passes':passes,
            'observations':len(observations),'style':style,
            'note':'Style descriptors are simple summaries of retrieved recent-match statistics, not proof of tactical intent. Small samples and provider coverage can distort them.'}

# ---------------- Metrics ----------------
def score_metrics(df):
    settled=df.dropna(subset=['actual_home','actual_away']).copy()
    if settled.empty: return settled, None
    actual=np.where(settled.actual_home>settled.actual_away,0,np.where(settled.actual_home==settled.actual_away,1,2))
    probs=np.clip(settled[['p_home','p_draw','p_away']].to_numpy(dtype=float),1e-9,1)
    onehot=np.eye(3)[actual]
    brier=float(np.mean(np.sum((probs-onehot)**2,axis=1)))
    logloss=float(-np.mean(np.log(probs[np.arange(len(actual)),actual])))
    pred=np.argmax(probs,axis=1)
    summary={'matches':len(settled),'1x2_accuracy':float(np.mean(pred==actual)), 'brier':brier,'logloss':logloss,
             'over25_accuracy':float(np.mean((settled.p_over25>=.5)==((settled.actual_home+settled.actual_away)>=3))),
             'btts_accuracy':float(np.mean((settled.p_btts>=.5)==((settled.actual_home>0)&(settled.actual_away>0))))}
    return settled,summary

# ---------------- Sidebar and tabs ----------------
with st.sidebar:
    st.header('Settings')
    sims=st.select_slider('Simulations per match',options=[10000,25000,50000,75000,100000],value=50000)
    seed=st.number_input('Random seed',min_value=1,max_value=9999999,value=20261111)
    uncertainty=st.slider('Rate uncertainty',.05,.45,.18,.01)
    home_adv=st.slider('Home advantage multiplier',1.0,1.25,1.08,.01)
    st.divider(); st.subheader('Data provider')
    api_key=st.text_input('API-Football API key',type='password',help='Stored only in this running session. Requests go to API-Sports.')
    season=st.number_input('Season start year',min_value=2015,max_value=2035,value=2026)
    league_id=st.number_input('League ID (optional)',min_value=0,value=0)
    st.caption('Coverage and endpoint access depend on your API-Football plan.')

tabs=st.tabs(['Match analysis','Research centre','Accuracy tracker','Batch analysis','Data & help'])

with tabs[0]:
    st.subheader('Build a match prediction')
    a,b=st.columns(2)
    with a:
        home=st.text_input('Home team',placeholder='e.g. Arsenal',key='v2_home')
        h_form=st.text_input('Recent form (newest first)',placeholder='WWDLW')
        h_for=st.number_input('Home goals scored per match',0.0,5.0,1.65,.05)
        h_against=st.number_input('Home goals conceded per match',0.0,5.0,1.05,.05)
        h_xg=st.number_input('Home xG per match (optional)',0.0,5.0,0.0,.05)
        h_missing=st.number_input('Home estimated key-player impact (goals)',0.0,1.5,0.0,.05)
        h_notes=st.text_area('Home lineup / absences / tactical notes',height=90)
    with b:
        away=st.text_input('Away team',placeholder='e.g. Chelsea',key='v2_away')
        a_form=st.text_input('Recent form (newest first)',placeholder='DLWWD')
        a_for=st.number_input('Away goals scored per match',0.0,5.0,1.35,.05)
        a_against=st.number_input('Away goals conceded per match',0.0,5.0,1.25,.05)
        a_xg=st.number_input('Away xG per match (optional)',0.0,5.0,0.0,.05)
        a_missing=st.number_input('Away estimated key-player impact (goals)',0.0,1.5,0.0,.05)
        a_notes=st.text_area('Away lineup / absences / tactical notes',height=90)
    c1,c2=st.columns(2)
    with c1: match_day=st.date_input('Match date',value=date.today()+timedelta(days=1))
    with c2: fixture_id=st.number_input('Fixture ID (optional; for lineups/injuries)',min_value=0,value=0)
    upload=st.file_uploader('Upload screenshot or JSON/CSV reference',type=['png','jpg','jpeg','webp','json','csv'],key='v2_single_upload')
    if upload and upload.type.startswith('image/'): st.image(upload,caption='Reference only — verify names and status against official sources.',width=500)
    if upload and upload.name.lower().endswith(('.json','.csv')): st.info('File attached. Review and enter relevant values; this build does not claim automatic screenshot OCR.')
    st.markdown('**Automatic research**')
    auto_research=st.checkbox('Fetch recent form, injuries, H2H, lineups and manager information when available',value=False)
    if st.button('Fetch match research',use_container_width=True,disabled=not auto_research):
        if not api_key or not home.strip() or not away.strip(): st.error('Enter both teams and an API-Football key in Settings.')
        else:
            try:
                ht=resolve_team(home,api_key); at=resolve_team(away,api_key)
                if not ht or not at: st.error('Could not resolve both teams. Try their official full names.')
                else:
                    hid,aid=ht['team']['id'],at['team']['id']
                    st.session_state['research_v2']={'home_team':ht['team'],'away_team':at['team'],'home_id':hid,'away_id':aid}
                    with st.spinner('Retrieving fixtures, injuries, H2H and lineups…'):
                        hf=get_recent_fixtures(hid,int(season),api_key,'home',10)
                        af=get_recent_fixtures(aid,int(season),api_key,'away',10)
                        hh=get_h2h(hid,aid,api_key,10)
                        nxt=find_next_fixture(hid,aid,api_key,int(season))
                        fid=int(fixture_id) or (nxt or {}).get('fixture',{}).get('id')
                        hi=get_injuries(hid,int(season),api_key,fid or None)
                        ai=get_injuries(aid,int(season),api_key,fid or None)
                        lu=get_lineups(fid,api_key) if fid else []
                        hc=tactical_summary(hid,int(season),api_key)
                        ac=tactical_summary(aid,int(season),api_key)
                    st.session_state['research_v2'].update({'home_form':hf,'away_form':af,'h2h':hh,'next_fixture':nxt,'fixture_id':fid,'home_injuries':hi,'away_injuries':ai,'lineups':lu,'home_tactics':hc,'away_tactics':ac})
                    if hf:
                        st.session_state['auto_home_form']=''.join(x['result'] for x in hf[:10])
                        st.session_state['auto_home_for']=float(np.mean([x['gf'] for x in hf]))
                        st.session_state['auto_home_against']=float(np.mean([x['ga'] for x in hf]))
                    if af:
                        st.session_state['auto_away_form']=''.join(x['result'] for x in af[:10])
                        st.session_state['auto_away_for']=float(np.mean([x['gf'] for x in af]))
                        st.session_state['auto_away_against']=float(np.mean([x['ga'] for x in af]))
                    st.success(f'Research retrieved. Fixture ID: {fid or "not found"}. Review the Research centre tab.')
            except Exception as e: st.error(f'Research request failed: {e}')
    research=st.session_state.get('research_v2')
    if research:
        st.caption(f"Resolved: {research['home_team']['name']} vs {research['away_team']['name']} • fixture {research.get('fixture_id') or 'not identified'}")
        st.caption('When recent-form fields are left blank, the prediction automatically uses the retrieved home/away form and goal averages.')
    if st.button('Run and save prediction',type='primary',use_container_width=True):
        if not home.strip() or not away.strip(): st.error('Enter both team names.')
        else:
            rr=st.session_state.get('research_v2',{})
            hf=h_form or ''.join(x['result'] for x in rr.get('home_form',[]))
            af=a_form or ''.join(x['result'] for x in rr.get('away_form',[]))
            hgf=safe_float(h_for); hag=safe_float(h_against); agf=safe_float(a_for); aag=safe_float(a_against)
            if rr.get('home_form') and not h_form:
                hgf=float(np.mean([x['gf'] for x in rr['home_form']])); hag=float(np.mean([x['ga'] for x in rr['home_form']]))
            if rr.get('away_form') and not a_form:
                agf=float(np.mean([x['gf'] for x in rr['away_form']])); aag=float(np.mean([x['ga'] for x in rr['away_form']]))
            # Blend attack with opponent defence; xG takes a partial role when provided.
            hl=.62*hgf+.38*(aag if aag>0 else 1.15); al=.62*agf+.38*(hag if hag>0 else 1.15)
            if h_xg>0: hl=.55*hl+.45*h_xg
            if a_xg>0: al=.55*al+.45*a_xg
            hl*=math.exp((form_strength(hf)-.5)*.28); al*=math.exp((form_strength(af)-.5)*.28)
            hl=max(.12,hl-safe_float(h_missing)); al=max(.12,al-safe_float(a_missing))
            gh,ga,tot=simulate(hl,al,sims,seed,home_adv,uncertainty); p=model_probabilities(gh,ga)
            score=p['scores'].iloc[0]
            record={'match_date':str(match_day),'home':home.strip(),'away':away.strip(),'simulations':int(sims),**{k:p[k] for k in ['p_home','p_draw','p_away','p_over15','p_over25','p_under25','p_btts','exp_home','exp_away']},'top_score':f"{int(score['home'])}-{int(score['away'])}",'source':'API research' if rr else 'manual'}
            pid=save_prediction(record); st.session_state['last_prediction_v2']={'id':pid,**record}
            st.success(f'Prediction #{pid} saved to your local database.')
            st.subheader(f"{home} vs {away}")
            k=st.columns(3)
            k[0].metric('Home win',f"{p['p_home']:.1%}"); k[1].metric('Draw',f"{p['p_draw']:.1%}"); k[2].metric('Away win',f"{p['p_away']:.1%}")
            k=st.columns(4)
            k[0].metric('Over 1.5',f"{p['p_over15']:.1%}"); k[1].metric('Over 2.5',f"{p['p_over25']:.1%}"); k[2].metric('Under 2.5',f"{p['p_under25']:.1%}"); k[3].metric('BTTS',f"{p['p_btts']:.1%}")
            st.write(f"Expected goals: **{home} {p['exp_home']:.2f} — {p['exp_away']:.2f} {away}**")
            disp=p['scores'].copy(); disp['score']=disp.apply(lambda x:f"{int(x.home)}-{int(x.away)}",axis=1)
            st.dataframe(disp[['score','probability']].style.format({'probability':'{:.2%}'}),hide_index=True,use_container_width=True)
            st.caption('Probabilities are estimates from the supplied/retrieved inputs. A large simulation count reduces simulation noise; it does not guarantee model accuracy.')

with tabs[1]:
    st.subheader('Research centre')
    rr=st.session_state.get('research_v2')
    if not rr: st.info('Open Match analysis, enter both team names, enable automatic research and select Fetch match research.')
    else:
        st.write(f"**{rr['home_team']['name']}** vs **{rr['away_team']['name']}**")
        st.caption(f"Provider fixture ID: {rr.get('fixture_id') or 'No matching upcoming fixture found'}")
        t1,t2,t3,t4,t5=st.tabs(['Recent form','Injuries','Lineups','H2H','Managers & tactics'])
        with t1:
            for label,key in [('Home team','home_form'),('Away team','away_form')]:
                rows=rr.get(key,[]); st.markdown(f'**{label}**')
                if rows: st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
                else: st.warning('No recent fixture data returned.')
        with t2:
            st.warning('Injury feeds may include reports linked to fixtures and can be incomplete. Verify important absences with the club or competition before using them.')
            for label,key in [('Home','home_injuries'),('Away','away_injuries')]:
                st.markdown(f'**{label}**')
                rows=rr.get(key,[])
                if rows: st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
                else: st.write('No injury records returned by the provider for this query. This does not prove the squad is fully fit.')
        with t3:
            fid=rr.get('fixture_id')
            if rr.get('lineups'):
                for x in rr['lineups']:
                    st.markdown(f"**{x['team']} — {x['formation']}** · Coach: {x['coach']}")
                    st.write('Starting XI: '+', '.join(x['startXI']))
                    st.write('Substitutes: '+', '.join(x['substitutes']))
            else: st.info('No lineup returned. Confirmed lineups are usually published shortly before kick-off; no predicted XI is labelled as confirmed here.')
        with t4:
            rows=rr.get('h2h',[])
            if rows:
                df=pd.DataFrame(rows); st.dataframe(df,hide_index=True,use_container_width=True)
                hw=sum(1 for x in rows if x['home_goals']>x['away_goals'] and x['home'].lower()==rr['home_team']['name'].lower())
                dr=sum(1 for x in rows if x['home_goals']==x['away_goals'])
                st.write(f'Retrieved meetings: **{len(rows)}**. H2H results are shown from each match’s listed home/away perspective.')
            else: st.info('No H2H records returned.')
        with t5:
            for label,key in [('Home','home_tactics'),('Away','away_tactics')]:
                x=rr.get(key,{})
                st.markdown(f"**{label} manager:** {x.get('coach','Not returned')}")
                if x.get('nationality'): st.write('Nationality:',x['nationality'])
                st.write(x.get('note',''))
                st.write('Observed formations:',', '.join(x.get('formations',[])) if x.get('formations') else 'Not available from returned lineup data')
            st.caption('Version 2 reports manager identity and observed formation data when the provider supplies them. It does not claim to infer a manager’s full tactical philosophy from incomplete data.')
        try:
            news=get_news(rr['home_team']['name']+' '+rr['away_team']['name'])
            with st.expander('Recent news headlines (discovery only)'):
                for n in news: st.markdown(f"- [{n['title']}]({n['link']}) — {n['date']}")
        except Exception as e: st.caption(f'News feed unavailable: {e}')

with tabs[2]:
    st.subheader('Historical accuracy tracker')
    hist=get_predictions()
    if hist.empty: st.info('Saved predictions will appear here. They persist between app restarts on this computer.')
    else:
        settled,metrics=score_metrics(hist)
        if metrics:
            cols=st.columns(5)
            cols[0].metric('Settled matches',metrics['matches']); cols[1].metric('1X2 accuracy',f"{metrics['1x2_accuracy']:.1%}")
            cols[2].metric('Brier score',f"{metrics['brier']:.3f}",help='Lower is better; multiclass 1X2 Brier score.')
            cols[3].metric('Log loss',f"{metrics['logloss']:.3f}",help='Lower is better.')
            cols[4].metric('Over 2.5 accuracy',f"{metrics['over25_accuracy']:.1%}")
            st.metric('BTTS accuracy',f"{metrics['btts_accuracy']:.1%}")
        else: st.info('No settled matches yet. Enter actual scores below to start measuring performance.')
        st.markdown('**Record actual match result**')
        pending=hist[hist.actual_home.isna()]
        if not pending.empty:
            labels={int(r.id):f"#{int(r.id)} · {r.home} vs {r.away} · {r.match_date}" for r in pending.itertuples()}
            with st.form('settle_form'):
                selected=st.selectbox('Prediction',list(labels.keys()),format_func=lambda x:labels[x])
                x,y=st.columns(2)
                actual_h=x.number_input('Actual home goals',min_value=0,max_value=30,value=0)
                actual_a=y.number_input('Actual away goals',min_value=0,max_value=30,value=0)
                submit=st.form_submit_button('Save actual result')
                if submit:
                    settle_prediction(selected,actual_h,actual_a); st.success('Result saved. Metrics updated.'); st.rerun()
        else: st.success('All saved predictions have actual results recorded.')
        st.markdown('**Prediction database**')
        view=hist.copy()
        view['settled']=view.actual_home.notna()
        st.dataframe(view[['id','created_at','match_date','home','away','p_home','p_draw','p_away','p_over25','p_btts','actual_home','actual_away','settled']],hide_index=True,use_container_width=True)
        st.download_button('Export full prediction history CSV',hist.to_csv(index=False).encode('utf-8'),file_name='football_match_lab_v2_history.csv',mime='text/csv')
        if metrics and len(settled)>=10:
            st.markdown('**Calibration by maximum 1X2 probability**')
            cal=settled.copy(); probs=cal[['p_home','p_draw','p_away']].to_numpy(dtype=float)
            actual_class=np.where(cal.actual_home>cal.actual_away,0,np.where(cal.actual_home==cal.actual_away,1,2))
            predicted_class=np.argmax(probs,axis=1)
            cal['confidence']=np.max(probs,axis=1); cal['correct']=(predicted_class==actual_class).astype(int)
            cal['band']=pd.cut(cal.confidence,bins=[0,.4,.5,.6,.7,.8,1.001],include_lowest=True)
            st.dataframe(cal.groupby('band',observed=False).agg(matches=('id','count'),mean_confidence=('confidence','mean'),realized_accuracy=('correct','mean')).reset_index(),hide_index=True,use_container_width=True)
        elif metrics: st.caption('Calibration bands are hidden until at least 10 settled matches are available.')

with tabs[3]:
    st.subheader('Batch prediction')
    st.write('Upload one row per fixture. Supported columns: home, away, home_for, home_against, away_for, away_against, home_form, away_form.')
    st.code('home,away,home_for,home_against,away_for,away_against,home_form,away_form\nArsenal,Chelsea,1.8,1.0,1.4,1.3,WWDLW,DLWWD',language='csv')
    file=st.file_uploader('Upload CSV or JSON',type=['csv','json'],key='batch_v2')
    if file and st.button('Run batch simulations'):
        try:
            if file.name.lower().endswith('.csv'): rows=pd.read_csv(file).to_dict('records')
            else:
                obj=json.load(file); rows=obj if isinstance(obj,list) else obj.get('matches',[])
            out=[]
            for i,r in enumerate(rows):
                hn=str(r.get('home','Home')); an=str(r.get('away','Away'))
                hf=safe_float(r.get('home_for'),1.4); ha=safe_float(r.get('home_against'),1.2)
                af=safe_float(r.get('away_for'),1.2); aa=safe_float(r.get('away_against'),1.3)
                lh=max(.12,.62*hf+.38*aa)*math.exp((form_strength(r.get('home_form',''))-.5)*.28)
                la=max(.12,.62*af+.38*ha)*math.exp((form_strength(r.get('away_form',''))-.5)*.28)
                gh,ga,tt=simulate(lh,la,sims,int(seed)+i,home_adv,uncertainty); p=model_probabilities(gh,ga)
                row={'match':f'{hn} vs {an}',**{k:p[k] for k in ['p_home','p_draw','p_away','p_over15','p_over25','p_under25','p_btts','exp_home','exp_away']}}
                out.append(row)
                save_prediction({'match_date':str(date.today()),'home':hn,'away':an,'simulations':int(sims),**{k:p[k] for k in ['p_home','p_draw','p_away','p_over15','p_over25','p_under25','p_btts','exp_home','exp_away']},'top_score':'batch','source':'batch'})
            result=pd.DataFrame(out); st.dataframe(result.style.format({c:'{:.1%}' for c in result.columns if c not in ['match','exp_home','exp_away']}).format({'exp_home':'{:.2f}','exp_away':'{:.2f}'}),use_container_width=True)
            st.download_button('Download batch results',result.to_csv(index=False).encode(),file_name='football_match_lab_v2_batch.csv',mime='text/csv')
            st.success(f'{len(out)} predictions saved to the history database.')
        except Exception as e: st.error(f'Batch file error: {e}')

with tabs[4]:
    st.subheader('Version 2 status and setup')
    status=pd.DataFrame([
      ('50,000+ Monte Carlo simulations','Working','Configurable from 10,000 to 100,000'),
      ('Manual match inputs','Working','Goals, form, xG, player impact and notes'),
      ('Screenshot uploads','Reference only','Image display; no automatic OCR or verification'),
      ('Automatic recent form','API required','Last 10 home/away fixtures where available'),
      ('Automatic injury verification','API required','Provider injury records; must cross-check official sources'),
      ('Confirmed lineup analysis','API required','Fixture lineup endpoint; only when provider publishes XI'),
      ('H2H automatic collection','API required','Previous meetings from provider'),
      ('Manager tactical analysis','Partial / API required','Manager identity and observed historical formations; not full tactical scouting'),
      ('Historical accuracy tracking','Working','Local SQLite database; manual score settlement; Brier and log loss'),
    ],columns=['Feature','Status','Details'])
    st.dataframe(status,hide_index=True,use_container_width=True)
    st.markdown('**Install on Windows**')
    st.code('''py -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py''',language='powershell')
    st.markdown('**Important data notes**')
    st.write('- Add an API-Football key in the sidebar for structured research. The app sends API requests only to API-Sports.')
    st.write('- API availability, competitions, historical coverage and rate limits depend on your subscription.')
    st.write('- An empty injury response does not establish that every player is fit. Confirmed lineups appear only when published and supplied by the provider.')
    st.write('- Manager identity and formations are not the same as deep tactical analysis. The app reports observed information and labels missing evidence.')
    st.write('- Predictions and actual scores are stored in `football_lab_v2.db` beside the app. Back up this file. Do not delete it if you want to keep your history.')
    st.write('- No bookmaker connection, auto-betting or guaranteed-win claims.')
    st.download_button('Download sample batch CSV',pd.DataFrame([{'home':'Arsenal','away':'Chelsea','home_for':1.8,'home_against':1.0,'away_for':1.4,'away_against':1.3,'home_form':'WWDLW','away_form':'DLWWD'}]).to_csv(index=False).encode(),file_name='sample_matches_v2.csv',mime='text/csv')
