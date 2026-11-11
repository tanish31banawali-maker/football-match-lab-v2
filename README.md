# Football Match Lab V2 — Web deployment package

This is a Streamlit web application. It can be hosted on Streamlit Community Cloud.

## Deploy online

1. Create or sign in to a GitHub account.
2. Create a new **public** repository, for example `football-match-lab-v2`.
3. Upload the contents of this folder to the repository root. `app.py` must be at the root (not inside another nested folder).
4. Visit https://share.streamlit.io/ and sign in with GitHub.
5. Choose **Create app**, select the repository and branch, and set **Main file path** to `app.py`.
6. Click **Deploy**. Streamlit will install packages from `requirements.txt`.
7. Open the generated `*.streamlit.app` URL.
8. In the app sidebar, enter your API-Football key when you want automatic data retrieval.

## Important hosting limitations

- This app currently stores prediction history in a local SQLite file (`football_lab_v2.db`). On ephemeral cloud hosting, local files may be lost when the app is restarted, rebuilt, or moved. Export your prediction history CSV regularly and keep a backup.
- The API key is entered in the app sidebar and is held only for the active app session. Do not put private API keys in a public GitHub repository.
- API-Football endpoint access and data coverage depend on your subscription.
- Streamlit Community Cloud is suitable for a personal prototype. For reliable long-term history, connect a managed database (for example Supabase or PostgreSQL) before relying on it as a permanent record system.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
