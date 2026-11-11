# Publish Football Match Lab V2 on the web

## Step 1 — Upload the project to GitHub

1. Sign in at https://github.com/.
2. Create a new repository named `football-match-lab-v2`.
3. For easiest deployment, set it to **Public**.
4. Choose **Add file → Upload files**.
5. Upload `app.py`, `requirements.txt`, `sample_matches.csv`, `.gitignore`, and the `.streamlit` folder from this package.
6. Commit the upload.

Check that `app.py` appears on the repository's main page. It must not be nested inside another project folder.

## Step 2 — Deploy with Streamlit Community Cloud

1. Open https://share.streamlit.io/.
2. Sign in using the same GitHub account.
3. Select **Create app**.
4. Choose repository `football-match-lab-v2`, branch `main`, and main file path `app.py`.
5. Select **Deploy** and wait for the build to finish.
6. Open the generated website URL ending in `.streamlit.app`.

## Step 3 — Enable automatic football data

Open the app's sidebar and enter your API-Football key. Keep the key private; never commit it to GitHub.

## Before using it seriously

The current SQLite prediction history is stored on the app's local filesystem. Cloud restarts/rebuilds can erase it. Export the history CSV regularly. A persistent hosted database is a separate upgrade needed for dependable long-term cloud history.
