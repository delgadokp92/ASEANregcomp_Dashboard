# ARIS — ASEAN Regulatory Information System

A Streamlit web application for exploring and managing financial-sector regulations across the 11 ASEAN member states. It supports public read-only access and authenticated editing by country representatives and administrators.

---

## Contents

1. [Quick start (local development)](#1-quick-start-local-development)
2. [Access levels](#2-access-levels)
3. [Production deployment — Linux](#3-production-deployment--linux)
4. [Production deployment — Windows Server](#4-production-deployment--windows-server)
5. [Enabling Keycloak token authentication](#5-enabling-keycloak-token-authentication)
6. [Data setup](#6-data-setup)
7. [Updating the app](#7-updating-the-app)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Quick start (local development)

### Requirements

- Python 3.10 or higher
- Git (optional — you can also just unzip the source)

### Steps

**Step 1 — Get the code**

```bash
git clone <repo-url> asean_regdash
cd asean_regdash/dev
```

Or unzip the provided archive and `cd` into the `dev/` folder.

**Step 2 — Create a virtual environment**

```bash
python -m venv .venv
```

**Step 3 — Activate the virtual environment**

| Platform | Command |
|---|---|
| Windows (PowerShell) | `.\.venv\Scripts\Activate.ps1` |
| Windows (CMD) | `.venv\Scripts\activate.bat` |
| Linux / macOS | `source .venv/bin/activate` |

> **Windows PowerShell note:** If you get a script execution policy error, run this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**Step 4 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 5 — Run the app**

```bash
streamlit run app.py
```

The browser opens automatically at `http://localhost:8501`. To stop the server, press `Ctrl+C`.

---

## 2. Access levels

The app has three access modes determined by the URL parameters.

### Using legacy URL parameters (development / no Keycloak)

| Role | URL | Capabilities |
|---|---|---|
| Public | `http://localhost:8501/` | Map + Table, read-only |
| Country editor | `http://localhost:8501/?country=Vietnam&user=Alice` | + Editor (locked to Vietnam) + Gap Analysis |
| Admin | `http://localhost:8501/?country=NA&user=Admin` | + Editor (all countries) + full Gap Analysis admin |

### Using dummy tokens (integration testing)

| Role | URL |
|---|---|
| Admin | `http://localhost:8501/?token=dummy-admin` |
| Vietnam editor | `http://localhost:8501/?token=dummy-vietnam` |
| Singapore editor | `http://localhost:8501/?token=dummy-singapore` |

### Using a real Keycloak token (production)

Keycloak redirects users to the dashboard with a JWT:

```
https://dashboard.example.com/?token=<keycloak_jwt>
```

See [§5 Enabling Keycloak token authentication](#5-enabling-keycloak-token-authentication) for setup.

---

## 3. Production deployment — Linux

Tested on **Ubuntu 22.04 LTS**. Adapt paths as needed.

### 3.1 System setup

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv git nginx
sudo useradd -m -s /bin/bash regdash
sudo su - regdash
```

### 3.2 Install the app

```bash
git clone <repo-url> /home/regdash/app
cd /home/regdash/app/dev
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify it starts:

```bash
streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
# Press Ctrl+C after confirming it starts without errors
```

### 3.3 Create a systemd service

Create `/etc/systemd/system/asean-regdash.service`:

```ini
[Unit]
Description=ARIS — ASEAN Regulatory Information System
After=network.target

[Service]
Type=simple
User=regdash
WorkingDirectory=/home/regdash/app/dev
Environment="PATH=/home/regdash/app/dev/.venv/bin"
Environment="KEYCLOAK_JWKS_URL=https://auth.example.com/realms/myrealm/protocol/openid-connect/certs"
Environment="KEYCLOAK_CLIENT_ID=asean-regdash"
ExecStart=/home/regdash/app/dev/.venv/bin/streamlit run app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable asean-regdash
sudo systemctl start asean-regdash
sudo systemctl status asean-regdash
```

### 3.4 Nginx reverse proxy

Create `/etc/nginx/sites-available/asean-regdash`:

```nginx
server {
    listen 80;
    server_name dashboard.example.com;

    # Redirect HTTP to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name dashboard.example.com;

    ssl_certificate     /etc/letsencrypt/live/dashboard.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/dashboard.example.com/privkey.pem;

    location / {
        proxy_pass         http://127.0.0.1:8501;
        proxy_http_version 1.1;

        # WebSocket support (required by Streamlit)
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";

        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;

        proxy_read_timeout 86400;
    }

    # Streamlit static assets
    location /_stcore/static {
        proxy_pass http://127.0.0.1:8501/_stcore/static;
    }
}
```

Enable and reload:

```bash
sudo ln -s /etc/nginx/sites-available/asean-regdash /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 3.5 Firewall

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

Port 8501 does **not** need to be open publicly — Nginx proxies it from 80/443.

### 3.6 SSL certificate (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d dashboard.example.com
```

Certbot auto-renews every 90 days. Verify with `sudo certbot renew --dry-run`.

---

## 4. Production deployment — Windows Server

### 4.1 Prerequisites

- Python 3.10+ — download from [python.org](https://python.org) (tick "Add to PATH" during install)
- Git for Windows (optional)

### 4.2 Install the app

Open PowerShell as Administrator:

```powershell
# Extract the zip or clone
cd C:\inetpub\asean-regdash\dev

# Create venv and install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4.3 Set environment variables

Open **System Properties → Advanced → Environment Variables** and add System variables:

| Name | Value |
|---|---|
| `KEYCLOAK_JWKS_URL` | `https://auth.example.com/realms/myrealm/protocol/openid-connect/certs` |
| `KEYCLOAK_CLIENT_ID` | `asean-regdash` |

Or set them per-service in NSSM (see below).

### 4.4 Run as a Windows Service using NSSM

Download [NSSM](https://nssm.cc/download) and place `nssm.exe` in `C:\Windows\System32\`.

```powershell
nssm install ARIS "C:\inetpub\asean-regdash\dev\.venv\Scripts\streamlit.exe"
nssm set ARIS AppParameters "run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"
nssm set ARIS AppDirectory "C:\inetpub\asean-regdash\dev"
nssm set ARIS DisplayName "ARIS — ASEAN Regulatory Information System"
nssm set ARIS Description "Streamlit dashboard for ASEAN regulatory data"
nssm set ARIS Start SERVICE_AUTO_START
nssm set ARIS AppEnvironmentExtra "KEYCLOAK_JWKS_URL=https://auth.example.com/realms/myrealm/protocol/openid-connect/certs"
nssm set ARIS AppEnvironmentExtra+= "KEYCLOAK_CLIENT_ID=asean-regdash"
nssm start ARIS
```

Manage with `nssm start/stop/restart ARIS` or the Services control panel.

### 4.5 Alternative: Scheduled Task

If NSSM is not available, create a scheduled task that runs at system startup:

```powershell
$action  = New-ScheduledTaskAction -Execute "C:\inetpub\asean-regdash\dev\.venv\Scripts\streamlit.exe" `
    -Argument "run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true" `
    -WorkingDirectory "C:\inetpub\asean-regdash\dev"
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0
Register-ScheduledTask -TaskName "ARIS" -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest
```

### 4.6 Reverse proxy with IIS (optional)

If exposing on port 80/443 instead of 8501 directly:

1. Install IIS + **Application Request Routing (ARR)** module
2. Enable proxy in ARR settings: IIS Manager → server node → Application Request Routing Cache → Server Proxy Settings → Enable proxy
3. Create an inbound URL rewrite rule:
   - Match URL: `(.*)`
   - Conditions: `{SERVER_PORT}` is `80`
   - Action: Rewrite to `http://localhost:8501/{R:1}`
4. Enable WebSocket support: IIS Manager → site → WebSocket Protocol → Enabled

Alternatively, allow port 8501 through Windows Firewall and access it directly:

```powershell
New-NetFirewallRule -DisplayName "ARIS" -Direction Inbound -Protocol TCP -LocalPort 8501 -Action Allow
```

---

## 5. Enabling Keycloak Token Authentication

The app ships with dummy tokens for development. Follow these steps to switch to real Keycloak JWTs.

### 5.1 What Keycloak needs to send

After a user authenticates, Keycloak must redirect them to the dashboard with the access token as a URL query parameter:

```
https://dashboard.example.com/?token=<access_token>
```

Configure this as the **Valid Redirect URI** in your Keycloak client settings, and use a post-login action or client-side redirect to append `?token=<token>` to the URL.

### 5.2 Required JWT claims

| Claim | Type | Description |
|---|---|---|
| `preferred_username` | string | Display name (shown in audit log) |
| `country` | string | ASEAN country name (e.g. `"Viet Nam"`) or `"NA"` for admins |
| `realm_roles` | string[] | Include `"dashboard-admin"` to grant admin access |

`country` is a **custom claim** — you must add it via a Keycloak mapper:

1. Keycloak Admin Console → your realm → **Clients** → `asean-regdash` → **Client Scopes** → `asean-regdash-dedicated`
2. **Add mapper** → By configuration → **User Attribute**
3. Name: `country`, User Attribute: `country`, Token Claim Name: `country`, Claim JSON Type: `String`
4. Set the `country` attribute on each user in **Users → (user) → Attributes**

### 5.3 Activate real JWT validation in app.py

**Step 1 — Ensure python-jose is installed** (already in requirements.txt):

```bash
pip install "python-jose[cryptography]"
```

**Step 2 — Set environment variables** (see §3.3 for Linux, §4.3 for Windows):

```bash
KEYCLOAK_JWKS_URL=https://auth.example.com/realms/myrealm/protocol/openid-connect/certs
KEYCLOAK_CLIENT_ID=asean-regdash
```

**Step 3 — Uncomment the validation block in `app.py`**

Find `parse_auth_token()` (search for `def parse_auth_token`) and uncomment the jose block:

```python
def parse_auth_token(token: str) -> dict:
    import os
    token = token.strip()

    # Remove or comment out dummy token check for production:
    # if token in _DUMMY_TOKENS:
    #     return _DUMMY_TOKENS[token]

    jwks_url  = os.environ.get("KEYCLOAK_JWKS_URL", "")
    client_id = os.environ.get("KEYCLOAK_CLIENT_ID", "")
    if jwks_url and client_id:
        try:
            from jose import jwt as _jwt
            import requests, json as _json
            jwks = requests.get(jwks_url, timeout=5).json()
            claims = _jwt.decode(
                token, jwks, algorithms=["RS256"],
                audience=client_id, options={"verify_at_hash": False},
            )
            return claims
        except Exception:
            return {}

    return {}
```

**Step 4 — Remove dummy tokens** from `_DUMMY_TOKENS` (or leave them for staging).

**Step 5 — Restart the service.**

### 5.4 Verification

Use the integration test app in `../integration_testing/`:

```bash
cd ../integration_testing
pip install -r requirements.txt
streamlit run token_check.py
```

Navigate to `http://localhost:8502/?token=<real_keycloak_jwt>`. The app decodes (without signature verification) and displays all claims — confirm `preferred_username`, `country`, and `realm_roles` are present.

---

## 6. Data Setup

### 6.1 CSV files

All regulation data lives in `src/categories/`. Each CSV file represents one regulatory category. The filename (without `.csv`) becomes the `Category` value throughout the app.

```
src/categories/
├── AML.csv
├── Consumer Protection.csv
├── Data Protection.csv
├── Fraud and incident management.csv
├── Fraud Management.csv
└── Licensing.csv
```

### 6.2 Adding a new category

1. Create a new CSV file in `src/categories/` named after the category (e.g. `Cybersecurity.csv`)
2. The app discovers it automatically on next restart (or when any CSV mtime changes)
3. Required columns (any recognised name variant works — see below):
   - Country name
   - Regulator name
   - Year
   - Regulation title / name
   - Source URL
4. Additional columns become "provision" columns shown in the country detail panel

### 6.3 Column naming

The loader recognises these column name variants for each logical field:

| Logical field | Accepted column names |
|---|---|
| Country | `Country`, `country`, `Country_std` |
| Regulator | `Regulator`, `regulator`, `Regulator_std` |
| Year | `Year`, `year`, `Year approved/implemented`, `Year Approved/Implemented` |
| Source URL | `Source`, `source`, `URL`, `Official Source`, `Official Source links`, `Source_URL` |
| Title | `Issuance`, `Regulation_Title`, `title`, `Regulation / Legal Instrument`, `Primary Legal/Regulatory Framework`, `Regulation` |

If the column name you have isn't on this list, either rename the column or add your variant to `META_COL_CANDIDATES` in `app.py`.

### 6.4 Gap data files

| File | Purpose | Auto-created? |
|---|---|---|
| `src/gap_benchmarks.csv` | Benchmark definitions | No — seed manually or via Editor |
| `src/gap_mappings.csv` | Per-country status records | No — created on first mapping save |
| `src/CBregs_audit_log.csv` | Audit trail | Yes — created on first Editor action |

If `gap_benchmarks.csv` or `gap_mappings.csv` do not exist, the Gap Analysis tab initialises with empty data.

### 6.5 Country name consistency

Country names in your CSVs must match the names used in the choropleth map. Use these exact names:

| Country | Correct name |
|---|---|
| Brunei | `Brunei Darussalam` |
| Cambodia | `Cambodia` |
| Indonesia | `Indonesia` |
| Laos | `Lao PDR` |
| Malaysia | `Malaysia` |
| Myanmar | `Myanmar` |
| Philippines | `Philippines` |
| Singapore | `Singapore` |
| Thailand | `Thailand` |
| Timor-Leste | `Timor-Leste` |
| Vietnam | `Viet Nam` |

Inconsistent names (e.g. `"Vietnam"` vs `"Viet Nam"`) will cause split counts — the map will show one country but the data loader will see two.

---

## 7. Updating the App

### Linux (systemd)

```bash
sudo su - regdash
cd /home/regdash/app/dev
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt   # if dependencies changed
exit
sudo systemctl restart asean-regdash
sudo systemctl status asean-regdash
```

### Windows (NSSM)

```powershell
# Stop service
nssm stop ARIS

# Update code
cd C:\inetpub\asean-regdash\dev
git pull origin main

# Update dependencies if needed
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Restart service
nssm start ARIS
```

---

## 8. Troubleshooting

### Port 8501 already in use

```bash
# Linux — find and kill the existing process
sudo lsof -i :8501
sudo kill <PID>

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

Or change the port in the systemd unit / NSSM config to `8502` and update the Nginx proxy accordingly.

---

### Permission error writing CSV files

The process user (`regdash` on Linux, the service account on Windows) must have read+write access to `src/`:

```bash
# Linux
sudo chown -R regdash:regdash /home/regdash/app/dev/src
chmod -R 755 /home/regdash/app/dev/src
```

On Windows, right-click `src\` → Properties → Security → grant Full Control to the service account.

---

### Data edits not reflected after saving

Streamlit caches data keyed on CSV file modification time. If a file was written but the mtime didn't update (e.g. a NFS mount with ctime rounding), force a cache bust:

```bash
# Linux / macOS — touch any CSV to update its mtime
touch src/categories/AML.csv

# Windows
(Get-Item .\src\categories\AML.csv).LastWriteTime = Get-Date
```

The next page render will re-read all CSVs.

---

### Navigation bar looks broken or unstyled

The app uses a sticky horizontal top bar for navigation — there is no sidebar. If the nav bar appears unstyled or misaligned, a browser extension (dark-mode override, accessibility tool, ad-blocker) may be injecting conflicting CSS. Disable extensions for the dashboard domain, or test in an incognito / private window.

---

### Keycloak token not being picked up

1. Confirm the token is in the URL as `?token=<jwt>` (not `#token=` — hash fragments are not sent to the server)
2. Open `http://localhost:8501/?token=dummy-admin` — if this works, the issue is the real token delivery
3. Run the integration test app in `../integration_testing/` to inspect the token claims
4. Confirm `KEYCLOAK_JWKS_URL` and `KEYCLOAK_CLIENT_ID` environment variables are visible to the Streamlit process: add `st.write(os.environ.get("KEYCLOAK_JWKS_URL"))` temporarily to `app.py` for debugging
5. Check the Keycloak realm's JWKS endpoint is reachable from the server: `curl $KEYCLOAK_JWKS_URL`

---

### Streamlit version errors

If you see errors about `st.query_params` not existing, your Streamlit version is below 1.28. Upgrade:

```bash
pip install --upgrade streamlit
```

The app includes a compatibility shim (`get_query_params()`) that handles both old and new Streamlit query param APIs.
