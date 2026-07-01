# ARIS — Keycloak Integration Test Tools

Tools for the integration team to validate that a Keycloak-issued JWT reaches the ARIS app correctly and contains the right claims.

**No need to run the full ARIS app.** These tools work completely standalone on any server.

Once the token handoff is confirmed working here, the only remaining step is pointing the same redirect URL at the main `app.py` — the auth logic is identical.

---

## What's here

| File | What it does |
|---|---|
| `check_token.py` | **CLI script** — paste a JWT, see all claims and what access ARIS would grant |
| `token_check.py` | **Browser UI** — pass token via URL, view decoded claims in a web page |
| `requirements.txt` | Minimal dependencies (`python-jose`, `streamlit`) |

---

## Step 0 — Install Python (fresh server)

Skip this section if Python 3.10+ is already installed (`python --version` to check).

### Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
python3 --version   # should say 3.10 or higher
```

If the version is below 3.10 (e.g. Ubuntu 20.04 ships 3.8):

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv
python3.11 --version
```

Use `python3.11` instead of `python3` in all commands below.

### Windows Server / Windows 10+

1. Download the installer from **https://www.python.org/downloads/** — choose the latest **3.11.x** or **3.12.x** Windows installer (64-bit).
2. Run the installer. **Tick "Add python.exe to PATH"** before clicking Install.
3. Open a new PowerShell window and verify:
   ```powershell
   python --version   # should say 3.11.x or 3.12.x
   ```

### macOS

```bash
# Using Homebrew (recommended)
brew install python@3.11
python3 --version
```

---

## Step 1 — Get the integration tools

### Option A — Clone the repo (if you have git)

```bash
git clone https://github.com/delgadokp92/ASEANregcomp_Dashboard.git
cd ASEANregcomp_Dashboard/integration
```

### Option B — Download just this folder

Download the zip of the repo from GitHub → Code → Download ZIP, then extract and navigate to the `integration/` folder.

---

## Step 2 — Set up the environment

```bash
# Create virtual environment (run once)
python3 -m venv .venv

# Activate — Linux / Mac
source .venv/bin/activate

# Activate — Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows PowerShell: if you get an execution policy error, run this first:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies
pip install -r requirements.txt
```

---

## Tool 1 — `check_token.py` (CLI, recommended)

The fastest way to inspect a token. No browser, no server.

```bash
python check_token.py <paste_jwt_here>
```

### Sample output

```
ARIS - ASEAN Regulatory Information System - Keycloak Token Diagnostic
Token length: 892 chars

--- Decoded claims (no signature check) -----------------
  country                       Singapore
  email                         alice@example.com
  exp                           9999999999
  iat                           1751400000
  preferred_username            alice
  realm_roles                   ["dashboard-editor"]
  sub                           user-uuid-001

--- Expiry ----------------------------------------------
  exp                     2026-07-01 10:30:00 UTC  (valid for 14 min)
  iat                     2026-07-01 10:16:00 UTC

--- Claims the dashboard app reads ----------------------
  preferred_username            alice
  country (custom)              Singapore
  realm_roles                   ["dashboard-editor"]
  sub                           user-uuid-001
  email                         alice@example.com

--- Access simulation (what the app would grant) --------
  User                    alice
  Country lock            Singapore
  IS_ADMIN                False
  IS_AUTHENTICATED        True
  Visible tabs            Map, Table, Gap Analysis, Editor

  [OK]  Authenticated as editor for Singapore.

--- Signature verification ------------------------------
  Skipped - set KEYCLOAK_JWKS_URL and KEYCLOAK_CLIENT_ID to enable.
```

### Optional: verify the JWT signature

```bash
# Linux / Mac
export KEYCLOAK_JWKS_URL=https://your-keycloak/realms/your-realm/protocol/openid-connect/certs
export KEYCLOAK_CLIENT_ID=asean-regdash
python check_token.py <jwt>

# Windows PowerShell
$env:KEYCLOAK_JWKS_URL = "https://your-keycloak/realms/your-realm/protocol/openid-connect/certs"
$env:KEYCLOAK_CLIENT_ID = "asean-regdash"
python check_token.py <jwt>
```

---

## Tool 2 — `token_check.py` (Browser UI)

Useful for sharing decoded output with non-technical stakeholders.

```bash
streamlit run token_check.py
```

Then open in browser:

```
http://localhost:8501/?token=<jwt>
```

Displays all decoded claims and highlights the fields ARIS uses.

---

## Claims ARIS requires

| Claim | Type | Required | Description |
|---|---|---|---|
| `preferred_username` | string | Yes | Display name shown in UI and written to audit log |
| `country` | string | Yes | **Custom claim** — ASEAN country name, or `"NA"` for admin |
| `realm_roles` | list | Yes | `"dashboard-admin"` = admin, `"dashboard-editor"` = country editor |
| `sub` | string | Yes | Standard Keycloak subject ID |
| `email` | string | No | Not used by the app, but useful for debugging |

> **Important:** `country` is a **custom** Keycloak claim. It will not appear in the token unless you add a User Attribute mapper in the Keycloak client. See below.

### Access levels

| `country` | `realm_roles` | Access |
|---|---|---|
| Any country name | `dashboard-editor` | Editor locked to that country + Gap Analysis |
| `NA` | `dashboard-admin` | Full admin — all countries, all features |
| Missing | any | Public read-only (Map + Table only) |

---

## Keycloak setup checklist

- [ ] Client created: `asean-regdash`, protocol `openid-connect`
- [ ] Custom mapper added: User Attribute → Token Claim Name `country`, add to access token ✓
- [ ] Realm roles created: `dashboard-admin`, `dashboard-editor`
- [ ] Test user has `country` attribute set (e.g. `Singapore`)
- [ ] Test user assigned a realm role

---

## Common issues

**`country` claim missing from token**
Add a mapper: Keycloak Admin → Clients → `asean-regdash` → Client Scopes → (dedicated scope) → Mappers → Add by configuration → User Attribute → set Token Claim Name to `country` and tick "Add to access token".

**`realm_roles` missing or empty**
Roles must be **realm-level**, not client-level. Keycloak Admin → Users → select user → Role Mappings → Realm Roles → assign `dashboard-editor` or `dashboard-admin`.

**Token expires before reaching the app**
Keycloak access tokens are short-lived (typically 5 minutes). The redirect from Keycloak to ARIS must complete within that window. Check `exp` in the `check_token.py` output.

**Signature verification fails**
Confirm `KEYCLOAK_JWKS_URL` is the correct JWKS endpoint for your realm and is reachable from the machine running the script:
```bash
curl $KEYCLOAK_JWKS_URL
```

---

## What's next — connecting to the full ARIS app

Once `check_token.py` shows `[OK] Authenticated as editor for <country>` (or admin), the token handoff is proven. The only remaining step is wiring the same redirect into the live app.

### What the ARIS team needs to do (already done in app.py)

The `parse_auth_token()` function in `app.py` is already wired and waiting. No code changes are needed on the ARIS side — just:

1. Set the two environment variables on the ARIS server:
   ```bash
   KEYCLOAK_JWKS_URL=https://your-keycloak/realms/your-realm/protocol/openid-connect/certs
   KEYCLOAK_CLIENT_ID=asean-regdash
   ```

2. Uncomment the JWT validation block inside `parse_auth_token()` in `app.py` (marked with `# TODO`).

3. Restart the ARIS app.

That's it. The dummy tokens (`dummy-admin`, `dummy-vietnam`, `dummy-singapore`) will still work for testing after the real token path is live — remove them only when going to full production.

### Go/no-go checklist before switching to live

- [ ] `check_token.py` shows all required claims present for a real Keycloak token
- [ ] Signature verification passes (`KEYCLOAK_JWKS_URL` + `KEYCLOAK_CLIENT_ID` set)
- [ ] Country name in token matches exactly one of the 11 ASEAN country names used in ARIS
- [ ] Admin user token contains `realm_roles: ["dashboard-admin"]`
- [ ] ARIS server environment variables set
- [ ] TODO block in `parse_auth_token()` uncommented
