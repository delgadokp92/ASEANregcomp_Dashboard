# ARIS — Keycloak Integration Test Tools

Tools for the integration team to validate that a Keycloak-issued JWT reaches the ARIS app correctly and contains the right claims.

No need to run the full ARIS app — these tools work completely standalone.

---

## What's here

| File | What it does |
|---|---|
| `check_token.py` | **CLI script** — paste a JWT, see all claims and what access ARIS would grant |
| `token_check.py` | **Browser UI** — pass token via URL, view decoded claims in a web page |
| `requirements.txt` | Minimal dependencies (`python-jose`, `streamlit`) |

---

## Setup

Requires **Python 3.10+**. Run these commands once:

```bash
python -m venv .venv

# Activate — Linux / Mac
source .venv/bin/activate

# Activate — Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Install
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
- [ ] Valid Redirect URI includes the ARIS app URL
- [ ] Custom mapper added: User Attribute → Token Claim Name `country`, add to access token ✓
- [ ] Realm roles created: `dashboard-admin`, `dashboard-editor`
- [ ] Test user has `country` attribute set (e.g. `Singapore`)
- [ ] Test user assigned a realm role
- [ ] Token delivered to app as `?token=<jwt>` in the redirect URL

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
