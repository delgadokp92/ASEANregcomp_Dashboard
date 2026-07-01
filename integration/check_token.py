"""
check_token.py — ARIS (ASEAN Regulatory Information System) token diagnostic

Usage:
    python check_token.py <jwt>
    python check_token.py --token <jwt>
    echo "<jwt>" | python check_token.py

Decodes a Keycloak JWT and shows exactly what the dashboard app would extract.
Signature verification is OFF by default (safe for testing — the token still
needs to arrive intact).  Set KEYCLOAK_JWKS_URL + KEYCLOAK_CLIENT_ID env vars
to also run a verified decode.

No Streamlit or server required — runs anywhere with python-jose installed.
"""

import json
import os
import sys
import textwrap
from datetime import datetime, timezone


# ── colour helpers (no external deps) ────────────────────────────────────────
def _c(code: str, text: str) -> str:
    """Wrap text in ANSI colour if stdout is a tty."""
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"

def ok(t):  return _c("92", t)   # green
def err(t): return _c("91", t)   # red
def hdr(t): return _c("96;1", t) # cyan bold
def dim(t): return _c("90", t)   # dark gray
def bold(t):return _c("1",  t)


# ── token source ──────────────────────────────────────────────────────────────
def get_token() -> str:
    args = sys.argv[1:]
    if "--token" in args:
        idx = args.index("--token")
        if idx + 1 < len(args):
            return args[idx + 1].strip()
    if args and not args[0].startswith("-"):
        return args[0].strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print(err("No token provided."))
    print()
    print("Usage:")
    print("  python check_token.py <jwt>")
    print("  python check_token.py --token <jwt>")
    print("  echo '<jwt>' | python check_token.py")
    sys.exit(1)


# ── decode ────────────────────────────────────────────────────────────────────
def decode_unverified(token: str) -> dict:
    """Decode JWT payload via raw base64 — no signature check, no external deps."""
    import base64
    parts = token.split(".")
    if len(parts) < 2:
        print(err("Not a valid JWT (expected 3 dot-separated parts)."))
        sys.exit(1)
    payload_b64 = parts[1]
    # JWT base64url — restore padding
    payload_b64 += "=" * (-len(payload_b64) % 4)
    try:
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        return json.loads(payload_bytes.decode("utf-8"))
    except Exception as e:
        print(err(f"Failed to decode token payload: {e}"))
        print()
        print("Raw token (first 120 chars):")
        print(dim(token[:120] + ("..." if len(token) > 120 else "")))
        sys.exit(1)


def decode_verified(token: str, jwks_url: str, client_id: str) -> dict | None:
    try:
        from jose import jwt, JWTError
        import urllib.request
        raw = urllib.request.urlopen(jwks_url, timeout=5).read()
        jwks = json.loads(raw)
        return jwt.decode(
            token, jwks,
            algorithms=["RS256"],
            audience=client_id,
            options={"verify_at_hash": False},
        )
    except Exception as e:
        return {"_error": str(e)}


# ── access simulation (mirrors app.py logic exactly) ─────────────────────────
def simulate_access(claims: dict) -> dict:
    import re
    def _san(s, pattern, maxlen=50):
        return re.sub(pattern, "", str(s or ""))[:maxlen] or None

    auth_country = _san(claims.get("country", ""),   r"[^A-Za-z0-9 \-_]")
    auth_user    = _san(claims.get("preferred_username", ""), r"[^A-Za-z0-9 \-_@.]")
    roles        = claims.get("realm_roles", [])

    is_admin = "dashboard-admin" in roles
    is_auth  = True  # token present → always authenticated

    tabs = ["Map", "Table", "Gap Analysis", "Editor"]
    if not is_auth:
        tabs = ["Map", "Table"]

    return {
        "auth_country": auth_country,
        "auth_user":    auth_user,
        "roles":        roles,
        "IS_ADMIN":     is_admin,
        "IS_AUTHENTICATED": is_auth,
        "visible_tabs": tabs,
        "country_lock": None if is_admin else auth_country,
    }


# ── display ───────────────────────────────────────────────────────────────────
def show_section(title: str):
    print()
    print(hdr(f"--- {title} {'-' * max(0, 52 - len(title))}"))

def show_kv(key: str, value, width: int = 22):
    val_str = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
    print(f"  {bold(key.ljust(width))}  {val_str}")

def show_claims_table(claims: dict):
    """Print all claims in a readable table."""
    for k, v in sorted(claims.items()):
        if k.startswith("_"):
            continue
        if isinstance(v, (dict, list)):
            v_str = json.dumps(v, ensure_ascii=False)
        else:
            v_str = str(v)
        # wrap long values
        lines = textwrap.wrap(v_str, width=70)
        print(f"  {dim(k.ljust(28))}  {lines[0] if lines else ''}")
        for extra in lines[1:]:
            print(f"  {' ' * 30}  {extra}")


def main():
    token = get_token()

    print()
    print(bold("ARIS - ASEAN Regulatory Information System - Keycloak Token Diagnostic"))
    print(dim(f"Token length: {len(token)} chars"))

    # ── Unverified decode ────────────────────────────────────────────────────
    show_section("Decoded claims (no signature check)")
    claims = decode_unverified(token)
    show_claims_table(claims)

    # ── Expiry check ─────────────────────────────────────────────────────────
    show_section("Expiry")
    exp = claims.get("exp")
    iat = claims.get("iat")
    if exp:
        exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)
        now    = datetime.now(tz=timezone.utc)
        delta  = exp_dt - now
        if delta.total_seconds() > 0:
            mins = int(delta.total_seconds() // 60)
            show_kv("exp", f"{exp_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}  ({ok(f'valid for {mins} min')})")
        else:
            show_kv("exp", f"{exp_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}  ({err('EXPIRED')})")
    else:
        show_kv("exp", err("(missing)"))
    if iat:
        iat_dt = datetime.fromtimestamp(iat, tz=timezone.utc)
        show_kv("iat", iat_dt.strftime("%Y-%m-%d %H:%M:%S UTC"))

    # ── App-specific claims ──────────────────────────────────────────────────
    show_section("Claims the dashboard app reads")
    app_claims = {
        "preferred_username": claims.get("preferred_username"),
        "country (custom)":   claims.get("country"),
        "realm_roles":        claims.get("realm_roles"),
        "sub":                claims.get("sub"),
        "email":              claims.get("email"),
    }
    missing = []
    for k, v in app_claims.items():
        if v is None:
            print(f"  {dim(k.ljust(28))}  {err('(missing)')}")
            missing.append(k)
        else:
            val = json.dumps(v) if isinstance(v, list) else str(v)
            print(f"  {dim(k.ljust(28))}  {ok(val)}")

    if missing:
        print()
        print(f"  {err('[WARN]')}  Missing claims: {', '.join(missing)}")
        if "country (custom)" in missing:
            print(dim("     → 'country' is a custom Keycloak claim — add a User Attribute mapper."))
        if "realm_roles" in missing:
            print(dim("     → 'realm_roles' comes from realm-level roles, not client roles."))

    # ── Access simulation ────────────────────────────────────────────────────
    show_section("Access simulation (what the app would grant)")
    access = simulate_access(claims)
    show_kv("User",          access["auth_user"]    or err("(none)"))
    show_kv("Country lock",  access["country_lock"] or (ok("None (admin)") if access["IS_ADMIN"] else err("(none — no country claim)")))
    show_kv("IS_ADMIN",      ok("True") if access["IS_ADMIN"] else "False")
    show_kv("IS_AUTHENTICATED", ok("True") if access["IS_AUTHENTICATED"] else "False")
    show_kv("Visible tabs",  ", ".join(access["visible_tabs"]))
    if access["IS_ADMIN"]:
        print(f"\n  {ok('[OK]')}  Admin access - all countries, full Gap Analysis admin actions.")
    elif access["IS_AUTHENTICATED"] and access["country_lock"]:
        print(f"\n  {ok('[OK]')}  Authenticated as editor for {bold(access['country_lock'])}.")
    else:
        print(f"\n  {err('[WARN]')}  No valid country claim - user would get public (read-only) view.")

    # ── Verified decode (optional) ───────────────────────────────────────────
    jwks_url  = os.environ.get("KEYCLOAK_JWKS_URL", "")
    client_id = os.environ.get("KEYCLOAK_CLIENT_ID", "")
    if jwks_url and client_id:
        show_section("Signature verification (KEYCLOAK_JWKS_URL set)")
        result = decode_verified(token, jwks_url, client_id)
        if result and "_error" in result:
            print(f"  {err('FAILED:')} {result['_error']}")
        elif result:
            print(f"  {ok('[OK]  Signature valid.')}")
    else:
        show_section("Signature verification")
        print(f"  {dim('Skipped - set KEYCLOAK_JWKS_URL and KEYCLOAK_CLIENT_ID to enable.')}")
        print(dim("  Example:"))
        print(dim("    KEYCLOAK_JWKS_URL=https://your-keycloak/realms/your-realm/protocol/openid-connect/certs"))
        print(dim("    KEYCLOAK_CLIENT_ID=asean-regdash"))
        print(dim("    python check_token.py <jwt>"))

    print()


if __name__ == "__main__":
    main()
