import streamlit as st
from jose import jwt, JWTError

st.set_page_config(page_title="Keycloak Token Check", layout="centered")
st.title("Keycloak SSO Token Check")

params = st.query_params
token = params.get("token", "")

if not token:
    st.warning("No token received. Pass it via URL: `?token=<jwt>`")
    st.stop()

st.success("Token received in query params.")

try:
    # Decode without verifying signature — test only
    claims = jwt.decode(token, key="", options={
        "verify_signature": False,
        "verify_aud": False,
        "verify_exp": False,
    })
    st.subheader("Token Claims")
    st.json(claims)

    # Highlight the fields Keycloak typically provides
    important = {k: claims[k] for k in ("sub", "preferred_username", "email", "realm_access", "exp") if k in claims}
    if important:
        st.subheader("Key Fields")
        st.json(important)

except JWTError as e:
    st.error(f"Failed to decode token: {e}")
    st.subheader("Raw Token")
    st.code(token)
