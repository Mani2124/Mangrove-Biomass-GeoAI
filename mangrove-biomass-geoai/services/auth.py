from __future__ import annotations

import hashlib
import hmac
import os
import re
import sqlite3
from uuid import uuid4

import streamlit as st

from services.database import connection, initialise_database, utc_now

ITERATIONS = 390_000


def _hash_password(password, salt=None):
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password, encoded):
    try:
        algorithm, iterations, salt, expected = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations))
        return hmac.compare_digest(actual.hex(), expected)
    except (TypeError, ValueError):
        return False


def create_account(email, password, display_name):
    initialise_database()
    email, display_name = email.strip().lower(), display_name.strip()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        raise ValueError("Enter a valid email address.")
    if len(password) < 10:
        raise ValueError("Use at least 10 characters for the password.")
    if not display_name:
        raise ValueError("Enter your name.")
    user_id = uuid4().hex
    try:
        with connection() as database:
            database.execute(
                "INSERT INTO users(id,email,password_hash,display_name,created_at) VALUES(?,?,?,?,?)",
                (user_id, email, _hash_password(password), display_name, utc_now()),
            )
    except sqlite3.IntegrityError as error:
        raise ValueError("An account already exists for this email.") from error
    return {"id": user_id, "email": email, "display_name": display_name, "plan": "demo"}


def sign_in(email, password):
    initialise_database()
    with connection() as database:
        row = database.execute(
            "SELECT id,email,password_hash,display_name,plan FROM users WHERE email=? COLLATE NOCASE",
            (email.strip(),),
        ).fetchone()
    if row is None or not _verify_password(password, row["password_hash"]):
        return None
    return {key: row[key] for key in ("id", "email", "display_name", "plan")}


def authentication_gate():
    initialise_database()
    if st.session_state.get("authenticated_user"):
        return st.session_state.authenticated_user
    st.title("SpaceVexa Biomass Platform")
    st.caption("Sign in to access private studies and scientific results.")
    sign_in_tab, register_tab = st.tabs(["Sign in", "Create account"])
    with sign_in_tab:
        with st.form("sign_in_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            user = sign_in(email, password)
            if user is None:
                st.error("Email or password is incorrect.")
            else:
                st.session_state.authenticated_user = user
                st.rerun()
    with register_tab:
        with st.form("register_form"):
            name = st.text_input("Name")
            email = st.text_input("Email", key="register_email")
            password = st.text_input("Password", type="password", key="register_password")
            submitted = st.form_submit_button("Create account")
        if submitted:
            try:
                st.session_state.authenticated_user = create_account(email, password, name)
                st.rerun()
            except ValueError as error:
                st.error(str(error))
    return None


def logout_button(user):
    with st.sidebar:
        st.caption(f"Signed in as **{user['display_name']}**")
        st.caption(f"Plan: {user['plan'].title()}")
        if st.button("Sign out", width="stretch"):
            st.session_state.clear()
            st.rerun()
