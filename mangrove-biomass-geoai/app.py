"""SpaceVexa entry point. Run: python -m streamlit run app.py"""
import streamlit as st

from services.auth import authentication_gate, logout_button
from ui.dashboard import main

st.set_page_config(
    page_title="SpaceVexa Biomass Platform",
    page_icon="🌍",
    layout="wide",
)

user = authentication_gate()
if user is None:
    st.stop()

logout_button(user)
main(user)
