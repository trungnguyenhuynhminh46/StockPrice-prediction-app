import streamlit as st
# Pages 
from intro import intro
from single import single
from multiple import multiple


page_names_to_funcs = {
    "—": intro,
    "Single": single,
    "Compare all models": multiple
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()