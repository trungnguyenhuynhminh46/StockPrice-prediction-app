import streamlit as st
# Pages 
from intro import intro
from single import single
from multiple import multiple


page_names_to_funcs = {
    "—": intro,
    "Dùng một mô hình để dự đoán": single,
    "So sánh các mô hình": multiple
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()