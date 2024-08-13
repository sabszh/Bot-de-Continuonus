import streamlit as st

col1, col2 = st.columns([6, 2])

with col1:
    "# Column 1"
    "Text " * 100

with col2:
    "# Column 2"
    "Text " * 20

st.markdown("""
    <style>
        [data-testid="column"]:nth-child(2){
            background-color: transparent;
        }
    </style>
    """, unsafe_allow_html=True
)
