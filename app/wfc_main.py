import os
import sys

import logzero
import streamlit as st

from logzero import logger

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

@st.cache_resource
def Initialize_wfc():
    logzero.logfile("wfc.log")
    logger.debug("wfc ..")

def wfcmain():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.image(os.path.join('images', 'main.png'))

    st.write("## WFC and Usage ")

    st.write("""
            * Use WFC to create summary, understand content 
            """)

st.set_page_config(page_title="wfc", page_icon=os.path.join('images', 'favicon.ico'), layout="wide", menu_items=None)

mod_page_style = """
        <style>
            .sidebar .sidebar-content {
                width: 375px;
            }
        </style>
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
    """
st.markdown(mod_page_style, unsafe_allow_html=True)

Initialize_wfc()
wfcmain()
