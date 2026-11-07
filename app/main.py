"""Sales Insight Copilot: KPIs for the chain or any store, and a junior-BA assistant.

    make app   ->  http://localhost:8501

Each page is one file in app/views/; shared helpers live in app/ui.py.
"""

import streamlit as st

st.set_page_config(page_title="Sales Insight Copilot", page_icon=":material/insights:",
                   layout="wide")

st.navigation([
    st.Page("views/kpis.py", title="KPIs", default=True),
    st.Page("views/assistant.py", title="Assistant"),
], position="top").run()
