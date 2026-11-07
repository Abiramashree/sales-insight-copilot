"""Sales Insight Copilot: what changed this week, why, and what to do.

    make app   ->  http://localhost:8501

Each page is one file in app/views/; shared helpers live in app/ui.py.
"""

import streamlit as st

st.set_page_config(page_title="Sales Insight Copilot", page_icon=":material/insights:",
                   layout="wide")

st.navigation([
    st.Page("views/week.py", title="Week", default=True),
    st.Page("views/stores.py", title="Stores"),
    st.Page("views/ask.py", title="Ask"),
], position="top").run()
