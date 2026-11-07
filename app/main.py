"""Sales Insight Copilot: data in, KPIs and anomalies, causes, actions, tests, questions.

    make app   ->  http://localhost:8501

Each page is one file in app/views/; shared helpers live in app/ui.py.
"""

import streamlit as st

st.set_page_config(page_title="Sales Insight Copilot", page_icon=":material/insights:",
                   layout="wide")

st.navigation([
    st.Page("views/pipeline.py", title="Pipeline", default=True),
    st.Page("views/actions.py", title="Actions"),
    st.Page("views/test.py", title="Test"),
    st.Page("views/ask.py", title="Ask"),
], position="top").run()
