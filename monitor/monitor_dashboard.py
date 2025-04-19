import streamlit as st 
import pandas as pd
from datetime import datetime
import altair as alt
import sys
import os
sys.path.append(os.path.abspath("."))
from monitor.db_logger import get_all_logs
from streamlit_autorefresh import st_autorefresh
from monitor.db_logger import get_all_logs, export_logs_to_csv
from io import StringIO

st.set_page_config(
    page_title="Chatbot Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st_autorefresh(interval=5000, limit=None, key="refresh")

st.title("Fiscozen Chatbot Monitoring Dashboard")

# Load logs from database
rows = get_all_logs()
if not rows:
    st.warning("⚠️ No log data found in the database.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(rows, columns=["id", "timestamp", "event", "query", "response", "feedback", "response_time"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Metrics
st.subheader("📌 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("✅ Answered", df[df.event == "answered"].shape[0])
col2.metric("❌ Out of Scope", df[df.event == "out_of_scope"].shape[0])
col3.metric("👍 Positive Feedback", df[df.feedback == "👍"].shape[0])
col4.metric("👎 Negative Feedback", df[df.feedback == "👎"].shape[0])

# Charts
st.markdown("---")
st.subheader("📉 Trends Over Time")

df = df.dropna(subset=["timestamp"])
if not df.empty:
    answered_over_time = df[df.event == "answered"].groupby(pd.Grouper(key="timestamp", freq="15min")).size().reset_index(name="count")
    out_of_scope_over_time = df[df.event == "out_of_scope"].groupby(pd.Grouper(key="timestamp", freq="15min")).size().reset_index(name="count")
    feedback_over_time = df[df.event == "feedback"].groupby([pd.Grouper(key="timestamp", freq="15min"), "feedback"]).size().unstack(fill_value=0).reset_index()
    response_times = df.dropna(subset=["response_time"])

    st.altair_chart(
        alt.Chart(answered_over_time)
        .mark_line(point=True)
        .encode(
            x="timestamp:T",
            y="count:Q",
            tooltip=["timestamp:T", "count:Q"]
        )
        .properties(title="✅ Answered Queries Over Time")
        .interactive(),
        use_container_width=True
    )

    st.altair_chart(
        alt.Chart(out_of_scope_over_time)
        .mark_line(point=True, color="orange")
        .encode(
            x="timestamp:T",
            y="count:Q",
            tooltip=["timestamp:T", "count:Q"]
        )
        .properties(title="❌ Out-of-Scope Queries Over Time")
        .interactive(),
        use_container_width=True
    )

    feedback_melted = feedback_over_time.melt(id_vars="timestamp", var_name="Feedback", value_name="Count")
    st.altair_chart(
        alt.Chart(feedback_melted)
        .mark_bar()
        .encode(
            x="timestamp:T",
            y="Count:Q",
            color=alt.Color("Feedback:N", legend=alt.Legend(title="Feedback")),
            tooltip=["timestamp:T", "Feedback:N", "Count:Q"]
        )
        .properties(title="👍👎 Feedback Over Time")
        .interactive(),
        use_container_width=True
    )

    if not response_times.empty:
        st.line_chart(response_times.set_index("timestamp")["response_time"].rename("Response Time (s)"))

else:
    st.info("No timestamped data available for time-series charts.")

# Raw Data Explorer
st.markdown("---")
st.subheader("📄 Explore Log Records")
st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)

# Fetch logs and convert to DataFrame
rows = get_all_logs()
df = pd.DataFrame(rows, columns=["id", "timestamp", "event", "query", "response", "feedback", "response_time"])

# Download Logs as CSV
st.markdown("### 📥 Download Logs")
csv_buffer = StringIO()
df.to_csv(csv_buffer, index=False)
st.download_button(
    label="Download Logs as CSV",
    data=csv_buffer.getvalue(),
    file_name="fiscozen_chatbot_logs.csv",
    mime="text/csv"
)
