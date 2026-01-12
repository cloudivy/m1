import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define functions (paste your code here)
def load_df_pidws(uploaded_file):
    df_pidws = pd.read_excel(uploaded_file)
    df_pidws.columns = df_pidws.columns.str.replace('\n', ' ').str.strip()
    df_pidws = df_pidws.rename(columns={'Avg. Perim. Pos (m)': 'chainage'})
    df_pidws['DateTime'] = pd.to_datetime(df_pidws['Date'] + ' ' + df_pidws['Time'], format='%d-%m-%Y %H:%M:%S')
    return df_pidws

def load_df_lds(uploaded_file):
    df_lds = pd.read_excel(uploaded_file)
    df_lds = df_lds.rename(columns={
        'Leak_Size_m3_hr': 'leak size',
        'Chainage_Location_km': 'chainage'
    })
    df_lds['DateTime'] = pd.to_datetime(df_lds['Date'].astype(str) + ' ' + df_lds['Time'])
    return df_lds

def parse_duration(dur_str):
    if pd.isna(dur_str):
        return pd.Timedelta(0)
    dur_str = str(dur_str).strip().lower().replace(' ', '')
    mins = 0
    secs = 0
    if 'm' in dur_str:
        m_part = dur_str.split('m')[0]
        if m_part.isdigit():
            mins = int(m_part)
        dur_str = dur_str.split('m')[-1]
    if 's' in dur_str:
        s_part = dur_str.replace('s', '')
        if s_part.isdigit():
            secs = int(s_part)
    return pd.Timedelta(minutes=mins, seconds=secs)

def classify_pilferage(pidws_df, lds_df, chainage_tol=1.0, time_window_hours=24):
    classified = []
    pidws_df = pidws_df.copy()
    pidws_df['duration_td'] = pidws_df['Event Duration'].apply(parse_duration)
    pidws_df['end_time'] = pidws_df['DateTime'] + pidws_df['duration_td']
    for _, event in pidws_df.iterrows():
        window_end = event['end_time'] + pd.Timedelta(hours=time_window_hours)
        mask = (lds_df['DateTime'] > window_end) & \
               (np.abs(lds_df['chainage'] - event['chainage']) <= chainage_tol)
        matches = lds_df[mask].copy()
        if not matches.empty:
            matches['linked_event_time'] = event['DateTime']
            matches['linked_chainage'] = event['chainage']
            time_diff_seconds = (matches['DateTime'] - window_end).dt.total_seconds()
            matches['pilferage_score'] = 1 / (1 + time_diff_seconds / 3600)
            classified.append(matches)
    if classified:
        return pd.concat(classified, ignore_index=True)
    return pd.DataFrame()

st.set_page_config(page_title="PIDWS-LDS Pilferage Classifier", layout="wide")

st.title("🔍 Pipeline Pilferage Classification: PIDWS vs LDS Events")

col1, col2 = st.columns(2)
with col1:
    pidws_file = st.file_uploader("Upload df_pidws_III.xlsx", type=["xlsx"], key="pidws")
with col2:
    lds_file = st.file_uploader("Upload df_lds_III.xlsx", type=["xlsx"], key="lds")

if pidws_file and lds_file:
    with st.spinner("Loading and processing data..."):
        df_pidws = load_df_pidws(pidws_file)
        df_lds = load_df_lds(lds_file)

    st.subheader("📊 Data Previews")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_pidws[['DateTime', 'chainage', 'Event Duration']].head())
    with col2:
        st.dataframe(df_lds[['DateTime', 'chainage', 'leak size']].head())

    # Parameters
    st.subheader("⚙️ Classification Parameters")
    col1, col2 = st.columns(2)
    chainage_tol = col1.slider("Chainage Tolerance (km)", 0.1, 5.0, 1.0)
    time_window = col2.slider("Time Window After PIDWS (hours)", 1, 168, 24)

    if st.button("🚀 Classify Pilferage Events"):
        with st.spinner("Classifying..."):
            pilferage_leaks = classify_pilferage(df_pidws, df_lds, chainage_tol, time_window)
        
        if not pilferage_leaks.empty:
            st.success(f"✅ Found {len(pilferage_leaks)} pilferage leak instances!")
            
            st.subheader("🔥 Pilferage Leaks")
            st.dataframe(pilferage_leaks[['DateTime', 'chainage', 'leak size', 'pilferage_score', 'linked_event_time']].sort_values('pilferage_score', ascending=False))
            
            # Download
            csv = pilferage_leaks.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Pilferage CSV", csv, "pilferage_leaks.csv", "text/csv")
            
            # Visualization
            st.subheader("📈 Visualization")
            fig = px.scatter(pilferage_leaks, x='DateTime', y='chainage', size='pilferage_score', color='leak size',
                             hover_data=['linked_event_time', 'linked_chainage'],
                             title="Pilferage Leaks: Score vs Time & Chainage")
            st.plotly_chart(fig, use_container_width=True)
            
            # Stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Matches", len(pilferage_leaks))
            col2.metric("Avg Score", pilferage_leaks['pilferage_score'].mean())
            col3.metric("Max Score", pilferage_leaks['pilferage_score'].max())
        else:
            st.warning("❌ No pilferage leaks classified. Try adjusting parameters.")

    # All LDS timeline
    st.subheader("🕐 All LDS Events Timeline")
    fig_timeline = px.timeline(df_lds, x_start="DateTime", x_end="DateTime", y="chainage", color="leak size")
    st.plotly_chart(fig_timeline, use_container_width=True)
