import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Page config
st.set_page_config(
    page_title="Pipeline Pilferage Detection",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛢️ Pipeline Pilferage Detection Dashboard")
st.markdown("---")

# Sidebar controls
st.sidebar.header("📁 Data Upload")
uploaded_pidws = st.sidebar.file_uploader("PIDWS data (df_pidws_III.xlsx)", type="xlsx")
uploaded_lds = st.sidebar.file_uploader("LDS data (df_lds_III.xlsx)", type="xlsx")

st.sidebar.header("⚙️ Parameters")
chainage_tol = st.sidebar.slider("Chainage Tolerance (km)", 0.1, 2.0, 0.5, 0.1)
time_window_hours = st.sidebar.slider("Time Window (hours)", 12, 72, 48, 6)

if st.sidebar.button("🔄 Analyze Data", type="primary"):
    if uploaded_pidws is not None and uploaded_lds is not None:
        with st.spinner("Processing pipeline data..."):
            df_pidws = pd.read_excel(uploaded_pidws)
            df_lds = pd.read_excel(uploaded_lds)
            
            st.session_state.df_pidws = df_pidws
            st.session_state.df_lds = df_lds
            st.session_state.chainage_tol = chainage_tol
            st.session_state.time_window_hours = time_window_hours
            st.session_state.status = "processed"
            st.rerun()
    else:
        st.sidebar.error("❌ Upload both files")

# Main analysis
if 'status' in st.session_state and st.session_state.status == "processed":
    df_pidws = st.session_state.df_pidws
    df_lds = st.session_state.df_lds
    chainage_tol = st.session_state.chainage_tol
    time_window_hours = st.session_state.time_window_hours
    
    @st.cache_data
    def preprocess_pidws(df):
        df = df.copy()
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d-%m-%Y %H:%M:%S')
        
        def parse_duration(dur_str):
            if pd.isna(dur_str):
                return pd.Timedelta(0)
            dur_str = str(dur_str).strip().lower().replace(' ', '')
            mins, secs = 0, 0
            if 'm' in dur_str:
                m_part = dur_str.split('m')[0]
                if m_part.isdigit():
                    mins = int(m_part)
            if 's' in dur_str:
                s_part = dur_str.replace('s', '')
                if s_part.isdigit():
                    secs = int(s_part)
            return pd.Timedelta(minutes=mins, seconds=secs)
        
        df['duration_td'] = df['Event Duration'].apply(parse_duration)
        df['end_time'] = df['DateTime'] + df['duration_td']
        return df
    
    @st.cache_data
    def preprocess_lds(df):
        df = df.copy()
        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])
        return df
    
    df_pidws = preprocess_pidws(df_pidws)
    df_lds = preprocess_lds(df_lds)
    
    @st.cache_data
    def classify_pilferage(pidws_df, lds_df, chainage_tol, time_window_hours):
        classified = []
        for _, event in pidws_df.iterrows():
            window_end = event['end_time'] + pd.Timedelta(hours=time_window_hours)
            mask = (lds_df['DateTime'] > window_end) & \
                   (np.abs(lds_df['chainage'] - event['chainage']) <= chainage_tol)
            matches = lds_df[mask].copy()
            if not matches.empty:
                matches['linked_event_time'] = event['DateTime']
                matches['linked_chainage'] = event['chainage']
                matches['pilferage_score'] = 1 / (1 + (matches['DateTime'] - window_end).dt.total_seconds() / 3600)
                classified.append(matches)
        return pd.concat(classified, ignore_index=True) if classified else pd.DataFrame()
    
    pilferage_leaks = classify_pilferage(df_pidws, df_lds, chainage_tol, time_window_hours)
    
    df_lds_classified = df_lds.copy()
    df_lds_classified['is_pilferage'] = False
    if not pilferage_leaks.empty:
        pilferage_ids = pilferage_leaks[['DateTime', 'chainage']].drop_duplicates()
        mask_pilferage = df_lds_classified.set_index(['DateTime', 'chainage']).index.isin(
            pilferage_ids.set_index(['DateTime', 'chainage']).index
        )
        df_lds_classified.loc[mask_pilferage, 'is_pilferage'] = True
    
    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total LDS Events", len(df_lds))
        st.metric("PIDWS Events", len(df_pidws))
        st.metric("🟡 Pilferage Events", len(pilferage_leaks))
        pilferage_pct = (len(pilferage_leaks)/len(df_lds)*100) if len(df_lds)>0 else 0
        st.metric("Pilferage Rate", f"{pilferage_pct:.1f}%")
    
    with col2:
        avg_pilferage = df_lds_classified[df_lds_classified['is_pilferage']]['leak size'].mean()
        avg_other = df_lds_classified[~df_lds_classified['is_pilferage']]['leak size'].mean()
        st.metric("Avg Pilferage Size", f"{avg_pilferage:.1f}" if not pd.isna(avg_pilferage) else "0")
        st.metric("Avg Other Size", f"{avg_other:.1f}")
    
    st.markdown("---")
    
    # Chainage Distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Chainage Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_pidws['chainage'], bins=30, alpha=0.7, label='PIDWS', color='orange', density=True)
        ax.hist(df_lds['chainage'], bins=30, alpha=0.7, label='LDS Leaks', color='blue', density=True)
        if not pilferage_leaks.empty:
            ax.axvline(pilferage_leaks['linked_chainage'].mean(), color='red', linestyle='--', 
                      label=f'Pilferage Mean')
        ax.set_xlabel('Chainage (km)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Classification Summary")
        summary = df_lds_classified['is_pilferage'].value_counts().reset_index()
        summary['is_pilferage'] = summary['is_pilferage'].map({True: 'Pilferage', False: 'Other'})
        st.dataframe(summary.rename(columns={'is_pilferage': 'Class', 'count': 'Events'}))
    
    # Time Series
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Events Over Time")
        all_events = pd.concat([
            df_pidws[['DateTime']].assign(type='Digging'),
            df_lds[['DateTime']].assign(type='Leak'),
            pilferage_leaks[['DateTime']].assign(type='Pilferage')
        ])
        hourly = all_events.groupby([all_events['DateTime'].dt.floor('H'), 'type']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        hourly.plot(ax=ax, linewidth=2, marker='o')
        ax.set_title('Events per Hour')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Leak Size Comparison")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df_lds_classified, x='is_pilferage', y='leak size', ax=ax)
        ax.set_title('Pilferage vs Other Leaks')
        st.pyplot(fig)
    
    # Top clusters
    if not pilferage_leaks.empty:
        st.subheader("🔥 Top Pilferage Clusters")
        clusters = pilferage_leaks.groupby('linked_chainage')['leak size'].agg(['count', 'mean', 'max']).round(1)
        st.dataframe(clusters.sort_values('count', ascending=False).head(10))
    
    # Scatter plot
    st.subheader("🗺️ Leak Timeline")
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['red' if x else 'blue' for x in df_lds_classified['is_pilferage']]
    sizes = np.clip(df_lds_classified['leak size'], 20, 200)
    ax.scatter(df_lds_classified['DateTime'], df_lds_classified['chainage'], 
              c=colors, s=sizes, alpha=0.6)
    ax.set_xlabel('DateTime')
    ax.set_ylabel('Chainage (km)')
    ax.grid(True, alpha=0.3)
    ax.legend(['Pilferage', 'Other'])
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Downloads
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        csv = df_lds_classified.to_csv(index=False).encode()
        st.download_button("📥 Classified LDS", csv, f"lds_classified.csv", "text/csv")
    with col2:
        if not pilferage_leaks.empty:
            csv_pilf = pilferage_leaks.to_csv(index=False).encode()
            st.download_button("📥 Pilferage Only", csv_pilf, "pilferage_events.csv", "text/csv")

else:
    st.info("👆 Upload files and click Analyze")
    st.markdown("""
    **PIDWS format:** Date, Time, chainage, Event Duration (2m 30s)
    **LDS format:** Date, Time, chainage, leak size
    """)

st.markdown("---")
st.markdown("*Indian Oil Corporation Limited | Pipeline Analytics*")
