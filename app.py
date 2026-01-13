import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings
warnings.filterwarnings('ignore')

# Plot config
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150

st.set_page_config(page_title="Pipeline Pilferage", layout="wide")

st.title("🛢️ Pipeline Pilferage Detection")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📁 Upload Data")
    pidws_file = st.file_uploader("PIDWS (xlsx)", type="xlsx")
    lds_file = st.file_uploader("LDS (xlsx)", type="xlsx")
    
    st.header("⚙️ Parameters")
    chainage_tol = st.slider("Chainage Tol (km)", 0.1, 2.0, 0.5)
    time_window = st.slider("Time Window (hrs)", 12, 72, 48)
    
    if st.button("🚀 ANALYZE", type="primary") and pidws_file and lds_file:
        with st.spinner("Analyzing..."):
            st.session_state.processed = True
            st.session_state.pidws = pd.read_excel(pidws_file)
            st.session_state.lds = pd.read_excel(lds_file)
            st.session_state.params = (chainage_tol, time_window)
        st.rerun()

if 'processed' in st.session_state:
    # Process data
    df_pidws = st.session_state.pidws
    df_lds = st.session_state.lds
    tol, window = st.session_state.params
    
    # Preprocessing functions
    @st.cache_data
    def clean_pidws(df):
        df = df.copy()
        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        df['duration_td'] = df['Event Duration'].astype(str).str.extract('(\d+)m?').fillna(0).astype(int)
        df['end_time'] = df['DateTime'] + pd.to_timedelta(df['duration_td'], unit='m')
        return df
    
    @st.cache_data
    def clean_lds(df):
        df = df.copy()
        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        return df
    
    df_pidws = clean_pidws(df_pidws)
    df_lds = clean_lds(df_lds)
    
    # Classification
    @st.cache_data
    def detect_pilferage(pidws, lds, tol, window):
        results = []
        for _, row in pidws.iterrows():
            end = row['end_time'] + pd.Timedelta(hours=window)
            mask = (lds['DateTime'] > end) & (np.abs(lds['chainage'] - row['chainage']) <= tol)
            matches = lds[mask].copy()
            if not matches.empty:
                matches['pilferage_score'] = 1 / (1 + (matches['DateTime'] - end).dt.total_seconds()/3600)
                results.append(matches)
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    pilferage = detect_pilferage(df_pidws, df_lds, tol, window)
    
    # Classify full dataset
    df_lds['is_pilferage'] = df_lds.index.isin(pilferage.index) if not pilferage.empty else False
    
    # === DASHBOARD ===
    c1, c2 = st.columns(2)
    with c1:
        st.metric("LDS Events", len(df_lds))
        st.metric("PIDWS Events", len(df_pidws))
        st.metric("🟡 Pilferage", len(pilferage))
    
    with c2:
        pct = len(pilferage)/len(df_lds)*100 if len(df_lds) else 0
        st.metric("Detection Rate", f"{pct:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(df_pidws.chainage, alpha=0.6, label='PIDWS', color='orange')
        ax.hist(df_lds.chainage, alpha=0.6, label='Leaks', color='blue')
        if len(pilferage):
            ax.axvline(pilferage.chainage.mean(), color='red', ls='--', label='Pilferage')
        ax.legend()
        ax.set_xlabel('Chainage (km)')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6,5))
        df_lds.boxplot(column='leak size', by='is_pilferage', ax=ax)
        ax.set_title('Leak Size by Type')
        st.pyplot(fig)
    
    # Timeline
    st.subheader("Timeline")
    fig, ax = plt.subplots(figsize=(12,6))
    colors = ['red' if x else 'blue' for x in df_lds.is_pilferage]
    sizes = np.clip(df_lds['leak size'], 30, 150)
    ax.scatter(df_lds.DateTime, df_lds.chainage, c=colors, s=sizes, alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Chainage')
    ax.grid(alpha=0.3)
    ax.legend(['Pilferage', 'Other'])
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Table
    if len(pilferage):
        st.subheader("Top Pilferage Locations")
        top = pilferage.groupby('chainage')['leak size'].agg(['count','mean']).round(2).sort_values('count', ascending=False).head()
        st.dataframe(top)
    
    # Download
    csv = df_lds.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results", csv, "pilferage_analysis.csv")

else:
    st.info("👆 Upload Excel files & click ANALYZE")
