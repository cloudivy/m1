import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="LDS-PIDWS Correlation", layout="wide", page_icon="📊")

st.title("🛢️ LDS-PIDWS Pilferage Detection Dashboard")
st.markdown("Interactive analysis of Leak Detection System (LDS) and Pipeline Integrity Data Warehouse System (PIDWS) correlation for pilferage classification.")

# File upload
uploaded_pidws = st.file_uploader("Upload PIDWS data (dfpidwsIII.xlsx)", type="xlsx", key="pidws")
uploaded_lds = st.file_uploader("Upload LDS data (dfldsIII.xlsx)", type="xlsx", key="lds")

if uploaded_pidws is not None and uploaded_lds is not None:
    # Load data
    dfpidws = pd.read_excel(uploaded_pidws)
    dflds = pd.read_excel(uploaded_lds)
    
    st.success(f"✅ Loaded PIDWS: {len(dfpidws)} rows | LDS: {len(dflds)} rows")
    
    # Sidebar parameters
    st.sidebar.header("Classification Parameters")
    chainage_tol = st.sidebar.slider("Chainage Tolerance (km)", 0.1, 2.0, 0.5, 0.1)
    time_window = st.sidebar.slider("Time Window (hours)", 12, 72, 48, 6)
    
    # Parse datetime PIDWS
    @st.cache_data
    def parse_datetime(df):
        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'], format='%d-%m-%Y %H:%M:%S')
        return df
    
    dfpidws = parse_datetime(dfpidws)
    dflds = parse_datetime(dflds)
    
    # Parse duration
    @st.cache_data
    def parse_duration(dur_str):
        if pd.isna(dur_str):
            return pd.Timedelta(0)
        dur_str = str(dur_str).strip().lower().replace(' ', '')
        mins, secs = 0, 0
        if 'm' in dur_str:
            m_part = dur_str.split('m')[0]
            if m_part.isdigit():
                mins = int(m_part)
            dur_str = dur_str.split('m')[1]
        if 's' in dur_str:
            s_part = dur_str.replace('s', '')
            if s_part.isdigit():
                secs = int(s_part)
        return pd.Timedelta(minutes=mins, seconds=secs)
    
    dfpidws['duration_td'] = dfpidws['Event Duration'].apply(parse_duration)
    dfpidws['end_time'] = dfpidws['DateTime'] + dfpidws['duration_td']
    
    # Classification function
    @st.cache_data
    def classify_pilferage(pidws_df, lds_df, chainage_tol, time_window_hours):
        classified = []
        for _, event in pidws_df.iterrows():
            window_end = event['end_time'] + pd.Timedelta(hours=time_window_hours)
            mask = (lds_df['DateTime'] > event['end_time']) & \
                   (lds_df['DateTime'] <= window_end) & \
                   (np.abs(lds_df['chainage'] - event['chainage']) <= chainage_tol)
            matches = lds_df[mask].copy()
            if not matches.empty:
                matches['linked_event_time'] = event['DateTime']
                matches['linked_chainage'] = event['chainage']
                matches['pilferage_score'] = 1 * (1 - (matches['DateTime'] - event['end_time']).dt.total_seconds() / 3600 / time_window_hours)
                classified.append(matches)
        if classified:
            return pd.concat(classified, ignore_index=True)
        return pd.DataFrame()
    
    pilferage_leaks = classify_pilferage(dfpidws, dflds, chainage_tol, time_window)
    
    # Add classification to LDS
    dflds_classified = dflds.copy()
    dflds_classified['is_pilferage'] = False
    if not pilferage_leaks.empty:
        pilferage_ids = pilferage_leaks[['DateTime', 'chainage']].drop_duplicates()
        mask_pilferage = dflds_classified.set_index(['DateTime', 'chainage']).index.isin(
            pilferage_ids.set_index(['DateTime', 'chainage']).index
        )
        dflds_classified.loc[mask_pilferage, 'is_pilferage'] = True
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    total_lds = len(dflds)
    pilferage_count = dflds_classified['is_pilferage'].sum()
    pilferage_rate = (pilferage_count / total_lds * 100) if total_lds > 0 else 0
    col1.metric("Total LDS Events", total_lds)
    col2.metric("Pilferage Events", pilferage_count)
    col3.metric("Pilferage Rate", f"{pilferage_rate:.1f}%")
    col4.metric("Chainage Clusters", len(pilferage_leaks['linked_chainage'].value_counts()) if not pilferage_leaks.empty else 0)
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🗺️ Chainage Dist.", "⏱️ Time Series", "📊 Leak Analysis"])
    
    with tab1:
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Chainage Distribution", "Temporal Pattern", "Leak Size vs Classification", "Chainage Scatter"))
        
        # Chainage hist
        fig.add_trace(go.Histogram(x=dfpidws['chainage'], name="PIDWS (Digging)", nbinsx=30, opacity=0.7), row=1, col=1)
        fig.add_trace(go.Histogram(x=dflds['chainage'], name="LDS Leaks", nbinsx=30, opacity=0.7), row=1, col=1)
        if not pilferage_leaks.empty:
            fig.add_vline(x=pilferage_leaks['linked_chainage'].mean(), line_dash="dash", line_color="red", row=1, col=1)
        
        # Time series
        all_events = pd.concat([
            dfpidws[['DateTime', 'chainage']].assign(type='Digging'),
            dflds[['DateTime', 'chainage']].assign(type='Leak'),
            pilferage_leaks[['DateTime', 'linked_chainage']].rename(columns={'linked_chainage': 'chainage'}).assign(type='Pilferage')
        ], ignore_index=True)
        time_pivot = all_events.groupby('DateTime')['type'].value_counts().unstack(fill_value=0)
        for col in time_pivot.columns:
            fig.add_trace(go.Scatter(x=time_pivot.index, y=time_pivot[col], mode='lines', name=col), row=1, col=2)
        
        # Boxplot leak size
        fig.add_trace(go.Box(y=dflds_classified[dflds_classified['is_pilferage']]['leak size'], name="Pilferage"), row=2, col=1)
        fig.add_trace(go.Box(y=dflds_classified[~dflds_classified['is_pilferage']]['leak size'], name="Other"), row=2, col=1)
        
        # Scatter
        colors = ['red' if x else 'blue' for x in dflds_classified['is_pilferage']]
        fig.add_trace(go.Scatter(x=dflds_classified['DateTime'], y=dflds_classified['chainage'], mode='markers', marker=dict(color=colors, size=8, opacity=0.6), name="Leaks"), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if not pilferage_leaks.empty:
            chainage_stats = pilferage_leaks.groupby('linked_chainage')['leak size'].agg(['count', 'mean', 'max']).round(1).sort_values('count', ascending=False).head(10)
            st.subheader("Top Chainage Clusters")
            st.dataframe(chainage_stats)
            
            fig_hist = px.histogram(pilferage_leaks, x='linked_chainage', nbins=50, title="Pilferage Chainage Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab3:
        if not pilferage_leaks.empty:
            fig_time = px.line(time_pivot, title="Events per Hour", markers=True)
            st.plotly_chart(fig_time, use_container_width=True)
    
    with tab4:
        st.subheader("Classification Summary")
        st.dataframe(dflds_classified['is_pilferage'].value_counts())
        
        if not pilferage_leaks.empty:
            st.subheader("Pilferage Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Leak Size", f"{pilferage_leaks['leak size'].mean():.0f}")
            with col2:
                st.metric("Avg Pilferage Score", f"{pilferage_leaks['pilferage_score'].mean():.3f}")
            
            fig_box = px.box(dflds_classified, x='is_pilferage', y='leak size', title="Leak Size: Pilferage vs Others")
            st.plotly_chart(fig_box)
    
    # Download classified data
    csv = dflds_classified.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Classified LDS Data", csv, "lds_classified.csv", "text/csv")

else:
    st.info("👆 Please upload both PIDWS and LDS Excel files to get started.")

# Footer
st.markdown("---")
st.markdown("Built for pipeline operations analysis | Powered by Streamlit")[file:1]
