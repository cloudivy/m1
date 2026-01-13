import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Streamlit page config
st.set_page_config(
    page_title="Pipeline Pilferage Detection",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛢️ Pipeline Pilferage Detection Dashboard")
st.markdown("---")

# Sidebar for file uploads and parameters
st.sidebar.header("📁 Data Upload")
uploaded_pidws = st.sidebar.file_uploader("Upload PIDWS data (df_pidws_III.xlsx)", type="xlsx")
uploaded_lds = st.sidebar.file_uploader("Upload LDS data (df_lds_III.xlsx)", type="xlsx")

st.sidebar.header("⚙️ Classification Parameters")
chainage_tol = st.sidebar.slider("Chainage Tolerance (km)", 0.1, 2.0, 0.5, 0.1)
time_window_hours = st.sidebar.slider("Time Window (hours)", 12, 72, 48, 6)

if st.sidebar.button("🔄 Analyze Data", type="primary"):
    if uploaded_pidws is not None and uploaded_lds is not None:
        with st.spinner("Processing pipeline data..."):
            # Load datasets
            df_pidws = pd.read_excel(uploaded_pidws)
            df_lds = pd.read_excel(uploaded_lds)
            
            st.session_state.df_pidws = df_pidws
            st.session_state.df_lds = df_lds
            st.session_state.chainage_tol = chainage_tol
            st.session_state.time_window_hours = time_window_hours
            st.session_state.status = "processed"
            
            st.sidebar.success("✅ Data processed!")
    else:
        st.sidebar.error("❌ Please upload both files")

# Main analysis section
if 'status' in st.session_state and st.session_state.status == "processed":
    df_pidws = st.session_state.df_pidws
    df_lds = st.session_state.df_lds
    chainage_tol = st.session_state.chainage_tol
    time_window_hours = st.session_state.time_window_hours
    
    # Parse PIDWS datetime and duration
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
                dur_str = dur_str.split('m')[1]
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
    
    # Classification function
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
        
        if classified:
            return pd.concat(classified, ignore_index=True)
        return pd.DataFrame()
    
    pilferage_leaks = classify_pilferage(df_pidws, df_lds, chainage_tol, time_window_hours)
    
    # Add classification to LDS
    df_lds_classified = df_lds.copy()
    df_lds_classified['is_pilferage'] = False
    
    if not pilferage_leaks.empty:
        pilferage_ids = pilferage_leaks[['DateTime', 'chainage']].drop_duplicates()
        mask_pilferage = df_lds_classified.set_index(['DateTime', 'chainage']).index.isin(
            pilferage_ids.set_index(['DateTime', 'chainage']).index
        )
        df_lds_classified.loc[mask_pilferage, 'is_pilferage'] = True
    
    st.session_state.pilferage_leaks = pilferage_leaks
    st.session_state.df_lds_classified = df_lds_classified
    
    # === DASHBOARD ===
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total LDS Events", len(df_lds))
        st.metric("PIDWS Events", len(df_pidws))
        st.metric("🟡 Pilferage Events", len(pilferage_leaks))
        pilferage_pct = (len(pilferage_leaks)/len(df_lds)*100) if len(df_lds)>0 else 0
        st.metric("Pilferage Rate", f"{pilferage_pct:.1f}%")
    
    with col2:
        col_a, col_b = st.columns(2)
        with col_a:
            avg_leak_size = df_lds_classified[df_lds_classified['is_pilferage']]['leak size'].mean()
            st.metric("Avg Pilferage Leak Size", f"{avg_leak_size:.1f}")
        with col_b:
            avg_other_size = df_lds_classified[~df_lds_classified['is_pilferage']]['leak size'].mean()
            st.metric("Avg Other Leak Size", f"{avg_other_size:.1f}")
    
    st.markdown("---")
    
    # Chainage Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Chainage Distribution")
        fig1 = px.histogram(
            df_pidws, x='chainage', opacity=0.7, 
            labels={'chainage': 'Chainage (km)'},
            title="PIDWS (Orange) vs LDS Leaks (Blue)",
            color_discrete_sequence=['orange', 'blue']
        )
        if not pilferage_leaks.empty:
            fig1.add_vline(
                x=pilferage_leaks['linked_chainage'].mean(),
                line_dash="dash", line_color="red",
                annotation_text=f"Pilferage Mean: {pilferage_leaks['linked_chainage'].mean():.1f}km"
            )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("📈 Classification Summary")
        summary_df = df_lds_classified['is_pilferage'].value_counts().reset_index()
        summary_df['is_pilferage'] = summary_df['is_pilferage'].map({True: 'Pilferage', False: 'Other'})
        fig2 = px.pie(summary_df, values='count', names='is_pilferage', 
                      title="LDS Events Classification")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Comprehensive visualization
    st.subheader("🎯 Event Timeline & Patterns")
    
    all_events = pd.concat([
        df_pidws[['DateTime', 'chainage']].assign(type='Digging', leak_size=0),
        df_lds[['DateTime', 'chainage', 'leak size']].assign(type='Leak'),
        pilferage_leaks[['DateTime', 'linked_chainage', 'leak size']].rename(
            columns={'linked_chainage':'chainage'}).assign(type='Pilferage')
    ], ignore_index=True)
    
    # Time series
    fig_time = px.line(
        all_events.groupby([all_events['DateTime'].dt.floor('H'), 'type']).size().reset_index(),
        x='DateTime', y=0, color='type',
        title="Events per Hour",
        labels={'0': 'Count', 'DateTime': 'Time'}
    )
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Leak size comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig_size = px.box(
            df_lds_classified, x='is_pilferage', y='leak size',
            color='is_pilferage',
            title="Leak Size Distribution",
            labels={'is_pilferage': 'Classification', 'leak size': 'Leak Size'}
        )
        st.plotly_chart(fig_size, use_container_width=True)
    
    with col2:
        if not pilferage_leaks.empty:
            st.subheader("🔥 Top Pilferage Clusters")
            clusters = pilferage_leaks.groupby('linked_chainage')['leak size'].agg(['count', 'mean', 'max']).round(1)
            st.dataframe(clusters.sort_values('count', ascending=False).head(10))
    
    # Spatio-temporal scatter
    st.subheader("🗺️ Spatio-Temporal Leak Map")
    colors = ['red' if x else 'blue' for x in df_lds_classified['is_pilferage']]
    fig_scatter = px.scatter(
        df_lds_classified, x='DateTime', y='chainage', color=df_lds_classified['is_pilferage'],
        color_discrete_map={True: 'red', False: 'blue'},
        title="Red=Pilferage, Blue=Other Leaks",
        hover_data=['leak size']
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Download section
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    csv_buffer = io.StringIO()
    df_lds_classified.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue().encode()
    
    with col1:
        st.download_button(
            label="📥 Download Classified LDS",
            data=csv_data,
            file_name=f"lds_classified_{chainage_tol}km_{time_window_hours}h.csv",
            mime="text/csv"
        )
    
    with col2:
        if not pilferage_leaks.empty:
            pilferage_csv = pilferage_leaks.to_csv(index=False).encode()
            st.download_button(
                label="📥 Download Pilferage Events",
                data=pilferage_csv,
                file_name="pilferage_events.csv",
                mime="text/csv"
            )
    
    with col3:
        st.info(f"**Parameters Used:** Chainage Tol: {chainage_tol}km, Time Window: {time_window_hours}h")

else:
    st.info("👆 Please upload your PIDWS and LDS Excel files in the sidebar and click 'Analyze Data'")
    
    st.markdown("""
    ### 📋 Required File Format
    
    **df_pidws_III.xlsx columns:**
    - Date (dd-mm-yyyy)
    - Time (HH:MM:SS) 
    - chainage (km)
    - Event Duration (Xm Ys format)
    
    **df_lds_III.xlsx columns:**
    - Date
    - Time
    - chainage (km)
    - leak size
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Developed for Indian Oil Corporation Limited | Pipeline Operations Analytics
    </div>
    """, 
    unsafe_allow_html=True
)
