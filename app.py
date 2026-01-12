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
    page_title="Pipeline Pilferage Classification",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛢️ Pipeline Pilferage Detection & Classification")
st.markdown("**LLM-Enhanced Leak Detection Analysis for IOCL Pipelines**")

# Sidebar for file uploads and parameters
st.sidebar.header("📁 Data Upload")
uploaded_pidws = st.sidebar.file_uploader("PIDWS Data (df_pidws_III.xlsx)", type="xlsx")
uploaded_lds = st.sidebar.file_uploader("LDS Data (df_lds_III.xlsx)", type="xlsx")

st.sidebar.header("⚙️ Classification Parameters")
chainage_tol = st.sidebar.slider("Chainage Tolerance (km)", 0.1, 2.0, 0.5, 0.1)
time_window = st.sidebar.slider("Time Window (hours)", 12, 72, 48, 6)

if uploaded_pidws is not None and uploaded_lds is not None:
    # Load data
    @st.cache_data
    def load_data(pidws_file, lds_file):
        df_pidws = pd.read_excel(pidws_file)
        df_lds = pd.read_excel(lds_file)
        return df_pidws, df_lds
    
    df_pidws, df_lds = load_data(uploaded_pidws, uploaded_lds)
    
    st.success(f"✅ Loaded {len(df_pidws)} PIDWS events and {len(df_lds)} LDS leaks")
    
    # Data preprocessing functions
    @st.cache_data
    def preprocess_data(df_pidws, df_lds):
        # Parse PIDWS datetime and duration
        df_pidws['DateTime'] = pd.to_datetime(df_pidws['Date'] + ' ' + df_pidws['Time'], 
                                            format='%d-%m-%Y %H:%M:%S')
        
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
        
        df_pidws['duration_td'] = df_pidws['Event Duration'].apply(parse_duration)
        df_pidws['end_time'] = df_pidws['DateTime'] + df_pidws['duration_td']
        
        # Parse LDS datetime
        df_lds['DateTime'] = pd.to_datetime(df_lds['Date'].astype(str) + ' ' + df_lds['Time'])
        return df_pidws, df_lds
    
    df_pidws, df_lds = preprocess_data(df_pidws, df_lds)
    
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
    
    # Run classification
    with st.spinner("🔍 Classifying pilferage events..."):
        pilferage_leaks = classify_pilferage(df_pidws, df_lds, chainage_tol, time_window)
    
    # Create classified LDS dataframe
    df_lds_classified = df_lds.copy()
    df_lds_classified['is_pilferage'] = False
    
    if not pilferage_leaks.empty:
        pilferage_ids = pilferage_leaks[['DateTime', 'chainage']].drop_duplicates()
        mask_pilferage = df_lds_classified.set_index(['DateTime', 'chainage']).index.isin(
            pilferage_ids.set_index(['DateTime', 'chainage']).index
        )
        df_lds_classified.loc[mask_pilferage, 'is_pilferage'] = True
    
    # Metrics display
    col1, col2, col3, col4 = st.columns(4)
    total_leaks = len(df_lds)
    pilferage_count = df_lds_classified['is_pilferage'].sum()
    pilferage_rate = (pilferage_count / total_leaks * 100) if total_leaks > 0 else 0
    
    with col1:
        st.metric("Total LDS Leaks", total_leaks)
    with col2:
        st.metric("Pilferage Events", pilferage_count)
    with col3:
        st.metric("Pilferage Rate", f"{pilferage_rate:.1f}%")
    with col4:
        st.metric("Avg Pilferage Score", 
                 f"{pilferage_leaks['pilferage_score'].mean():.3f}" if not pilferage_leaks.empty else "0")
    
    # Results table
    st.subheader("📊 Classification Results")
    
    if not pilferage_leaks.empty:
        st.dataframe(pilferage_leaks[['DateTime', 'chainage', 'leak size', 'pilferage_score', 
                                    'linked_chainage', 'linked_event_time']].round(2),
                    use_container_width=True)
    else:
        st.warning("No pilferage events detected with current parameters")
    
    # Visualizations
    st.subheader("📈 Analysis Dashboard")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Chainage Distribution', 'Temporal Patterns', 
                       'Leak Size Distribution', 'Spatiotemporal View'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Chainage distribution
    fig.add_trace(
        go.Histogram(x=df_pidws['chainage'], name="PIDWS (Digging)", 
                    marker_color='orange', opacity=0.7, nbinsx=30),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=df_lds['chainage'], name="All LDS Leaks", 
                    marker_color='lightblue', opacity=0.7, nbinsx=30),
        row=1, col=1
    )
    if not pilferage_leaks.empty:
        fig.add_vline(x=pilferage_leaks['linked_chainage'].mean(), 
                     line_dash="dash", line_color="red", 
                     annotation_text="Pilferage Mean", row=1, col=1)
    
    # 2. Temporal patterns
    all_events = pd.concat([
        df_pidws[['DateTime', 'chainage']].assign(type='Digging'),
        df_lds[['DateTime', 'chainage']].assign(type='Leak'),
        pilferage_leaks[['DateTime', 'linked_chainage']].rename(
            columns={'linked_chainage':'chainage'}).assign(type='Pilferage')
    ], ignore_index=True)
    
    time_counts = all_events.groupby([all_events['DateTime'].dt.floor('H'), 'type']).size().unstack(fill_value=0)
    for col in time_counts.columns:
        fig.add_trace(
            go.Scatter(x=time_counts.index, y=time_counts[col], name=col, mode='lines'),
            row=1, col=2
        )
    
    # 3. Leak size distribution
    fig.add_trace(
        go.Box(y=df_lds_classified[df_lds_classified['is_pilferage']==False]['leak size'], 
              name="Non-Pilferage", marker_color='blue'),
        row=2, col=1
    )
    if pilferage_count > 0:
        fig.add_trace(
            go.Box(y=df_lds_classified[df_lds_classified['is_pilferage']==True]['leak size'], 
                  name="Pilferage", marker_color='red'),
            row=2, col=1
        )
    
    # 4. Spatiotemporal scatter
    colors = ['red' if x else 'blue' for x in df_lds_classified['is_pilferage']]
    fig.add_trace(
        go.Scatter(x=df_lds_classified['DateTime'], y=df_lds_classified['chainage'],
                  mode='markers', marker=dict(color=colors, size=8, opacity=0.6),
                  name='Leaks (Red=Pilferage)'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Pipeline Analysis Dashboard")
    st.plotly_chart(fig, use_container_width=True)
    
    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        csv_buffer = io.StringIO()
        df_lds_classified.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Classified LDS (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"lds_classified_chainage_{chainage_tol}_time_{time_window}.csv",
            mime="text/csv"
        )
    
    with col2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df_lds_classified.to_excel(writer, sheet_name='Classified_LDS', index=False)
            pilferage_leaks.to_excel(writer, sheet_name='Pilferage_Details', index=False)
        st.download_button(
            label="📥 Download Full Analysis (Excel)",
            data=excel_buffer.getvalue(),
            file_name="pipeline_pilferage_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # Summary statistics
    st.subheader("📋 Detailed Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Top Chainage Cluster", 
                 f"{pilferage_leaks['linked_chainage'].mode().iloc[0] if not pilferage_leaks.empty else 'N/A':.1f} km")
        st.metric("Max Leak Size (Pilferage)", 
                 f"{pilferage_leaks['leak size'].max():.2f}" if not pilferage_leaks.empty else "N/A")
    
    with col2:
        if not pilferage_leaks.empty:
            st.dataframe(pilferage_leaks.groupby('linked_chainage')['leak size']
                        .agg(['count', 'mean', 'max']).round(1).head(),
                        use_container_width=True)

else:
    st.info("👆 Please upload both PIDWS and LDS Excel files to begin analysis")
    st.markdown("""
    ### Expected File Formats:
    **df_pidws_III.xlsx columns:**
    - Date, Time, chainage, Event Duration
    
    **df_lds_III.xlsx columns:**
    - Date, Time, chainage, leak size
    """)
