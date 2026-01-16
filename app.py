import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from io import BytesIO

# Configure matplotlib to prevent memory warnings
plt.rcParams['figure.max_open_warning'] = 50

st.set_page_config(page_title="Leak Detection Dashboard", layout="wide")
st.title("🔍 Identifying and implementing newer methods of quick detection of location of leaks in the mainline ")

# --- Enhanced data loading (SAFE - works without repo files) ---
tab1, tab2 = st.tabs(["📁 Upload Files", "📊 Analysis"])

with tab1:
    uploaded_files = st.file_uploader(
        "Upload Excel files:",
        type=['xlsx'], accept_multiple_files=True,
        help="Upload: df_pidws.xlsx, df_manual_digging.xlsx, df_lds_IV.xlsx"
    )
    
    if uploaded_files:
        file_dict = {f.name: f for f in uploaded_files}
        st.success(f"✅ Uploaded {len(uploaded_files)} file(s)")
        
        # Load dataframes safely
        try:
            df_pidws = pd.read_excel(file_dict['df_pidws.xlsx']) if 'df_pidws.xlsx' in file_dict else pd.DataFrame()
            df_manual_digging = pd.read_excel(file_dict['df_manual_digging.xlsx']) if 'df_manual_digging.xlsx' in file_dict else pd.DataFrame()
            df_lds_IV = pd.read_excel(file_dict['df_lds_IV.xlsx']) if 'df_lds_IV.xlsx' in file_dict else pd.DataFrame()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Digging Data", f"{len(df_manual_digging):,}", delta="✅")
            col2.metric("Leak Data", f"{len(df_lds_IV):,}", delta="✅")
            col3.metric("PIDWS Data", f"{len(df_pidws):,}", delta="✅")
            
        except Exception as e:
            st.error(f"❌ Error reading files: {e}")
            st.stop()
    else:
        st.info("👆 Upload your Excel files to start analysis")
        st.stop()

# Switch to analysis tab for inputs
with tab2:
    st.write("Imported matplotlib.pyplot as plt and seaborn as sns.")
    
    # Data previews
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Manual Digging Data")
        st.dataframe(df_manual_digging[['Event Duration', 'Original_chainage', 'DateTime']].head())
    
    with col2:
        st.subheader("LDS IV Data") 
        st.dataframe(df_lds_IV.head())
    
    # Data prep (your exact code)
    df_manual_digging['Date_new'] = df_manual_digging['DateTime'].dt.date
    df_manual_digging['Time_new'] = df_manual_digging['DateTime'].dt.time
    
    # LDS IV prep (your exact code)
    df_lds_IV['Date'] = pd.to_datetime(df_lds_IV['Date'])
    df_lds_IV['Time'] = pd.to_timedelta(df_lds_IV['Time'].astype(str))
    df_lds_IV['DateTime'] = df_lds_IV['Date'] + df_lds_IV['Time']
    
    st.success("✅ Data preprocessing complete")
    
    # Interactive inputs
    col1, col2 = st.columns(2)
    with col1:
        target_chainage = st.number_input(
            "🎯 Target Chainage (km):",
            value=25.4, min_value=0.0, step=0.1
        )
    with col2:
        tolerance = st.number_input(
            "Tolerance (±km):", value=1.0, min_value=0.1, step=0.1
        )
    
    if st.button("🚀 Analyze Chainage", type="primary"):
        # Filter data (your exact logic)
        df_digging_filtered = df_manual_digging[
            abs(df_manual_digging['Original_chainage'] - target_chainage) <= tolerance
        ].copy()
        
        df_leaks_filtered = df_lds_IV[
            abs(df_lds_IV['chainage'] - target_chainage) <= tolerance
        ].copy()
        
        # Metrics
        col1, col2 = st.columns(2)
        col1.metric("🔵 Digging Events", len(df_digging_filtered))
        col2.metric("🔴 Leak Events", len(df_leaks_filtered))
        
        # Main scatter plot (your EXACT visualization)
        plt.close('all')  # Clear previous figures
        fig, ax = plt.subplots(figsize=(16, 10))
        
        if not df_digging_filtered.empty:
            sns.scatterplot(
                data=df_digging_filtered, x='DateTime', y='Original_chainage',
                color='blue', label='Digging Events', marker='o', s=60, ax=ax
            )
        
        if not df_leaks_filtered.empty:
            sns.scatterplot(
                data=df_leaks_filtered, x='DateTime', y='chainage',
                color='red', label='Leak Events', marker='X', s=100, ax=ax
            )
        
        plt.title(f'Digging vs Leak Events - Chainage {target_chainage} ±{tolerance}km', fontsize=16, pad=20)
        plt.xlabel('DateTime', fontsize=12)
        plt.ylabel('Chainage (km)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        
        st.pyplot(fig)
        st.balloons()

# --- All chainages analysis (optional, collapsible) ---
if st.checkbox("🔄 Show ALL Unique Chainages Analysis"):
    st.subheader("📈 Complete Chainage Analysis")
    
    unique_chainages = sorted(df_manual_digging['Original_chainage'].unique())
    st.write(f"Found {len(unique_chainages)} unique chainages")
    
    tolerance_all = st.slider("Analysis Tolerance", 0.1, 2.0, 1.0)
    
    for chainage in unique_chainages[:20]:  # Limit to 20 to avoid memory issues
        with st.expander(f"Chainage {chainage:.1f}"):
            df_dig_loop = df_manual_digging[
                abs(df_manual_digging['Original_chainage'] - chainage) <= tolerance_all
            ]
            df_leak_loop = df_lds_IV[
                abs(df_lds_IV['chainage'] - chainage) <= tolerance_all
            ]
            
            col1, col2 = st.columns(2)
            col1.metric("Digging", len(df_dig_loop))
            col2.metric("Leaks", len(df_leak_loop))
            
            if len(df_dig_loop) + len(df_leak_loop) > 0:
                plt.close('all')
                fig_loop, ax_loop = plt.subplots(figsize=(14, 7))
                
                if not df_dig_loop.empty:
                    sns.scatterplot(data=df_dig_loop, x='DateTime', y='Original_chainage',
                                  color='blue', marker='o', s=50, ax=ax_loop)
                if not df_leak_loop.empty:
                    sns.scatterplot(data=df_leak_loop, x='DateTime', y='chainage',
                                  color='red', marker='X', s=80, ax=ax_loop)
                
                plt.title(f'Chainage {chainage:.1f} ±{tolerance_all}')
                plt.tight_layout()
                st.pyplot(fig_loop)

st.markdown("---")
st.caption("💡 Upload files → Adjust chainage → Click Analyze → View correlations!")
