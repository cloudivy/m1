import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import traceback

st.set_page_config(page_title="SCC Probability Visualization", layout="wide")

st.title("🔬 SCC Probability Estimation - CHAKSU MATHURA SECTION")

uploaded_file = st.file_uploader("Choose cm section.xlsx or scc_IV_dataset.xlsx", type=["xlsx"])

if uploaded_file is not None:
    try:
        plt.close('all')  # Prevent memory issues in reruns

        # 1. Load the dataset into df_scc_II
        df_scc_II = pd.read_excel(uploaded_file)
        st.subheader("📊 Original Data Preview")
        st.dataframe(df_scc_II.head())

        # 2. Normalize 'Wd (ID)' as 'Normalized_Distance_from_Pump(KM)'
        scaler_distance = MinMaxScaler()
        df_scc_II['Normalized_Distance_from_Pump(KM)'] = scaler_distance.fit_transform(df_scc_II[['Wd (ID)']])

        # 3. Normalize 'OFF PSP (VE V)'
        scaler_off_psp = MinMaxScaler()
        df_scc_II['Normalized_OFF_PSP_VE_V'] = scaler_off_psp.fit_transform(df_scc_II[['OFF PSP (VE V)']])

        # 4. Create the Inverse_Normalized_OFF_PSP_VE_V column
        df_scc_II['Inverse_Normalized_OFF_PSP_VE_V'] = 1 - df_scc_II['Normalized_OFF_PSP_VE_V']

        # 5. Define feature weights and calculate 'Stress_Corrosion_Probability_Score_Normalized_V2'
        feature_weights_normalized = {
            'conductivity': 0.186,
            'Hoop stress% of SMYS': 0.08,
            'Normalized_Distance_from_Pump(KM)': 0.165,
            'Inverse_Normalized_OFF_PSP_VE_V': 0.142
        }

        df_scc_II['Stress_Corrosion_Probability_Score_Normalized_V2'] = (
            df_scc_II['conductivity'] * feature_weights_normalized['conductivity'] +
            (df_scc_II['Hoop stress% of SMYS'] * feature_weights_normalized['Hoop stress% of SMYS']) +
            (df_scc_II['Normalized_Distance_from_Pump(KM)'] * feature_weights_normalized['Normalized_Distance_from_Pump(KM)']) +
            (df_scc_II['Inverse_Normalized_OFF_PSP_VE_V'] * feature_weights_normalized['Inverse_Normalized_OFF_PSP_VE_V'])
        )

        st.subheader("📈 Key Metrics")
        col1, col2, col3 = st.columns(3)
        high_risk_threshold_normalized = df_scc_II['Stress_Corrosion_Probability_Score_Normalized_V2'].quantile(0.95)
        with col1:
            st.metric("Mean SCC Score", f"{df_scc_II['Stress_Corrosion_Probability_Score_Normalized_V2'].mean():.4f}")
        with col2:
            st.metric("Max SCC Score", f"{df_scc_II['Stress_Corrosion_Probability_Score_Normalized_V2'].max():.4f}")
        with col3:
            st.metric("High Risk Segments (>95th %ile)", (df_scc_II['Stress_Corrosion_Probability_Score_Normalized_V2'] > high_risk_threshold_normalized).sum())

        # 6. Exact Visualization: Normalized Stress Corrosion Probability Score vs. Stationing (m)
        st.subheader("📊 Normalized Stress Corrosion Probability Score vs. Stationing (m)")
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.scatterplot(x='Stationing (m)', y='Stress_Corrosion_Probability_Score_Normalized_V2', 
                        data=df_scc_II, alpha=0.6, s=10, label='Normalized Stress Corrosion Probability Score per Stationing', ax=ax)
        ax.axhline(y=high_risk_threshold_normalized, color='red', linestyle='--', 
                   label=f'High Risk Threshold ({high_risk_threshold_normalized:.4f})')
        ax.set_title('Normalized Stress Corrosion Probability Score vs. Stationing (m) in df_scc_II for CHAKSU MATHURA SECTION')
        ax.set_xlabel('Stationing (m)')
        ax.set_ylabel('Normalized Stress Corrosion Probability Score')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        st.subheader("💾 Download Processed Data")
        csv_buffer = io.StringIO()
        df_scc_II.to_csv(csv_buffer, index=False)
        st.download_button("Download Processed CSV", csv_buffer.getvalue(), "scc_processed_chaksu_mathura.csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.code(traceback.format_exc())
else:
    st.info("👆 Please upload your 'cm section.xlsx' file to see the exact visualization from your Colab code #58.")
