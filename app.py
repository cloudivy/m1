import streamlit as st
import pandas as pd
import tsfel
from sklearn.preprocessing import StandardScaler
import numpy as np
from ts2vec import TS2Vec
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("TS2Vec + TSFEL Anomaly Detection")

uploaded = st.file_uploader("Upload sensor CSV (with columns KAN, BV5, BV4, BV3, BV2, BV1, VIR)", type=["csv"])

if uploaded is not None:
    df3 = pd.read_csv(uploaded)

    sensor_cols = ['KAN', 'BV5', 'BV4', 'BV3', 'BV2', 'BV1', 'VIR']

    # basic check
    missing = [c for c in sensor_cols if c not in df3.columns]
    if missing:
        st.error(f"Missing columns in file: {missing}")
    else:
        # preprocessing
        for col in sensor_cols:
            df3[col] = pd.to_numeric(df3[col], errors='coerce')
            df3[col] = df3[col].fillna(df3[col].mean())

        X = df3[sensor_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        cfg = tsfel.get_features_by_domain()

        feature_matrix = None
        for idx, col in enumerate(sensor_cols):
            feats = tsfel.time_series_features_extractor(cfg, X_scaled[:, idx])
            if feature_matrix is None:
                feature_matrix = feats
            else:
                feature_matrix = pd.concat([feature_matrix, feats], axis=1)

        def to_windows(arr, window=64, step=16):
            return np.array([arr[i:i+window] for i in range(0, len(arr)-window+1, step)])

        X_windows = to_windows(X, window=64, step=16)

        model = TS2Vec(input_dims=len(sensor_cols), device='cpu')
        with st.spinner("Training TS2Vec model..."):
            model.fit(X_windows)

        seq_embeds = model.encode(X_windows)
        seq_embeds_averaged = np.mean(seq_embeds, axis=1)

        anomaly_detector_tsfel = IsolationForest(contamination=0.05, random_state=42)
        anomaly_labels_tsfel = anomaly_detector_tsfel.fit_predict(feature_matrix)
        anomalous_indices_tsfel = np.where(anomaly_labels_tsfel == -1)[0]

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_ts2vec = kmeans.fit_predict(seq_embeds_averaged)
        distances_ts2vec = np.linalg.norm(
            seq_embeds_averaged - kmeans.cluster_centers_[labels_ts2vec],
            axis=1
        )
        threshold_ts2vec = np.percentile(distances_ts2vec, 95)
        anomalous_windows_ts2vec = np.where(distances_ts2vec > threshold_ts2vec)[0]

        st.subheader("Detected anomalies")
        st.write("TSFEL anomalies at indices:", anomalous_indices_tsfel.tolist())
        st.write("TS2Vec anomalous windows:", anomalous_windows_ts2vec.tolist())

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(distances_ts2vec, label="Anomaly Score (TS2Vec)")
        ax.scatter(
            anomalous_windows_ts2vec,
            distances_ts2vec[anomalous_windows_ts2vec],
            color="red",
            label="Anomalies (TS2Vec)"
        )
        ax.set_xlabel("Window index")
        ax.set_ylabel("Anomaly score (distance)")
        ax.set_title("TS2Vec Embedding Anomaly Detection")
        ax.legend()

        st.pyplot(fig)
