import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, Dict

logger = logging.getLogger("ShopperSpectrumPipeline.Clustering")

class ClusteringModel:
    @staticmethod
    def train(rfm_df: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, StandardScaler, KMeans, Dict[int, str]]:
        logger.info(f"Training K-Means model with {n_clusters} clusters...")
        
        rfm_base = rfm_df[["Recency", "Frequency", "Monetary"]]
        rfm_log = np.log1p(rfm_base)
        
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_log)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm_df["Cluster"] = kmeans.fit_predict(rfm_scaled)
        
        sil = silhouette_score(rfm_scaled, rfm_df["Cluster"], sample_size=500, random_state=42)
        logger.info(f"K-Means trained. Inertia (Loss): {kmeans.inertia_:.2f} | Silhouette: {sil:.4f}")
        
        cluster_profiles = rfm_df.groupby("Cluster").mean()
        sorted_by_monetary = cluster_profiles.sort_values(by="Monetary")
        monetary_ranks = sorted_by_monetary.index.tolist()
        
        cluster_mapping = {}
        cluster_mapping[monetary_ranks[3]] = "High-Value"
        cluster_mapping[monetary_ranks[2]] = "Regular"
        
        if cluster_profiles.loc[monetary_ranks[0], "Recency"] > cluster_profiles.loc[monetary_ranks[1], "Recency"]:
            cluster_mapping[monetary_ranks[0]] = "At-Risk"
            cluster_mapping[monetary_ranks[1]] = "Occasional"
        else:
            cluster_mapping[monetary_ranks[0]] = "Occasional"
            cluster_mapping[monetary_ranks[1]] = "At-Risk"
            
        logger.info(f"Dynamic Cluster Mapping identified: {cluster_mapping}")
        rfm_df["Segment"] = rfm_df["Cluster"].map(cluster_mapping)
        
        return rfm_df, scaler, kmeans, cluster_mapping
