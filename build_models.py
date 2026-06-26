import pandas as pd
import numpy as np
import logging
import pickle
import os
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ShopperSpectrumPipeline")


class ECommerceDataPipeline:
    """End-to-end professional data pipeline for RFM clustering and product recommendations."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw_df = None
        self.cleaned_df = None
        self.df_purchases_only = None
        self.pre_cancellation_df = None
        self.rfm_df = None
        self.scaler = None
        self.kmeans = None
        self.cluster_mapping = None
        self.recommendations = None
        self.association_rules = None

    def load_data(self) -> pd.DataFrame:
        """Loads dataset from the specified CSV file with fallback encoding."""
        logger.info(f"Loading raw dataset from {self.filepath}...")
        try:
            self.raw_df = pd.read_csv(self.filepath, encoding="ISO-8859-1")
            logger.info(f"Loaded dataset successfully. Initial shape: {self.raw_df.shape}")
            return self.raw_df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def clean_data(self) -> pd.DataFrame:
        """Applies professional data cleansing strategies: deduplication, handling missing data,
        handling returns/cancellations, and removing administrative transaction codes.
        """
        if self.raw_df is None:
            self.load_data()

        logger.info("Starting data cleaning process...")
        df = self.raw_df.copy()

        # 1. Remove exact duplicate records
        before_dup = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {before_dup - len(df)} duplicate records.")

        # 2. Handle missing CustomerIDs
        before_null_cust = len(df)
        df = df.dropna(subset=["CustomerID"])
        df["CustomerID"] = df["CustomerID"].astype(int)
        logger.info(f"Removed {before_null_cust - len(df)} rows with missing CustomerID.")

        # Keep a copy of the dataset BEFORE cancellation filtering (but after basic ID/dup filtering)
        # to calculate the supplemental ReturnRatio metric.
        self.pre_cancellation_df = df.copy()

        # 3. Clean Description column
        df["Description"] = df["Description"].astype(str).str.upper().str.strip()

        # 4. Filter out administrative/non-product StockCodes and transaction descriptions
        invalid_stockcodes = ["POST", "D", "C2", "M", "BANK CHARGES", "PADS", "DOT", "CRUK", "AMAZONFEE", "S"]
        df = df[~df["StockCode"].astype(str).isin(invalid_stockcodes)]
        df = df[~df["StockCode"].astype(str).str.startswith("gift_", na=False)]

        invalid_keywords = [
            "POSTAGE", "DOTCOM POSTAGE", "CRUK COMMISSION", "MANUAL", 
            "AMAZON FEE", "SAMPLES", "ADJUST", "CHECK", "DAMAGED", "LOST", 
            "WRONG", "??", "WRONG CODE", "SHIPPING"
        ]
        for kw in invalid_keywords:
            df = df[~df["Description"].str.contains(kw, na=False, case=False, regex=False)]

        # 5. Filter out cancelled invoices (InvoiceNo starting with 'C')
        before_cancel = len(df)
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C", na=False)]
        logger.info(f"Removed {before_cancel - len(df)} cancelled invoices (InvoiceNo starting with 'C').")

        # 6. Calculate Total Transaction Amount
        df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
        
        # Parse Dates
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format='mixed')

        logger.info(f"Data cleaning completed. Cleaned dataset shape: {df.shape}")
        self.cleaned_df = df

        # Create df_purchases_only by removing rows with negative/zero Quantity or UnitPrice
        before_purch_filter = len(df)
        df_purch_only = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
        logger.info(f"Created df_purchases_only. Removed {before_purch_filter - len(df_purch_only)} rows with negative/zero Quantity or UnitPrice.")
        self.df_purchases_only = df_purch_only

        return df

    def engineer_rfm_features(self) -> pd.DataFrame:
        """Engineers Recency, Frequency, Monetary (RFM), and customer Return Ratio features.
        
        Note: Core RFM features (Recency, Frequency, Monetary) are calculated using 
        self.df_purchases_only (purchases-only data containing positive quantities & unit prices).
        The ReturnRatio metric is computed as a supplemental signal from the pre-cancellation-filtered
        data (retaining negative transaction lines) to represent customer return propensity accurately.
        """
        if self.cleaned_df is None or self.df_purchases_only is None:
            self.clean_data()

        logger.info("Engineering RFM and returns features...")
        
        # Max transaction date in purchases dataset (reference date)
        max_date = self.df_purchases_only["InvoiceDate"].max()
        
        # Aggregate core RFM features at CustomerID level using purchases-only data
        # Monetary is sum of purchases only, satisfying spec constraints.
        rfm = self.df_purchases_only.groupby("CustomerID").agg({
            "TotalAmount": "sum"
        }).rename(columns={"TotalAmount": "Monetary"})

        # Get Recency and Frequency from purchases-only data
        purchase_metrics = self.df_purchases_only.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (max_date - x.max()).days,
            "InvoiceNo": "nunique"
        }).rename(columns={"InvoiceDate": "Recency", "InvoiceNo": "Frequency"})

        # Join the metrics
        self.rfm_df = rfm.join(purchase_metrics, how="inner")
        
        # Remove customers with zero/negative net spend (though df_purchases_only naturally guarantees positive spend)
        self.rfm_df = self.rfm_df[self.rfm_df["Monetary"] > 0]

        # Calculate Supplemental Return Ratio from the pre-cancellation-filtered dataset (where returns/cancellations exist)
        returns_count = self.pre_cancellation_df[self.pre_cancellation_df["Quantity"] < 0].groupby("CustomerID")["InvoiceNo"].nunique()
        total_invoices = self.pre_cancellation_df.groupby("CustomerID")["InvoiceNo"].nunique()
        return_ratio = (returns_count / total_invoices).fillna(0).to_frame(name="ReturnRatio")
        
        # Join supplemental metrics
        self.rfm_df = self.rfm_df.join(return_ratio, how="left").fillna(0)
        
        logger.info(f"Features engineered. Unique customers: {self.rfm_df.shape[0]}")
        
        # Capping outliers at 99.5th percentile to prevent centroid distortion in K-Means
        logger.info("Applying professional outlier capping (99.5th percentile)...")
        for col in ["Recency", "Frequency", "Monetary"]:
            cap_val = self.rfm_df[col].quantile(0.995)
            self.rfm_df[col] = np.clip(self.rfm_df[col], None, cap_val)
            
        return self.rfm_df

    def train_clustering_model(self, n_clusters: int = 4) -> Tuple[StandardScaler, KMeans, Dict[int, str]]:
        """Log-transforms, scales, and clusters RFM data using K-Means."""
        if self.rfm_df is None:
            self.engineer_rfm_features()

        logger.info(f"Training K-Means model with {n_clusters} clusters...")
        
        # 1. Log transform only core RFM features to handle skewness
        rfm_base = self.rfm_df[["Recency", "Frequency", "Monetary"]]
        rfm_log = np.log1p(rfm_base)
        
        # 2. Scale features
        self.scaler = StandardScaler()
        rfm_scaled = self.scaler.fit_transform(rfm_log)
        
        # 3. Fit K-Means
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.rfm_df["Cluster"] = self.kmeans.fit_predict(rfm_scaled)
        
        # Evaluation Metrics
        sil = silhouette_score(rfm_scaled, self.rfm_df["Cluster"], sample_size=10000, random_state=42)
        logger.info(f"K-Means trained. Inertia (Loss): {self.kmeans.inertia_:.2f} | Silhouette: {sil:.4f}")
        
        # 4. Dynamic Profiling based on Monetary rank
        cluster_profiles = self.rfm_df.groupby("Cluster").mean()
        sorted_by_monetary = cluster_profiles.sort_values(by="Monetary")
        monetary_ranks = sorted_by_monetary.index.tolist()
        
        # Mapping labels dynamically
        self.cluster_mapping = {}
        self.cluster_mapping[monetary_ranks[3]] = "High-Value"
        self.cluster_mapping[monetary_ranks[2]] = "Regular"
        
        # Out of the two lower monetary clusters, the one with higher Recency is At-Risk
        if cluster_profiles.loc[monetary_ranks[0], "Recency"] > cluster_profiles.loc[monetary_ranks[1], "Recency"]:
            self.cluster_mapping[monetary_ranks[0]] = "At-Risk"
            self.cluster_mapping[monetary_ranks[1]] = "Occasional"
        else:
            self.cluster_mapping[monetary_ranks[0]] = "Occasional"
            self.cluster_mapping[monetary_ranks[1]] = "At-Risk"
            
        logger.info(f"Dynamic Cluster Mapping identified: {self.cluster_mapping}")
        
        self.rfm_df["Segment"] = self.rfm_df["Cluster"].map(self.cluster_mapping)
        return self.scaler, self.kmeans, self.cluster_mapping

    def build_recommendations(self, min_purchases: int = 5, top_n: int = 5) -> Dict[str, List[str]]:
        """Builds similarity matrix and precomputes recommendations with bestseller fallback."""
        if self.df_purchases_only is None:
            self.clean_data()

        logger.info("Building product recommendation matrix...")
        
        # Filter out unpopular products using df_purchases_only to reduce noise
        product_counts = self.df_purchases_only["Description"].value_counts()
        popular_products = product_counts[product_counts >= min_purchases].index
        df_filtered = self.df_purchases_only[self.df_purchases_only["Description"].isin(popular_products)]
        
        # Pivot table: Product vs Customer (binary interaction)
        product_customer_matrix = df_filtered.pivot_table(
            index="Description",
            columns="CustomerID",
            values="Quantity",
            aggfunc="count",
            fill_value=0
        ).clip(upper=1)
        
        # Calculate Cosine Similarity
        similarity = cosine_similarity(product_customer_matrix)
        similarity_df = pd.DataFrame(
            similarity,
            index=product_customer_matrix.index,
            columns=product_customer_matrix.index
        )
        
        # Precompute top recommendations
        self.recommendations = {}
        for product in similarity_df.index:
            similar_items = similarity_df[product].sort_values(ascending=False)
            similar_items = similar_items.drop(labels=[product], errors="ignore")
            # Store tuple of (item_name, similarity_score)
            self.recommendations[product] = list(zip(
                similar_items.head(top_n).index.tolist(),
                similar_items.head(top_n).values.tolist()
            ))
            
        # Add a special fallback key of overall top selling products from purchases data
        overall_top_counts = self.df_purchases_only["Description"].value_counts().head(top_n)
        overall_top = list(zip(overall_top_counts.index.tolist(), overall_top_counts.values.tolist()))
        self.recommendations["__FALLBACK__"] = overall_top
        
        logger.info("Recommendation dictionary with cold-start fallback built successfully.")
        return self.recommendations

    def build_association_rules(self, min_support_invoices: int = 5, top_n: int = 5) -> Dict[str, List[Tuple[str, float, float]]]:
        """Calculates association rules (Frequently Bought Together) using pure Pandas/Numpy.
        Returns a dictionary mapping a product to its top recommended items based on Lift.
        """
        if self.df_purchases_only is None:
            self.clean_data()

        logger.info("Computing Market Basket Association Rules...")
        
        # 1. Filter out unpopular items to keep computation fast
        product_counts = self.df_purchases_only["Description"].value_counts()
        popular_products = product_counts[product_counts >= min_support_invoices].index.tolist()
        df_filtered = self.df_purchases_only[self.df_purchases_only["Description"].isin(popular_products)]
        
        # 2. Get unique transactions (InvoiceNo, Description)
        transactions = df_filtered[["InvoiceNo", "Description"]].drop_duplicates()
        
        # 3. Count support of each item (invoices containing the item)
        item_support = transactions["Description"].value_counts().to_dict()
        total_invoices = transactions["InvoiceNo"].nunique()
        
        # 4. Self-join transactions on InvoiceNo to find pairs
        pairs = pd.merge(transactions, transactions, on="InvoiceNo", suffixes=("_A", "_B"))
        
        # Filter out self-pairs (A == B)
        pairs = pairs[pairs["Description_A"] != pairs["Description_B"]]
        
        # Group by Description_A and Description_B to count co-occurrences
        pair_counts = pairs.groupby(["Description_A", "Description_B"]).size().reset_index(name="co_occurrences")
        
        # Dynamically scale co-occurrences threshold to prevent high-lift bias on rare co-purchases
        min_co_occurrences = max(3, min_support_invoices // 2)
        pair_counts = pair_counts[pair_counts["co_occurrences"] >= min_co_occurrences]
        
        # Calculate Confidence and Lift
        pair_counts["support_A"] = pair_counts["Description_A"].map(item_support)
        pair_counts["support_B"] = pair_counts["Description_B"].map(item_support)
        pair_counts["confidence"] = pair_counts["co_occurrences"] / pair_counts["support_A"]
        pair_counts["lift"] = (pair_counts["co_occurrences"] * total_invoices) / (pair_counts["support_A"] * pair_counts["support_B"])
        
        # Create output dictionary mapping each product to top_n recommendations (sorted by lift)
        rules_dict = {}
        for product in popular_products:
            prod_rules = pair_counts[pair_counts["Description_A"] == product]
            if not prod_rules.empty:
                # Sort by lift, then confidence
                top_rules = prod_rules.sort_values(by=["lift", "confidence"], ascending=False).head(top_n)
                # Store tuple: (recommended_product, lift, confidence)
                rules_dict[product] = list(zip(
                    top_rules["Description_B"].tolist(),
                    top_rules["lift"].tolist(),
                    top_rules["confidence"].tolist()
                ))
        
        # Fallback to top overall products
        overall_top = self.df_purchases_only["Description"].value_counts().head(top_n).index.tolist()
        rules_dict["__FALLBACK__"] = [(item, 1.0, 1.0) for item in overall_top]
        
        self.association_rules = rules_dict
        logger.info("Market Basket Association Rules computed successfully.")
        return rules_dict

    def build_clv_models(self) -> None:
        """Fits Beta-Geometric/Negative Binomial (BG/NBD) and Gamma-Gamma models
        to predict customer future purchases and lifetime value.
        """
        logger.info("Building BG/NBD and Gamma-Gamma CLV models...")
        try:
            from lifetimes import BetaGeoFitter, GammaGammaFitter
            from lifetimes.utils import summary_data_from_transaction_data

            # Prepare data
            df_purch = self.df_purchases_only.copy()
            df_purch["InvoiceDate"] = pd.to_datetime(df_purch["InvoiceDate"])

            summary = summary_data_from_transaction_data(
                df_purch,
                customer_id_col="CustomerID",
                datetime_col="InvoiceDate",
                monetary_value_col="TotalAmount",
                observation_period_end=df_purch["InvoiceDate"].max()
            )

            # Scale days to months to ensure optimizer stability
            summary_scaled = summary.copy()
            summary_scaled['recency'] = summary_scaled['recency'] / 30.0
            summary_scaled['T'] = summary_scaled['T'] / 30.0

            # Fit BG/NBD Fitter (0.05 penalizer for convergence safety)
            bgf = BetaGeoFitter(penalizer_coef=0.05)
            bgf.fit(summary_scaled['frequency'], summary_scaled['recency'], summary_scaled['T'])

            # Fit Gamma-Gamma Fitter (frequency > 0 only)
            returning_customers = summary_scaled[summary_scaled['frequency'] > 0]
            ggf = GammaGammaFitter(penalizer_coef=0.05)
            ggf.fit(returning_customers['frequency'], returning_customers['monetary_value'])

            # Expected transactions in next 30 days (t=1.0 month)
            summary['ExpectedPurchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
                1.0, summary_scaled['frequency'], summary_scaled['recency'], summary_scaled['T']
            )

            # Predicted CLV in next 3 months (time=3 months, 1% monthly discount)
            summary['PredictedCLV'] = ggf.customer_lifetime_value(
                bgf,
                summary_scaled['frequency'],
                summary_scaled['recency'],
                summary_scaled['T'],
                summary_scaled['monetary_value'],
                time=3,
                discount_rate=0.01
            ).fillna(0)

            # Merge predictions into rfm_df
            if self.rfm_df is not None:
                self.rfm_df = self.rfm_df.join(summary[['ExpectedPurchases', 'PredictedCLV']], how='left').fillna(0)

            self.bgf_model = bgf
            self.ggf_model = ggf
            logger.info("CLV models trained successfully and predictions merged into RFM.")
        except Exception as e:
            logger.error(f"Error building CLV models: {e}")

    def save_artifacts(self) -> None:
        """Saves generated models and datasets locally."""
        logger.info("Saving generated pipeline artifacts...")
        
        # Save datasets
        if self.rfm_df is not None:
            self.rfm_df.to_csv("rfm_clustered.csv")
            self.rfm_df[["Recency", "Frequency", "Monetary"]].to_csv("rfm_data.csv")
        
        if self.cleaned_df is not None:
            self.cleaned_df.to_csv("cleaned_dataset.csv", index=False)

        # Save models/scalers
        with open("scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open("kmeans.pkl", "wb") as f:
            pickle.dump(self.kmeans, f)
        with open("cluster_mapping.pkl", "wb") as f:
            pickle.dump(self.cluster_mapping, f)

        # Save CLV models
        if hasattr(self, 'bgf_model') and self.bgf_model is not None:
            import dill
            with open("bgf.pkl", "wb") as f:
                dill.dump(self.bgf_model, f)
        if hasattr(self, 'ggf_model') and self.ggf_model is not None:
            import dill
            with open("ggf.pkl", "wb") as f:
                dill.dump(self.ggf_model, f)
        
        # Save recommendations
        if self.recommendations is not None:
            with open("recommendations.pkl", "wb") as f:
                pickle.dump(self.recommendations, f)
            with open("product_list.pkl", "wb") as f:
                # Exclude the fallback key from the auto-suggest list in the frontend catalog
                catalog = [k for k in self.recommendations.keys() if k != "__FALLBACK__"]
                pickle.dump(catalog, f)
                
        # Save Association Rules
        if self.association_rules is not None:
            with open("association_rules.pkl", "wb") as f:
                pickle.dump(self.association_rules, f)
                
        logger.info("Artifacts saved successfully.")

    def run_pipeline(self, n_clusters: int = 4, min_purchases: int = 5, top_n: int = 5) -> None:
        """Executes the complete pipeline in a strict sequential order:
        Clean -> RFM Feature Engineering -> K-Means Clustering -> CLV Forecasting -> Collaborative Filtering -> Market Basket.
        """
        logger.info("Executing pipeline in sequence: Clean -> RFM -> Cluster -> CLV -> RecSys -> MarketBasket")
        self.load_data()
        self.clean_data()
        self.engineer_rfm_features()
        self.train_clustering_model(n_clusters=n_clusters)
        self.build_clv_models()
        self.build_recommendations(min_purchases=min_purchases, top_n=top_n)
        self.build_association_rules(min_support_invoices=min_purchases, top_n=top_n)
        self.save_artifacts()
        logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    pipeline = ECommerceDataPipeline("dataset.csv")
    pipeline.run_pipeline()
