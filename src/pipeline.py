import logging
import pickle
from .config_loader import load_config
from .preprocessor import Preprocessor
from .features import FeatureEngineer
from .models.clustering import ClusteringModel
from .models.recommendations import RecommendationEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ShopperSpectrumPipeline")

def run_pipeline():
    logger.info("Executing modularized pipeline in sequence.")
    
    # Load config
    config = load_config()
    paths = config['paths']
    params = config['pipeline']
    
    # 1. Clean Data
    raw_df = Preprocessor.load_data(paths['input_data'])
    cleaned_df, df_purchases_only, pre_cancellation_df = Preprocessor.clean_data(raw_df)
    
    # 2. Engineer RFM Features
    rfm_df = FeatureEngineer.engineer_rfm(df_purchases_only, pre_cancellation_df)
    
    # 3. K-Means Clustering
    rfm_df, scaler, kmeans, cluster_mapping = ClusteringModel.train(rfm_df, n_clusters=params['n_clusters'])
    
    # 4. CLV Models
    rfm_df, bgf_model, ggf_model = FeatureEngineer.build_clv(
        df_purchases_only, rfm_df, 
        time_months=params['clv_months'], 
        discount_rate=params['clv_discount_rate']
    )
    
    # 5. Product Recommendations & Market Basket
    recommendations, catalog = RecommendationEngine.build_recommendations(
        df_purchases_only, min_purchases=params['min_purchases'], top_n=params['top_n_recommendations']
    )
    association_rules = RecommendationEngine.build_association_rules(
        df_purchases_only, min_support_invoices=params['min_purchases'], top_n=params['top_n_recommendations']
    )
    
    # 6. Save Artifacts (Parquet for datasets, Pickle for models)
    logger.info("Saving generated pipeline artifacts...")
    
    # Datasets (using Fast Parquet format)
    rfm_df.to_parquet(paths['rfm_clustered'])
    rfm_df[["Recency", "Frequency", "Monetary"]].to_parquet(paths['rfm_data'])
    cleaned_df.to_parquet(paths['cleaned_data'])
    
    # Models
    with open(paths['scaler'], "wb") as f:
        pickle.dump(scaler, f)
    with open(paths['kmeans'], "wb") as f:
        pickle.dump(kmeans, f)
    with open(paths['cluster_mapping'], "wb") as f:
        pickle.dump(cluster_mapping, f)
        
    if bgf_model is not None:
        import dill
        with open(paths['bgf'], "wb") as f:
            dill.dump(bgf_model, f)
    if ggf_model is not None:
        import dill
        with open(paths['ggf'], "wb") as f:
            dill.dump(ggf_model, f)
            
    with open(paths['recommendations'], "wb") as f:
        pickle.dump(recommendations, f)
    with open(paths['product_list'], "wb") as f:
        pickle.dump(catalog, f)
    with open(paths['association_rules'], "wb") as f:
        pickle.dump(association_rules, f)
        
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    run_pipeline()
