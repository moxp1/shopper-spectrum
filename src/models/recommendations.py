import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

logger = logging.getLogger("ShopperSpectrumPipeline.Recommendations")

class RecommendationEngine:
    @staticmethod
    def build_recommendations(df_purchases: pd.DataFrame, min_purchases: int = 5, top_n: int = 5) -> Tuple[Dict[str, List[tuple]], List[str]]:
        logger.info("Building product recommendation matrix...")
        
        product_counts = df_purchases["Description"].value_counts()
        popular_products = product_counts[product_counts >= min_purchases].index
        df_filtered = df_purchases[df_purchases["Description"].isin(popular_products)]
        
        customer_u = df_filtered["CustomerID"].astype("category")
        description_u = df_filtered["Description"].astype("category")
        
        row = description_u.cat.codes
        col = customer_u.cat.codes
        data = np.ones(len(df_filtered))
        
        product_customer_sparse = csr_matrix((data, (row, col)), shape=(len(description_u.cat.categories), len(customer_u.cat.categories)))
        product_customer_sparse.data = np.clip(product_customer_sparse.data, 0, 1)
        
        similarity = cosine_similarity(product_customer_sparse)
        similarity_df = pd.DataFrame(
            similarity,
            index=description_u.cat.categories,
            columns=description_u.cat.categories
        )
        
        recommendations = {}
        for product in similarity_df.index:
            similar_items = similarity_df[product].sort_values(ascending=False)
            similar_items = similar_items.drop(labels=[product], errors="ignore")
            recommendations[product] = list(zip(
                similar_items.head(top_n).index.tolist(),
                similar_items.head(top_n).values.tolist()
            ))
            
        overall_top_counts = df_purchases["Description"].value_counts().head(top_n)
        overall_top = list(zip(overall_top_counts.index.tolist(), overall_top_counts.values.tolist()))
        recommendations["__FALLBACK__"] = overall_top
        
        catalog = [k for k in recommendations.keys() if k != "__FALLBACK__"]
        
        logger.info("Recommendation dictionary built.")
        return recommendations, catalog

    @staticmethod
    def build_association_rules(df_purchases: pd.DataFrame, min_support_invoices: int = 5, top_n: int = 5) -> Dict[str, List[Tuple[str, float, float]]]:
        logger.info("Computing Market Basket Association Rules...")
        
        product_counts = df_purchases["Description"].value_counts()
        popular_products = product_counts[product_counts >= min_support_invoices].index.tolist()
        df_filtered = df_purchases[df_purchases["Description"].isin(popular_products)]
        
        transactions = df_filtered[["InvoiceNo", "Description"]].drop_duplicates()
        
        item_support = transactions["Description"].value_counts().to_dict()
        total_invoices = transactions["InvoiceNo"].nunique()
        
        pairs = pd.merge(transactions, transactions, on="InvoiceNo", suffixes=("_A", "_B"))
        pairs = pairs[pairs["Description_A"] != pairs["Description_B"]]
        
        pair_counts = pairs.groupby(["Description_A", "Description_B"]).size().reset_index(name="co_occurrences")
        
        min_co_occurrences = max(3, min_support_invoices // 2)
        pair_counts = pair_counts[pair_counts["co_occurrences"] >= min_co_occurrences]
        
        pair_counts["support_A"] = pair_counts["Description_A"].map(item_support)
        pair_counts["support_B"] = pair_counts["Description_B"].map(item_support)
        pair_counts["confidence"] = pair_counts["co_occurrences"] / pair_counts["support_A"]
        pair_counts["lift"] = (pair_counts["co_occurrences"] * total_invoices) / (pair_counts["support_A"] * pair_counts["support_B"])
        
        rules_dict = {}
        for product in popular_products:
            prod_rules = pair_counts[pair_counts["Description_A"] == product]
            if not prod_rules.empty:
                top_rules = prod_rules.sort_values(by=["lift", "confidence"], ascending=False).head(top_n)
                rules_dict[product] = list(zip(
                    top_rules["Description_B"].tolist(),
                    top_rules["lift"].tolist(),
                    top_rules["confidence"].tolist()
                ))
        
        overall_top = df_purchases["Description"].value_counts().head(top_n).index.tolist()
        rules_dict["__FALLBACK__"] = [(item, 1.0, 1.0) for item in overall_top]
        
        logger.info("Market Basket Association Rules computed successfully.")
        return rules_dict
