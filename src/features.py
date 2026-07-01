import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("ShopperSpectrumPipeline.Features")

class FeatureEngineer:
    @staticmethod
    def engineer_rfm(df_purchases: pd.DataFrame, pre_cancellation_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering RFM and returns features...")
        
        max_date = df_purchases["InvoiceDate"].max()
        
        rfm = df_purchases.groupby("CustomerID").agg({
            "TotalAmount": "sum"
        }).rename(columns={"TotalAmount": "Monetary"})

        purchase_metrics = df_purchases.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (max_date - x.max()).days,
            "InvoiceNo": "nunique"
        }).rename(columns={"InvoiceDate": "Recency", "InvoiceNo": "Frequency"})

        rfm_df = rfm.join(purchase_metrics, how="inner")
        rfm_df = rfm_df[rfm_df["Monetary"] > 0]

        returns_count = pre_cancellation_df[pre_cancellation_df["Quantity"] < 0].groupby("CustomerID")["InvoiceNo"].nunique()
        total_invoices = pre_cancellation_df.groupby("CustomerID")["InvoiceNo"].nunique()
        return_ratio = (returns_count / total_invoices).fillna(0).to_frame(name="ReturnRatio")
        
        rfm_df = rfm_df.join(return_ratio, how="left").fillna(0)
        
        logger.info("Applying 99.5th percentile outlier capping...")
        for col in ["Recency", "Frequency", "Monetary"]:
            cap_val = rfm_df[col].quantile(0.995)
            rfm_df[col] = np.clip(rfm_df[col], None, cap_val)
            
        return rfm_df

    @staticmethod
    def build_clv(df_purchases: pd.DataFrame, rfm_df: pd.DataFrame, time_months=3, discount_rate=0.01) -> tuple:
        logger.info("Building CLV models...")
        try:
            from lifetimes import BetaGeoFitter, GammaGammaFitter
            from lifetimes.utils import summary_data_from_transaction_data

            df_purch = df_purchases.copy()
            summary = summary_data_from_transaction_data(
                df_purch,
                customer_id_col="CustomerID",
                datetime_col="InvoiceDate",
                monetary_value_col="TotalAmount",
                observation_period_end=df_purch["InvoiceDate"].max()
            )

            summary_scaled = summary.copy()
            summary_scaled['recency'] = summary_scaled['recency'] / 30.0
            summary_scaled['T'] = summary_scaled['T'] / 30.0

            bgf = BetaGeoFitter(penalizer_coef=0.05)
            bgf.fit(summary_scaled['frequency'], summary_scaled['recency'], summary_scaled['T'])

            returning_customers = summary_scaled[summary_scaled['frequency'] > 0]
            ggf = GammaGammaFitter(penalizer_coef=0.05)
            ggf.fit(returning_customers['frequency'], returning_customers['monetary_value'])

            summary['ExpectedPurchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
                1.0, summary_scaled['frequency'], summary_scaled['recency'], summary_scaled['T']
            )

            summary['PredictedCLV'] = ggf.customer_lifetime_value(
                bgf,
                summary_scaled['frequency'],
                summary_scaled['recency'],
                summary_scaled['T'],
                summary_scaled['monetary_value'],
                time=time_months,
                discount_rate=discount_rate,
                freq="M"
            ).fillna(0)

            rfm_df = rfm_df.join(summary[['ExpectedPurchases', 'PredictedCLV']], how='left').fillna(0)
            return rfm_df, bgf, ggf
        except Exception as e:
            logger.error(f"Error building CLV models: {e}")
            return rfm_df, None, None
