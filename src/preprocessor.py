import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger("ShopperSpectrumPipeline.Preprocessor")

class Preprocessor:
    @staticmethod
    def load_data(filepath: str) -> pd.DataFrame:
        logger.info(f"Loading raw dataset from {filepath}...")
        try:
            df = pd.read_csv(filepath, encoding="ISO-8859-1")
            logger.info(f"Loaded dataset successfully. Initial shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    @staticmethod
    def clean_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Cleans data and returns 3 DataFrames:
        - cleaned_df (full cleaned data with returns)
        - df_purchases_only (only positive purchases)
        - pre_cancellation_df (before filtering 'C' invoices for return ratio)
        """
        logger.info("Starting data cleaning process...")
        df = raw_df.copy()

        # 1. Remove exact duplicate records
        before_dup = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {before_dup - len(df)} duplicate records.")

        # 2. Handle missing CustomerIDs
        before_null_cust = len(df)
        df = df.dropna(subset=["CustomerID"])
        df["CustomerID"] = df["CustomerID"].astype(int)
        logger.info(f"Removed {before_null_cust - len(df)} rows with missing CustomerID.")

        # Keep a copy before cancellation filtering for return ratio calculation
        pre_cancellation_df = df.copy()

        # 3. Clean Description column
        df["Description"] = df["Description"].astype(str).str.upper().str.strip()

        # 4. Filter out administrative/non-product StockCodes
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

        # 5. Filter out cancelled invoices
        before_cancel = len(df)
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C", na=False)]
        logger.info(f"Removed {before_cancel - len(df)} cancelled invoices.")

        # 6. Calculate Total Amount
        df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
        
        # Parse Dates
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format='mixed')

        logger.info(f"Data cleaning completed. Cleaned dataset shape: {df.shape}")

        # Purchases only
        df_purchases_only = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

        return df, df_purchases_only, pre_cancellation_df
