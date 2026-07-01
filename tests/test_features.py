import pytest
import pandas as pd
from src.features import FeatureEngineer

def test_engineer_rfm_no_negative_monetary():
    # Setup - simulate data that only has positive purchases (since preprocessor outputs df_purchases_only)
    purchases_data = {
        'InvoiceNo': ['1', '2', '3'],
        'CustomerID': [10, 10, 11],
        'InvoiceDate': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-01']),
        'TotalAmount': [100.0, 50.0, 10.0]
    }
    df_purch = pd.DataFrame(purchases_data)
    
    pre_cancel_data = {
        'InvoiceNo': ['1', '2', 'C9', '3'],
        'CustomerID': [10, 10, 10, 11],
        'Quantity': [1, 1, -1, 1]
    }
    df_precancel = pd.DataFrame(pre_cancel_data)
    
    # Act
    rfm_df = FeatureEngineer.engineer_rfm(df_purch, df_precancel)
    
    # Assert
    assert all(rfm_df['Monetary'] > 0)
    # The max value will be slightly capped by the 99.5th percentile rule on small lists
    assert rfm_df.loc[10, 'Monetary'] > 140.0
    assert rfm_df.loc[11, 'Monetary'] == 10.0
