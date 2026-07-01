import pytest
import pandas as pd
from src.preprocessor import Preprocessor

def test_clean_data_removes_duplicates():
    # Setup
    data = {
        'InvoiceNo': ['123', '123'],
        'StockCode': ['A', 'A'],
        'Description': ['TEST', 'TEST'],
        'Quantity': [1, 1],
        'InvoiceDate': ['2023-01-01', '2023-01-01'],
        'UnitPrice': [10.0, 10.0],
        'CustomerID': [100, 100],
        'Country': ['UK', 'UK']
    }
    df = pd.DataFrame(data)
    
    # Act
    cleaned, _, _ = Preprocessor.clean_data(df)
    
    # Assert
    assert len(cleaned) == 1

def test_clean_data_handles_cancelled_orders():
    # Setup
    data = {
        'InvoiceNo': ['123', 'C124'],
        'StockCode': ['A', 'B'],
        'Description': ['TEST', 'TEST'],
        'Quantity': [1, -1],
        'InvoiceDate': ['2023-01-01', '2023-01-01'],
        'UnitPrice': [10.0, 10.0],
        'CustomerID': [100, 101],
        'Country': ['UK', 'UK']
    }
    df = pd.DataFrame(data)
    
    # Act
    cleaned, _, _ = Preprocessor.clean_data(df)
    
    # Assert
    assert len(cleaned) == 1
    assert cleaned.iloc[0]['InvoiceNo'] == '123'
