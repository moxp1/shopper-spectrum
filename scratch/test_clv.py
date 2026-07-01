import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

# Load data
df = pd.read_csv("dataset.csv", encoding="ISO-8859-1")
df = df.dropna(subset=["CustomerID"])
df["CustomerID"] = df["CustomerID"].astype(int)
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format='mixed')
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# Summary data
summary = summary_data_from_transaction_data(
    df, 
    customer_id_col="CustomerID", 
    datetime_col="InvoiceDate", 
    monetary_value_col="TotalAmount",
    observation_period_end=df["InvoiceDate"].max()
)

print(summary.head())

summary['recency_scaled'] = summary['recency'] / 30.0
summary['T_scaled'] = summary['T'] / 30.0

bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(summary['frequency'], summary['recency_scaled'], summary['T_scaled'])
print(bgf)

# Predict future transactions (time=1 month since scaled to months)
summary['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    1.0, summary['frequency'], summary['recency_scaled'], summary['T_scaled']
)
print("Predicted purchases head:")
print(summary['predicted_purchases'].head())

# Filter customers with at least one purchase for Gamma-Gamma
returning_customers_summary = summary[summary['frequency'] > 0]
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(returning_customers_summary['frequency'], returning_customers_summary['monetary_value'])
print(ggf)

# Predict CLV
summary['predicted_clv'] = ggf.customer_lifetime_value(
    bgf,
    summary['frequency'],
    summary['recency_scaled'],
    summary['T_scaled'],
    summary['monetary_value'],
    time=1, # 1 month
    discount_rate=0.01, # 1% discount rate monthly
    freq="M"
)
print("Predicted CLV head:")
print(summary['predicted_clv'].head())
