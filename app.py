import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import scipy.stats as stats

# Pipeline imports
from src.preprocessor import Preprocessor
from src.features import FeatureEngineer
from src.models.clustering import ClusteringModel

st.set_page_config(
    page_title="Shopper Spectrum - E-Commerce Analytics Hub",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom Google Fonts and CSS for premium neon-indigo styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sidebar Overrides */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090d16 0%, #111827 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.15);
    }
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #05070c 0%, #0c0f1d 100%);
        color: #e2e8f0;
    }
    
    /* Headers styling */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .main-title {
        font-size: 2.8rem;
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        font-weight: 800;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.8rem;
        font-weight: 300;
    }
    
    /* Custom Card Style */
    .premium-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.75) 0%, rgba(30, 41, 59, 0.45) 100%);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(20px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 20px;
    }
    
    .premium-card:hover {
        transform: translateY(-5px);
        border-color: rgba(236, 72, 153, 0.4);
        box-shadow: 0 15px 35px rgba(139, 92, 246, 0.15);
    }
    
    .premium-card h3 {
        margin-top: 0;
        margin-bottom: 15px;
        font-size: 1.35rem;
        color: #f8fafc;
    }
    
    /* Recommendations item cards */
    .rec-card {
        background: rgba(15, 23, 42, 0.55);
        border-left: 4px solid #a855f7;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        transition: all 0.25s ease;
    }
    
    .rec-card:hover {
        background: rgba(15, 23, 42, 0.85);
        transform: scale(1.015);
        border-left-color: #38bdf8;
    }
    
    .rec-icon {
        font-size: 1.5rem;
        margin-right: 15px;
        background: rgba(168, 85, 247, 0.15);
        padding: 8px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
    }
    
    .rec-text {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f8fafc;
    }
    
    /* Custom status badges */
    .badge {
        padding: 8px 18px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.9rem;
        text-transform: uppercase;
        display: inline-block;
        letter-spacing: 0.06em;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .badge-high {
        background-color: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);
    }
    
    .badge-regular {
        background-color: rgba(59, 130, 246, 0.12);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.4);
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.15);
    }
    
    .badge-occasional {
        background-color: rgba(245, 158, 11, 0.12);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.4);
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.15);
    }
    
    .badge-atrisk {
        background-color: rgba(239, 68, 68, 0.12);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.15);
    }
    
    /* Statistic box styling */
    .stat-val {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 5px;
        background: linear-gradient(90deg, #f8fafc 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-lbl {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    /* Glow buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #0284c7 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.25) !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.45) !important;
        background: linear-gradient(90deg, #0ea5e9 0%, #8b5cf6 100%) !important;
    }

    /* Input focus styling */
    div[data-baseweb="input"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
    }
    
    div[data-baseweb="input"]:focus-within {
        border-color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- DATA LOADERS (DYNAMIC + PRE-TRAINED FALLBACKS) -----------------
@st.cache_resource
def load_pretrained_resources():
    """Loads pre-built models and transactional configurations with high resilience."""
    scaler = None
    kmeans = None
    cluster_mapping = None
    recommendations = None
    product_list = None
    rfm_data = None
    association_rules = None
    bgf = None
    ggf = None

    # Load each resource independently to ensure partial loading success
    if os.path.exists("scaler.pkl"):
        try:
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load scaler.pkl: {e}")

    if os.path.exists("kmeans.pkl"):
        try:
            with open("kmeans.pkl", "rb") as f:
                kmeans = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load kmeans.pkl: {e}")

    if os.path.exists("cluster_mapping.pkl"):
        try:
            with open("cluster_mapping.pkl", "rb") as f:
                cluster_mapping = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load cluster_mapping.pkl: {e}")

    if os.path.exists("recommendations.pkl"):
        try:
            with open("recommendations.pkl", "rb") as f:
                recommendations = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load recommendations.pkl: {e}")

    if os.path.exists("product_list.pkl"):
        try:
            with open("product_list.pkl", "rb") as f:
                product_list = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load product_list.pkl: {e}")

    if os.path.exists("association_rules.pkl"):
        try:
            with open("association_rules.pkl", "rb") as f:
                association_rules = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load association_rules.pkl: {e}")

    if os.path.exists("rfm_clustered.parquet"):
        try:
            rfm_data = pd.read_parquet("rfm_clustered.parquet")
            if "Recency" in rfm_data.columns:
                rfm_data["Recency"] = rfm_data["Recency"].round().astype(int)
            if "Frequency" in rfm_data.columns:
                rfm_data["Frequency"] = rfm_data["Frequency"].round().astype(int)
        except Exception as e:
            st.warning(f"Could not load rfm_clustered.parquet: {e}")

    if os.path.exists("bgf.pkl"):
        try:
            import dill
            with open("bgf.pkl", "rb") as f:
                bgf = dill.load(f)
        except Exception as e:
            st.warning(f"Could not load bgf.pkl: {e}")

    if os.path.exists("ggf.pkl"):
        try:
            import dill
            with open("ggf.pkl", "rb") as f:
                ggf = dill.load(f)
        except Exception as e:
            st.warning(f"Could not load ggf.pkl: {e}")

    # Fallback fit-on-the-fly if files load partially but scaler/kmeans is missing
    if (scaler is None or kmeans is None) and rfm_data is not None:
        try:
            rfm_fit = rfm_data.copy()
            # Use the clustering model from src pipeline
            _, scaler_fb, kmeans_fb, mapping_fb = ClusteringModel.train(rfm_fit, n_clusters=4)
            if scaler is None: scaler = scaler_fb
            if kmeans is None: kmeans = kmeans_fb
            if cluster_mapping is None: cluster_mapping = mapping_fb
        except Exception as e:
            st.warning(f"Failed to fit fallback models: {e}")

    return scaler, kmeans, cluster_mapping, recommendations, product_list, rfm_data, association_rules, bgf, ggf

@st.cache_data
def load_cleaned_transactions():
    """Loads cleaned transactional dataset for EDA."""
    if os.path.exists("cleaned_dataset.parquet"):
        try:
            df = pd.read_parquet("cleaned_dataset.parquet")
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
            if "Quantity" in df.columns:
                df["Quantity"] = df["Quantity"].round().astype(int)
            if "CustomerID" in df.columns:
                df["CustomerID"] = df["CustomerID"].astype(int)
            for col in ["Description", "Country", "InvoiceNo", "StockCode"]:
                if col in df.columns:
                    df[col] = df[col].astype("category")
            return df
        except Exception as e:
            st.error(f"Error loading cleaned dataset: {e}")
            return None
    return None

# Initial load of default resources
p_scaler, p_kmeans, p_cluster_mapping, p_recommendations, p_product_list, p_rfm_data, p_association_rules, p_bgf, p_ggf = load_pretrained_resources()
default_df = load_cleaned_transactions()

# ----------------- DYNAMIC PROCESSING FOR CUSTOM FILE UPLOADS -----------------
def clean_and_build_pipeline(uploaded_file):
    """Processes uploaded dataset and fits KMeans / similarities on the fly."""
    try:
        raw_df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        
        # Use modular pipeline functions
        _, df_purchases_only, pre_cancellation_df = Preprocessor.clean_data(raw_df)
        rfm = FeatureEngineer.engineer_rfm(df_purchases_only, pre_cancellation_df)
        rfm, scaler, kmeans, cluster_mapping = ClusteringModel.train(rfm, n_clusters=4)
        
        if "Recency" in rfm.columns:
            rfm["Recency"] = rfm["Recency"].round().astype(int)
        if "Frequency" in rfm.columns:
            rfm["Frequency"] = rfm["Frequency"].round().astype(int)
        if "Quantity" in df_purchases_only.columns:
            df_purchases_only["Quantity"] = df_purchases_only["Quantity"].round().astype(int)
            
        product_list = sorted(df_purchases_only["Description"].unique().tolist())
        
        return df_purchases_only, rfm, scaler, kmeans, cluster_mapping, product_list
    except Exception as e:
        st.error(f"Failed to compile custom file pipeline: {e}")
        return None, None, None, None, None, None

# ----------------- DYNAMIC RECOMMENDATION ALGORITHMS -----------------
@st.cache_resource
def build_content_recommender(df):
    """Computes TF-IDF on descriptions for content-based matching."""
    try:
        unique_desc = df[["Description"]].drop_duplicates().dropna()
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(unique_desc["Description"])
        return unique_desc.reset_index(drop=True), tfidf_matrix
    except Exception as e:
        st.error(f"Error building TF-IDF recommender: {e}")
        return None, None

def get_content_recommendations(target_desc, prod_df, tfidf_matrix, top_n=5):
    """Fetches text similarity recommendation."""
    if prod_df is None or tfidf_matrix is None or target_desc not in prod_df["Description"].values:
        return []
    idx = prod_df[prod_df["Description"] == target_desc].index[0]
    
    # Compute similarity only for the target product
    target_vector = tfidf_matrix[idx]
    cosine_sim_vector = cosine_similarity(target_vector, tfidf_matrix).flatten()
    
    sim_scores = list(enumerate(cosine_sim_vector))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [(prod_df.iloc[i]["Description"], round(score * 100, 1)) for i, score in sim_scores]

@st.cache_data
def get_user_collaborative_recs(df, target_customer_id, top_n=5):
    """Computes recommendations for a Customer based on similar users' purchase histories."""
    try:
        # Filter popular items for memory efficiency
        p_counts = df["Description"].value_counts()
        popular_p = p_counts[p_counts >= 3].index
        df_filtered = df[df["Description"].isin(popular_p)]
        
        # Build User-Item Sparse Matrix
        customer_u = df_filtered["CustomerID"].astype("category")
        description_u = df_filtered["Description"].astype("category")
        
        row = customer_u.cat.codes
        col = description_u.cat.codes
        data = np.ones(len(df_filtered))
        
        user_item_sparse = csr_matrix((data, (row, col)), shape=(len(customer_u.cat.categories), len(description_u.cat.categories)))
        user_item_sparse.data = np.clip(user_item_sparse.data, 0, 1)
        
        if target_customer_id not in customer_u.cat.categories:
            # Fallback to general popular items
            top_overall = df["Description"].value_counts().head(top_n).index.tolist()
            return [(item, 1.0) for item in top_overall]
            
        target_idx = customer_u.cat.categories.get_loc(target_customer_id)
        
        # Compute similarity only for the target user (O(U) instead of O(U^2))
        user_sim = cosine_similarity(user_item_sparse[target_idx], user_item_sparse).flatten()
        
        # Get top 5 similar users (excluding the user themselves)
        similar_user_indices = user_sim.argsort()[::-1][1:6]
        
        # Calculate recommendation scores
        target_purchases = user_item_sparse[target_idx].toarray().flatten()
        unbought_mask = (target_purchases == 0)
        
        similar_users_purchases = user_item_sparse[similar_user_indices].toarray()
        recommendation_scores = similar_users_purchases.mean(axis=0) * unbought_mask
        
        # Get top items
        top_item_indices = recommendation_scores.argsort()[::-1][:top_n]
        
        top_items = []
        for idx in top_item_indices:
            score = recommendation_scores[idx]
            if score > 0:
                top_items.append((description_u.cat.categories[idx], round(score, 2)))
                
        return top_items
    except Exception as e:
        st.error(f"Error compiling User-Based CF: {e}")
        return []

# ----------------- SIDEBAR INTERACTIVE LOADER -----------------
with st.sidebar:
    st.markdown("<div style='text-align: center; padding-top: 10px;'>", unsafe_allow_html=True)
    st.image("https://img.icons8.com/isometric-line/100/38bdf8/shopping-cart.png", width=70)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; margin-top: 5px; color: #f8fafc; font-size: 1.5rem;'>Shopper Spectrum</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-style: italic; font-size: 0.9rem; margin-bottom: 15px;'>E-Commerce Analytics Hub</p>", unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    
    menu = st.radio(
        "Navigation Menu",
        [
            "📊 Dashboard & Analytics",
            "📈 Exploratory Data Analysis (EDA)",
            "🔍 Customer Segmentation Module",
            "🎯 Product Recommendation Engine",
            "🧪 Statistical Hypothesis Tests"
        ],
        index=0
    )
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    st.markdown("#### **Algorithm Stack**")
    st.info("📊 **Algorithm:** K-Means Clustering\n\n🎯 **Recommendation:** Collaborative Filtering & MBA\n\n🧪 **Hypothesis Testing:** Welch's t-test, ANOVA")

# Resolve active dataset and models
active_df = default_df
if active_df is not None:
    if "Quantity" in active_df.columns:
        active_df["Quantity"] = active_df["Quantity"].round().astype(int)
    if "CustomerID" in active_df.columns:
        active_df["CustomerID"] = active_df["CustomerID"].astype(int)

active_rfm = p_rfm_data
if active_rfm is not None:
    if "Recency" in active_rfm.columns:
        active_rfm["Recency"] = active_rfm["Recency"].round().astype(int)
    if "Frequency" in active_rfm.columns:
        active_rfm["Frequency"] = active_rfm["Frequency"].round().astype(int)

active_scaler = p_scaler
active_kmeans = p_kmeans
active_cluster_mapping = p_cluster_mapping
active_product_list = p_product_list
active_rules = p_association_rules
active_bgf = p_bgf
active_ggf = p_ggf

# Header banner
st.markdown("""
<div style='background: linear-gradient(135deg, #1e1b4b 0%, #311042 50%, #0f172a 100%); padding: 22px 28px; border-radius: 18px; margin-bottom: 25px; border: 1px solid rgba(139, 92, 246, 0.25); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);'>
    <h1 style='margin: 0; font-size: 2rem; color: #ffffff;'>Shopper Spectrum Analytics Hub</h1>
    <p style='margin: 5px 0 0 0; color: #cbd5e1; font-size: 0.95rem; font-weight: 300;'>Customer Analytics, Market Basket Rules, and E-Commerce Decision Engines</p>
</div>
""", unsafe_allow_html=True)

# ----------------- PAGE 1: DASHBOARD & COHORTS -----------------
if menu == "📊 Dashboard & Analytics":
    if active_rfm is not None:
        c1, c2, c3, c4 = st.columns(4)
        total_cust = len(active_rfm)
        high_val_count = len(active_rfm[active_rfm['Segment'] == 'High-Value'])
        
        with c1:
            st.markdown(f"<div class='premium-card'><div class='stat-lbl'>Total Customers</div><div class='stat-val'>{total_cust:,}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='premium-card'><div class='stat-lbl'>High-Value Cohort</div><div class='stat-val'>{high_val_count:,} <span style='font-size:0.9rem;color:#10b981;font-weight:600;'>({high_val_count/total_cust*100:.1f}%)</span></div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='premium-card'><div class='stat-lbl'>Average Recency</div><div class='stat-val'>{int(round(active_rfm['Recency'].mean()))} <span style='font-size:0.9rem;color:#94a3b8;'>Days</span></div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='premium-card'><div class='stat-lbl'>Avg Customer Spend</div><div class='stat-val'>${active_rfm['Monetary'].mean():,.2f}</div></div>", unsafe_allow_html=True)
            
        col_table, col_chart = st.columns([1.1, 0.9])
        
        with col_table:
            st.markdown("### 📋 Segment Profiles Summary")
            agg_dict = {'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean', 'Cluster': 'count'}
            if 'ReturnRatio' in active_rfm.columns:
                agg_dict['ReturnRatio'] = 'mean'
            if 'ExpectedPurchases' in active_rfm.columns:
                agg_dict['ExpectedPurchases'] = 'mean'
            if 'PredictedCLV' in active_rfm.columns:
                agg_dict['PredictedCLV'] = 'mean'
                
            profiles = active_rfm.groupby('Segment').agg(agg_dict).rename(columns={'Cluster': 'Customer Count'})
            profiles['Recency'] = profiles['Recency'].map(lambda x: f"{int(round(x))} days")
            profiles['Frequency'] = profiles['Frequency'].map(lambda x: f"{int(round(x))} orders")
            profiles['Monetary'] = profiles['Monetary'].map(lambda x: f"${x:,.2f}")
            if 'ReturnRatio' in active_rfm.columns:
                profiles['ReturnRatio'] = profiles['ReturnRatio'].map(lambda x: f"{x*100:.2f}%")
            if 'ExpectedPurchases' in active_rfm.columns:
                profiles['ExpectedPurchases'] = profiles['ExpectedPurchases'].map(lambda x: f"{x:.3f} orders")
            if 'PredictedCLV' in active_rfm.columns:
                profiles['PredictedCLV'] = profiles['PredictedCLV'].map(lambda x: f"${x:,.2f}")
            profiles['Customer Count'] = profiles['Customer Count'].map(lambda x: f"{x:,}")
            st.dataframe(profiles, use_container_width=True)
            
            st.markdown("""
            <div class='premium-card' style='margin-top: 15px;'>
                <h4 style='color:#38bdf8; margin-top:0;'>💡 Executive Segment Profiles</h4>
                <p style='color:#94a3b8; font-size:0.95rem; line-height:1.6; margin-bottom:0;'>
                    The customer base is structured into 4 primary profiles. <b>High-Value</b> buyers account for the lion's share of revenue despite representing a minority size. <b>At-Risk</b> represents churning accounts requiring reactivation incentives, whereas <b>Occasional</b> buyers represent targets for catalog recommendation outreach.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_chart:
            st.markdown("### 📊 Distribution of Segments")
            seg_dist = active_rfm['Segment'].value_counts().reset_index()
            seg_dist.columns = ['Segment', 'Count']
            
            fig_bar = px.bar(
                seg_dist, x='Segment', y='Count', color='Segment',
                color_discrete_map={"High-Value": "#10b981", "Regular": "#3b82f6", "Occasional": "#f59e0b", "At-Risk": "#ef4444"},
                text='Count'
            )
            fig_bar.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False, margin=dict(l=10, r=10, t=10, b=10)
            )
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)
            
    else:
        st.warning("Please build the models first or upload a dataset.")

# ----------------- PAGE 2: EXPLORATORY DATA ANALYSIS (EDA) -----------------
elif menu == "📈 Exploratory Data Analysis (EDA)":
    if active_df is not None:
        tab_geo, tab_prod, tab_temp, tab_spend, tab_rfm = st.tabs([
            "🌍 Geo & Market Revenue", 
            "📦 Product Performance", 
            "📅 Temporal Trends", 
            "🛒 Cart & Loyalty", 
            "📊 RFM Distributions"
        ])
        
        # 1. Geography Tab (Charts 1, 2, 3)
        with tab_geo:
            st.markdown("### 🌍 Geographical & Market Revenue Metrics")
            c1, c2 = st.columns(2)
            with c1:
                country_vol = active_df["Country"].value_counts().head(10).reset_index()
                country_vol.columns = ["Country", "Transactions"]
                fig1 = px.bar(country_vol, x="Transactions", y="Country", orientation="h",
                              title="1. Countries by Transaction Volume", color="Transactions", color_continuous_scale="viridis",
                              hover_data={"Transactions": ":d", "Country": True})
                fig1.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"), xaxis=dict(tickformat="d"))
                st.plotly_chart(fig1, use_container_width=True)
                st.info("💡 **Geographic Volume Insight:** The United Kingdom heavily dominates transactional volume, accounting for over 90% of total rows. This suggests a highly centralized core market.")

            with c2:
                country_rev = active_df.groupby("Country")["TotalAmount"].sum().sort_values(ascending=False).head(10).reset_index()
                fig2 = px.bar(country_rev, x="TotalAmount", y="Country", orientation="h",
                              title="2. Countries by Total Revenue ($)", color="TotalAmount", color_continuous_scale="blues",
                              hover_data={"TotalAmount": ":$.2f", "Country": True})
                fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig2, use_container_width=True)
                st.info("💡 **Revenue Concentration Insight:** Revenue closely mirrors transaction counts. However, countries like Germany and France exhibit premium purchasing power.")

            aov_df = active_df.groupby(["Country", "InvoiceNo"])["TotalAmount"].sum().reset_index()
            aov_country = aov_df.groupby("Country")["TotalAmount"].mean().sort_values(ascending=False).head(10).reset_index()
            aov_country.columns = ["Country", "Average Order Value ($)"]
            fig3 = px.bar(aov_country, x="Average Order Value ($)", y="Country", orientation="h",
                          title="3. Average Order Value (AOV) by Country", color="Average Order Value ($)", color_continuous_scale="teal",
                          hover_data={"Average Order Value ($)": ":$.2f", "Country": True})
            fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig3, use_container_width=True)
            st.info("💡 **AOV Insight:** While the UK leads in total volume and revenue, other international destinations show significantly higher Average Order Values per cart, highlighting potential export market opportunities.")

        # 2. Products Tab (Charts 4, 5, 6)
        with tab_prod:
            st.markdown("### 📦 Product Performance & Distributions")
            c1, c2 = st.columns(2)
            with c1:
                top_qty = active_df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10).reset_index()
                fig4 = px.bar(top_qty, x="Quantity", y="Description", orientation="h",
                              title="4. Top 10 Best-Selling Products (by Qty)", color="Quantity", color_continuous_scale="plasma",
                              hover_data={"Quantity": ":d", "Description": True})
                fig4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"), xaxis=dict(tickformat="d"))
                st.plotly_chart(fig4, use_container_width=True)
                st.info("💡 **Product Velocity Insight:** High-volume items are dominated by low-cost commodity/party items. These require optimized inventory control due to high turnover rates.")

            with c2:
                top_prod_rev = active_df.groupby("Description")["TotalAmount"].sum().sort_values(ascending=False).head(10).reset_index()
                fig5 = px.bar(top_prod_rev, x="TotalAmount", y="Description", orientation="h",
                              title="5. Top 10 Revenue-Generating Products ($)", color="TotalAmount", color_continuous_scale="dense",
                              hover_data={"TotalAmount": ":$.2f", "Description": True})
                fig5.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig5, use_container_width=True)
                st.info("💡 **Product Revenue Insight:** Premium products contribute the largest share of absolute gross profits. Focus marketing campaigns on these high-margin hero items.")

            fig6 = px.histogram(active_df[active_df["UnitPrice"] <= 15], x="UnitPrice", nbins=50,
                                title="6. Distribution of Unit Prices (Clipped at $15)", color_discrete_sequence=["#3b82f6"])
            fig6.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="Unit Price ($)"), yaxis=dict(title="Count"))
            st.plotly_chart(fig6, use_container_width=True)
            st.info("💡 **Pricing Strategy Insight:** The catalog is heavily skewed toward low-cost items under $3.00, suggesting a retail environment built on high-volume, impulsive accessory shopping.")

        # 3. Temporal Tab (Charts 7, 8, 9, 10)
        with tab_temp:
            st.markdown("### 📅 Temporal Trends & Operational Peak Times")
            monthly_df = active_df.copy()
            monthly_df["InvoiceMonth"] = monthly_df["InvoiceDate"].dt.to_period("M").astype(str)
            
            monthly_sales = monthly_df.groupby("InvoiceMonth")["TotalAmount"].sum().reset_index()
            fig7 = px.line(monthly_sales, x="InvoiceMonth", y="TotalAmount", title="7. Monthly Revenue Trend ($)", markers=True, color_discrete_sequence=["#10b981"],
                           hover_data={"TotalAmount": ":$.2f", "InvoiceMonth": True})
            fig7.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig7, use_container_width=True)
            st.info("💡 **Seasonality Insight:** Revenue peaks drastically in Q4 (specifically October/November), indicative of heavy holiday shopping prep. Operations must scale logistics to support this seasonal surge.")

            monthly_tx = monthly_df.groupby("InvoiceMonth")["InvoiceNo"].nunique().reset_index()
            fig8 = px.line(monthly_tx, x="InvoiceMonth", y="InvoiceNo", title="8. Monthly Order Volume (Transactions)", markers=True, color_discrete_sequence=["#38bdf8"],
                           hover_data={"InvoiceNo": ":d", "InvoiceMonth": True})
            fig8.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(title="Order Count", tickformat="d"))
            st.plotly_chart(fig8, use_container_width=True)
            st.info("💡 **Order Volume Insight:** Order volume correlates closely with revenue trends, confirming that sales gains in late autumn are driven by a surge in transaction counts.")

            c1, c2 = st.columns(2)
            with c1:
                day_df = active_df.copy()
                day_df["DayOfWeek"] = day_df["InvoiceDate"].dt.day_name()
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"]
                day_counts = day_df.groupby("DayOfWeek")["InvoiceNo"].nunique().reindex(day_order).reset_index()
                fig9 = px.bar(day_counts, x="DayOfWeek", y="InvoiceNo", title="9. Order Volume by Day of Week", color="InvoiceNo", color_continuous_scale="purples",
                              hover_data={"InvoiceNo": ":d", "DayOfWeek": True})
                fig9.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, yaxis=dict(tickformat="d"))
                st.plotly_chart(fig9, use_container_width=True)
                st.info("💡 **Daily Activity Insight:** Thursdays and Tuesdays show the highest transactional volumes. Saturdays have zero transactions in this dataset, indicating operational downtime.")

            with c2:
                hour_df = active_df.copy()
                hour_df["Hour"] = hour_df["InvoiceDate"].dt.hour
                hour_counts = hour_df.groupby("Hour")["InvoiceNo"].nunique().reset_index()
                fig10 = px.line(hour_counts, x="Hour", y="InvoiceNo", title="10. Peak Ordering Times (Hour of Day)", markers=True, color_discrete_sequence=["#f59e0b"],
                                hover_data={"InvoiceNo": ":d", "Hour": True})
                fig10.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(tickformat="d"))
                st.plotly_chart(fig10, use_container_width=True)
                st.info("💡 **Peak Hour Insight:** Transactions peak between 10:00 AM and 3:00 PM. Schedule marketing promotions during these peak hours to maximize response rates.")

        # 4. Cart & Loyalty Tab (Charts 11, 12, 13, 14)
        with tab_spend:
            st.markdown("### 🛒 Cart & Loyalty Metrics")
            c1, c2 = st.columns(2)
            with c1:
                pos_amounts = active_df[active_df["Quantity"] > 0]["TotalAmount"]
                fig11 = px.histogram(pos_amounts[pos_amounts <= 100], x="TotalAmount", nbins=50,
                                     title="11. Distribution of Transaction Line Values ($0 - $100)", color_discrete_sequence=["#ec4899"],
                                     hover_data={"TotalAmount": ":$.2f"})
                fig11.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="Line Value ($)"))
                st.plotly_chart(fig11, use_container_width=True)
                st.info("💡 **Basket Value Insight:** The vast majority of transaction invoice lines represent values under $20. Product catalog items should be grouped or cross-sold to increase average basket sizes.")

            with c2:
                fig12 = px.histogram(active_df[active_df["Quantity"] <= 50], x="Quantity", nbins=50,
                                     title="12. Distribution of Quantity Ordered (Clipped at 50)", color_discrete_sequence=["#14b8a6"],
                                     hover_data={"Quantity": ":d"})
                fig12.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="Quantity per Invoice Line", tickformat="d"), yaxis=dict(tickformat="d"))
                st.plotly_chart(fig12, use_container_width=True)
                st.info("💡 **Quantity Insight:** Orders are concentrated in smaller quantities (1 to 12 items per line). Wholesale bulk-purchasing accounts for outliers that were clipped here.")

            scatter_df = active_df.sample(n=min(len(active_df), 1000), random_state=42)
            fig13 = px.scatter(scatter_df[scatter_df["UnitPrice"] <= 50], x="UnitPrice", y="Quantity",
                               title="13. Unit Price vs Quantity Ordered", color="TotalAmount", color_continuous_scale="electric",
                               hover_data={"UnitPrice": ":$.2f", "Quantity": ":d", "TotalAmount": ":$.2f"})
            fig13.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="Unit Price ($)"), yaxis=dict(title="Quantity Ordered", tickformat="d"))
            st.plotly_chart(fig13, use_container_width=True)
            st.info("💡 **Price Elasticity Insight:** High-quantity orders are concentrated exclusively among low unit-price items, confirming typical downward-sloping price elasticity of demand.")

            cohort_df = active_df.copy()
            cohort_df["FirstPurchaseMonth"] = cohort_df.groupby("CustomerID")["InvoiceDate"].transform("min").dt.to_period("M").astype(str)
            new_cust = cohort_df.groupby("FirstPurchaseMonth")["CustomerID"].nunique().reset_index()
            new_cust.columns = ["Cohort Month", "New Customers Acquired"]
            fig14 = px.bar(new_cust, x="Cohort Month", y="New Customers Acquired", title="14. New Customer Cohorts Acquisition", color="New Customers Acquired", color_continuous_scale="purp",
                           hover_data={"New Customers Acquired": ":d", "Cohort Month": True})
            fig14.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(tickformat="d"))
            st.plotly_chart(fig14, use_container_width=True)
            st.info("💡 **Acquisition Insight:** Cohort registration spiked significantly in early 2011, followed by standard customer acquisition patterns. Target retention activities to retain these older cohorts.")

        # 5. RFM Distributions (Chart 15)
        with tab_rfm:
            st.markdown("### 📊 RFM Distributions & Heatmap")
            if active_rfm is not None:
                corr_matrix = active_rfm[["Recency", "Frequency", "Monetary"]].corr()
                fig15 = px.imshow(corr_matrix, text_auto=".3f", title="15. Correlation Heatmap of RFM Features",
                                  color_continuous_scale="viridis", aspect="auto")
                fig15.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig15, use_container_width=True)
                st.info("💡 **Feature Correlation Insight:** Frequency and Monetary features exhibit a strong positive correlation (+0.46), suggesting that customers who order frequently also spend more in aggregate. Recency is negatively correlated with spend.")
            else:
                st.warning("Please construct the segmentation data first.")
    else:
        st.warning("Cleaned transactional dataset not found. Please load a dataset first.")

# ----------------- PAGE 3: CUSTOMER SEGMENTATION MODULE -----------------
elif menu == "🔍 Customer Segmentation Module":
    if active_rfm is not None:
        tab_viz, tab_pred = st.tabs(["🚀 Segment Visualizations", "🔮 Live Segment Predictor"])
        
        with tab_viz:
            st.markdown("### 🌌 Interactive 3D RFM Cluster Space")
            # Downsample for responsiveness
            sample_df = active_rfm.sample(n=min(len(active_rfm), 3000), random_state=42)
            fig_3d = px.scatter_3d(
                sample_df, x="Recency", y="Frequency", z="Monetary", color="Segment",
                color_discrete_map={"High-Value": "#10b981", "Regular": "#3b82f6", "Occasional": "#f59e0b", "At-Risk": "#ef4444"},
                opacity=0.75, log_y=True, log_z=True,
                hover_data={"Recency": ":d", "Frequency": ":d", "Monetary": ":$.2f", "Segment": True}
            )
            fig_3d.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                scene=dict(
                    xaxis=dict(title='Recency (Days)', gridcolor='rgba(255,255,255,0.05)', tickformat="d"),
                    yaxis=dict(title='Frequency (Orders) [Log]', type='log', gridcolor='rgba(255,255,255,0.05)', tickformat="d"),
                    zaxis=dict(title='Monetary ($) [Log]', type='log', gridcolor='rgba(255,255,255,0.05)', tickformat="$.2f"),
                    bgcolor="rgba(0,0,0,0)"
                ),
                margin=dict(l=0, r=0, b=0, t=10)
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            
        with tab_pred:
            st.markdown("### 🔮 Predict Customer Cohort Segment")
            if active_scaler is not None and active_kmeans is not None:
                col_in, col_out = st.columns([1, 1.2])
                
                with col_in:
                    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                    st.markdown("#### Input Metrics:")
                    r_in = st.slider("Recency (Days since last purchase)", 1, 365, 30)
                    f_in = st.slider("Frequency (Total purchase visits)", 1, 100, 5)
                    m_in = st.number_input("Monetary (Total Spend amount in USD)", value=500.0)
                    predict_btn = st.button("🔮 Predict Customer Segment", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with col_out:
                    if predict_btn:
                        try:
                            # Construct a pandas DataFrame with correct feature names to avoid UserWarning/Errors
                            input_df = pd.DataFrame([[r_in, f_in, m_in]], columns=["Recency", "Frequency", "Monetary"])
                            input_log = np.log1p(input_df)
                            scaled_vals = active_scaler.transform(input_log)
                            pred_cluster = int(active_kmeans.predict(scaled_vals)[0])
                            
                            if active_cluster_mapping is not None:
                                pred_segment = active_cluster_mapping.get(pred_cluster, "Regular")
                            else:
                                pred_segment = "Regular"
                        except Exception as pred_err:
                            st.error(f"Prediction failed: {pred_err}")
                            pred_segment = "Regular"

                        # Expected Purchases and CLV Predictions via BG/NBD & Gamma-Gamma Fitters
                        expected_purchases = 0.0
                        predicted_clv = 0.0
                        if active_bgf is not None and active_ggf is not None:
                            try:
                                # Scale inputs to lifetimes format
                                scaled_freq = max(0.0, float(f_in - 1))
                                scaled_rec = float(max(0.0, float(f_in - 1)))
                                scaled_T = scaled_rec + (float(r_in) / 30.0)
                                monetary_val = float(m_in) / max(1.0, float(f_in))

                                expected_purchases = float(active_bgf.conditional_expected_number_of_purchases_up_to_time(
                                    1.0, scaled_freq, scaled_rec, scaled_T
                                ))
                                
                                if scaled_freq > 0:
                                    predicted_clv = float(active_ggf.customer_lifetime_value(
                                        active_bgf,
                                        pd.Series([scaled_freq]),
                                        pd.Series([scaled_rec]),
                                        pd.Series([scaled_T]),
                                        pd.Series([monetary_val]),
                                        time=3, # 3 months CLV projection
                                        discount_rate=0.01,
                                        freq="M"
                                    ).iloc[0])
                            except Exception as clv_err:
                                expected_purchases = 0.0
                                predicted_clv = 0.0
                        
                        badge_class = "badge-regular"
                        description = ""
                        strategy = ""
                        
                        if pred_segment == "High-Value":
                            badge_class = "badge-high"
                            description = "Regular, frequent, recent, and big spenders. They represent your most profitable customer cohort."
                            strategy = "Enlist in VIP club, offer premium preview products, send exclusive gifts, and seek referral feedback."
                        elif pred_segment == "Regular":
                            badge_class = "badge-regular"
                            description = "Steady, loyal purchasers who buy moderately but consistently."
                            strategy = "Offer loyalty points multipliers, cross-sell regular bundles, and send periodic feedback surveys."
                        elif pred_segment == "Occasional":
                            badge_class = "badge-occasional"
                            description = "Recent buyers who order rarely or sporadically, with smaller overall expenditure."
                            strategy = "Incentivize subsequent purchase with limited-time coupons or introduce low-cost bestseller recommendations."
                        elif pred_segment == "At-Risk":
                            badge_class = "badge-atrisk"
                            description = "Dormant customers who haven't completed a transaction in a long time. High chance of churn."
                            strategy = "Trigger automated 'We Miss You' emails with deep discounts, execute win-back surveys, or suggest high-popularity products."
                        
                        st.markdown(f"""
<div class='premium-card' style='height: 100%; text-align: center;'>
<h4 style='color: #a855f7; margin-bottom: 20px;'>🔮 Prediction Output</h4>
<span class='badge {badge_class}' style='font-size: 1.4rem; padding: 12px 30px; margin-bottom: 20px;'>{pred_segment}</span>
<div style='display: flex; justify-content: space-around; margin: 15px 0; background: rgba(255,255,255,0.03); padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);'>
<div>
<div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;'>Expected Purchases (30d)</div>
<div style='font-size: 1.4rem; font-weight: 800; color: #38bdf8; margin-top: 4px;'>{expected_purchases:.2f}</div>
</div>
<div>
<div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;'>Predicted CLV (3m)</div>
<div style='font-size: 1.4rem; font-weight: 800; color: #10b981; margin-top: 4px;'>${predicted_clv:,.2f}</div>
</div>
</div>
<div style='text-align: left; margin-top: 15px;'>
<h5 style='color: #f8fafc; margin-bottom: 5px;'>📋 Profile:</h5>
<p style='color: #94a3b8; font-size: 0.95rem; line-height: 1.6;'>{description}</p>
<h5 style='color: #f8fafc; margin-bottom: 5px; margin-top: 15px;'>💡 Focus Action:</h5>
<p style='color: #94a3b8; font-size: 0.95rem; line-height: 1.6;'>{strategy}</p>
</div>
</div>
""", unsafe_allow_html=True)
                    else:
                        st.markdown("""
<div class='premium-card' style='height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; min-height: 250px;'>
    <span style='font-size: 3rem; margin-bottom: 15px;'>👈</span>
    <h4 style='color: #94a3b8; margin: 0 0 10px 0;'>Awaiting Input Parameters</h4>
    <p style='color: #64748b; font-size: 0.95rem; margin: 0;'>
        Adjust the customer metrics on the left panel and click the <b>Predict Customer Segment</b> button to compute classifications.
    </p>
</div>
""", unsafe_allow_html=True)
            else:
                st.error("Scaler/KMeans models could not be resolved.")
    else:
        st.warning("Segmentation data is empty.")

# ----------------- PAGE 4: PRODUCT RECOMMENDATION ENGINE -----------------
elif menu == "🎯 Product Recommendation Engine":
    tab_prod_rec, tab_cust_rec = st.tabs(["🛍️ Product-Based Recommendations", "👤 Customer-Based Recommendations"])
    
    with tab_prod_rec:
        if active_product_list is not None and active_df is not None:
            col_search, col_recs = st.columns([1, 1.2])
            
            with col_search:
                st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
                st.markdown("### 🔎 Catalog Product Search")
                selected_product = st.selectbox(
                    "Select E-Commerce Product:",
                    options=active_product_list,
                    index=0,
                    key="prod_selectbox"
                )
                
                strategy = st.radio("Recommendation Strategy:", [
                    "Content-Based (Description TF-IDF similarity)",
                    "Item-Based Collaborative Filtering",
                    "Market Basket Analysis (Association Rules)"
                ])
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_recs:
                st.markdown("### ✨ Top Recommendations")
                if strategy == "Content-Based (Description TF-IDF similarity)":
                    prod_df, cosine_sim = build_content_recommender(active_df)
                    recs = get_content_recommendations(selected_product, prod_df, cosine_sim)
                    if recs:
                        for item, score in recs:
                            st.markdown(f"""
                            <div class='rec-card'>
                                <span class='rec-icon'>📖</span>
                                <div style='flex-grow:1;'>
                                    <span class='rec-text'>{item}</span>
                                    <div style='font-size:0.8rem; color:#ec4899; margin-top:4px;'>Description Match: <b>{score}%</b></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No content similarities found.")
                        
                elif strategy == "Item-Based Collaborative Filtering":
                    # Fallback to pre-trained if matching or calculate on raw
                    if p_recommendations is not None:
                        raw_recs = p_recommendations.get(selected_product, [])
                        if not raw_recs:
                            raw_recs = p_recommendations.get("__FALLBACK__", [])
                        for item in raw_recs:
                            rec_name = item[0] if isinstance(item, (tuple, list)) else item
                            score = item[1] if isinstance(item, (tuple, list)) else 1.0
                            st.markdown(f"""
                            <div class='rec-card'>
                                <span class='rec-icon'>🎁</span>
                                <div style='flex-grow:1;'>
                                    <span class='rec-text'>{rec_name}</span>
                                    <div style='font-size:0.8rem; color:#a855f7; margin-top:4px;'>Similarity Factor: <b>{score:.4f}</b></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Dynamic Item-Based CF (Memory Optimized)
                        df_filtered = active_df.dropna(subset=["Description", "CustomerID"]).copy()
                        customer_u = df_filtered["CustomerID"].astype('category')
                        description_u = df_filtered["Description"].astype('category')
                        
                        row = description_u.cat.codes
                        col = customer_u.cat.codes
                        data = np.ones(len(df_filtered))
                        
                        item_user_sparse = csr_matrix((data, (row, col)), shape=(len(description_u.cat.categories), len(customer_u.cat.categories)))
                        
                        if selected_product in description_u.cat.categories:
                            target_idx = description_u.cat.categories.get_loc(selected_product)
                            item_sim = cosine_similarity(item_user_sparse[target_idx], item_user_sparse).flatten()
                            
                            similar_indices = item_sim.argsort()[::-1][1:6]
                            
                            for idx in similar_indices:
                                prod = description_u.cat.categories[idx]
                                score = item_sim[idx]
                                st.markdown(f"""
                                <div class='rec-card'>
                                    <span class='rec-icon'>🎁</span>
                                    <div style='flex-grow:1;'>
                                        <span class='rec-text'>{prod}</span>
                                        <div style='font-size:0.8rem; color:#a855f7; margin-top:4px;'>Similarity Factor: <b>{score:.4f}</b></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("Product not found in user transactional matrix.")
                            
                elif strategy == "Market Basket Analysis (Association Rules)":
                    if p_association_rules is not None:
                        raw_rules = p_association_rules.get(selected_product, [])
                        if not raw_rules:
                            raw_rules = p_association_rules.get("__FALLBACK__", [])
                        for item in raw_rules:
                            rec_name = item[0] if isinstance(item, (tuple, list)) else item
                            lift = item[1] if isinstance(item, (tuple, list)) else 1.0
                            conf = item[2] if isinstance(item, (tuple, list)) else 1.0
                            st.markdown(f"""
                            <div class='rec-card'>
                                <span class='rec-icon'>🛒</span>
                                <div style='flex-grow:1;'>
                                    <span class='rec-text'>{rec_name}</span>
                                    <div style='font-size:0.8rem; color:#38bdf8; margin-top:4px;'>Confidence: <b>{conf*100:.1f}%</b> | Lift: <b>{lift:.2f}x</b></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Dynamic Basket Rules
                        basket = active_df.groupby(["InvoiceNo", "Description"])["Quantity"].sum().unstack().reset_index().fillna(0).set_index("InvoiceNo")
                        basket = basket.map(lambda x: 1 if x > 0 else 0)
                        
                        if selected_product in basket.columns:
                            support_target = basket[selected_product].mean()
                            rules = []
                            for prod in basket.columns:
                                if prod != selected_product:
                                    support_both = (basket[selected_product] & basket[prod]).mean()
                                    if support_both >= 0.01:
                                        confidence = support_both / support_target
                                        lift = confidence / basket[prod].mean()
                                        rules.append((prod, lift, confidence))
                            rules = sorted(rules, key=lambda x: x[1], reverse=True)[:5]
                            for r_name, lift, conf in rules:
                                st.markdown(f"""
                                <div class='rec-card'>
                                    <span class='rec-icon'>🛒</span>
                                    <div style='flex-grow:1;'>
                                        <span class='rec-text'>{r_name}</span>
                                        <div style='font-size:0.8rem; color:#38bdf8; margin-top:4px;'>Confidence: <b>{conf*100:.1f}%</b> | Lift: <b>{lift:.2f}x</b></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("Product not present in transaction matrix.")
        else:
            st.warning("Product list or dataset is missing.")
            
    with tab_cust_rec:
        if active_df is not None:
            st.markdown("### 👤 User-Based Collaborative Recommendations")
            customers = sorted(active_df["CustomerID"].dropna().unique().tolist())
            selected_customer = st.selectbox("Select Customer ID:", options=customers)
            
            if st.button("✨ Get Customer Recommendations"):
                user_recs = get_user_collaborative_recs(active_df, selected_customer)
                if user_recs:
                    for item, score in user_recs:
                        st.markdown(f"""
                        <div class='rec-card' style='border-left-color: #ec4899;'>
                            <span class='rec-icon' style='background: rgba(236,72,153,0.15);'>👤</span>
                            <div style='flex-grow:1;'>
                                <span class='rec-text'>{item}</span>
                                <div style='font-size:0.8rem; color:#ec4899; margin-top:4px;'>Co-purchase Mean Score: <b>{score}</b></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No collaborative recommendations matching this customer ID profile.")

# ----------------- PAGE 5: STATISTICAL HYPOTHESIS TESTS -----------------
elif menu == "🧪 Statistical Hypothesis Tests":
    st.markdown("### 🧪 Hypothesis Testing & Benchmarks")
    st.markdown("<p style='color:#94a3b8;'>Formulate and execute classical statistical checks on retail transaction spend values.</p>", unsafe_allow_html=True)
    
    if active_df is not None:
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("""
            <div class='premium-card' style='height:100%;'>
                <h4 style='color:#38bdf8; margin-top:0;'>🧪 1. Welch's t-test (UK vs Non-UK Sales)</h4>
                <p style='color:#94a3b8; font-size:0.9rem;'>Tests if the average line spending ($) of UK customers is statistically different from international transactions.</p>
            """, unsafe_allow_html=True)
            
            active_df["IsUK"] = active_df["Country"] == "United Kingdom"
            uk_spend = active_df[active_df["IsUK"]]["TotalAmount"]
            non_uk_spend = active_df[~active_df["IsUK"]]["TotalAmount"]
            
            t_stat, p_val = stats.ttest_ind(uk_spend, non_uk_spend, equal_var=False)
            st.write(f"T-Statistic: **{t_stat:.4f}**")
            st.write(f"P-Value: **{p_val:.4g}**")
            if p_val < 0.05:
                st.success("Reject H₀: There is a statistically significant difference between UK and international transaction values.")
            else:
                st.warning("Fail to reject H₀: No statistically significant difference detected.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c2:
            st.markdown("""
            <div class='premium-card' style='height:100%;'>
                <h4 style='color:#a855f7; margin-top:0;'>🧪 2. Pearson Correlation Test</h4>
                <p style='color:#94a3b8; font-size:0.9rem;'>Assesses the strength of linear relationship between Item Quantities and Unit Prices.</p>
            """, unsafe_allow_html=True)
            
            # Sample to prevent overflow
            sample_corr = active_df.sample(n=min(len(active_df), 10000), random_state=42)
            corr_coeff, p_val_corr = stats.pearsonr(sample_corr["Quantity"], sample_corr["UnitPrice"])
            
            st.write(f"Pearson Correlation Coefficient: **{corr_coeff:.4f}**")
            st.write(f"P-Value: **{p_val_corr:.4g}**")
            if p_val_corr < 0.05:
                st.success(f"Reject H₀: There is a statistically significant linear correlation ({corr_coeff:.2f}) between Quantity and UnitPrice.")
            else:
                st.warning("Fail to reject H₀: No correlation detected.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='premium-card'>
            <h4 style='color:#ec4899; margin-top:0;'>🧪 3. One-Way ANOVA (Weekday Spending Variations)</h4>
            <p style='color:#94a3b8; font-size:0.9rem;'>Tests if customer transaction spends vary significantly across different days of the week.</p>
        """, unsafe_allow_html=True)
        
        active_df["DayName"] = active_df["InvoiceDate"].dt.day_name()
        days = active_df["DayName"].unique()
        groups = [active_df[active_df["DayName"] == day]["TotalAmount"] for day in days]
        
        f_stat, p_val_anova = stats.f_oneway(*groups)
        st.write(f"F-Statistic: **{f_stat:.4f}**")
        st.write(f"P-Value: **{p_val_anova:.4g}**")
        if p_val_anova < 0.05:
            st.success("Reject H₀: Spending values vary significantly depending on the day of the week.")
        else:
            st.warning("Fail to reject H₀: Daily spending variations are not statistically significant.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Dataset not loaded.")
