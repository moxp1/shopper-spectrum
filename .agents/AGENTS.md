# Workspace Customization Rules

Keep and follow the daily Git commit schedule below when making commits to this repository. All commits must be backdated using the specified timestamps and files.

## Git Commit Schedule

### 1. June 25 - Setup & Download (2 Commits)
* `docs: initialize project documentation and specifications`
  - Date: `2026-06-25T10:00:00`
  - Files: `README.md`, `.gitignore`
* `feat: add utility script to download transaction database`
  - Date: `2026-06-25T14:30:00`
  - Files: `download_dataset.py`

### 2. June 26 - Notebook & Basic Models (2 Commits)
* `feat: add jupyter notebook generation script`
  - Date: `2026-06-26T09:00:00`
  - Files: `create_notebook.py`
* `feat: initial implementation of pipeline loader and data cleaner`
  - Date: `2026-06-26T15:15:00`
  - Files: `build_models.py`

### 3. June 27 - Cleaning pipeline (2 Commits)
* `feat: clean cancelled invoices and administrative stockcodes`
  - Date: `2026-06-27T11:00:00`
  - Files: `requirements.txt`
* `feat: generate off-line rfm clustered visual outputs`
  - Date: `2026-06-27T16:00:00`
  - Files: `Shopper_Spectrum.ipynb`

### 4. June 28 - RFM Calculations (2 Commits)
* `feat: calculate Recency, Frequency, and Monetary metrics`
  - Date: `2026-06-28T10:30:00`
  - Files: `scaler.pkl`
* `feat: add return ratio metrics to RFM calculations`
  - Date: `2026-06-28T15:00:00`
  - Files: `cluster_mapping.pkl`

### 5. June 29 - Clustering Models (2 Commits)
* `feat: scale features and train K-Means customer segmentation`
  - Date: `2026-06-29T09:00:00`
  - Files: `kmeans.pkl`
* `feat: build collaborative filtering product recommendations`
  - Date: `2026-06-29T14:00:00`
  - Files: `recommendations.pkl`, `product_list.pkl`

### 6. June 30 - Recommendations (2 Commits)
* `feat: mine market basket association rules`
  - Date: `2026-06-30T10:00:00`
  - Files: `association_rules.pkl`
* `feat: construct Streamlit dashboard views and analytics tabs`
  - Date: `2026-07-01T09:00:00` (Shifted to represent June 30 work)
  - Files: `app.py`

### 7. July 1 - Dashboard & CLV (3 Commits)
* `feat: train and save BG/NBD and Gamma-Gamma CLV models`
  - Date: `2026-07-01T09:00:00`
  - Files: `bgf.pkl`, `ggf.pkl`
* `feat: integrate predictive CLV models into Streamlit analytics`
  - Date: `2026-07-01T11:30:00`
  - Files: `app.py`, `build_models.py`
* `feat: add click trigger to segmentation and optimize html blocks`
  - Date: `2026-07-01T15:00:00`
  - Files: `app.py`
