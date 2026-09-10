# 🛒 Smart Cart Customer Clustering E Commerce Segmentation System
An unsupervised learning project that segments retail customers into distinct groups based on their demographics, spending habits, and purchasing channels — enabling targeted marketing strategies for each segment.

---

## 📌 Project Overview

Retailers rarely treat all customers the same way, and neither should marketing. This project uses **unsupervised machine learning** to uncover natural customer segments hidden inside a retail dataset, then translates each segment into a concrete marketing strategy.

The pipeline covers the full workflow:

1. Data cleaning and feature engineering
2. Outlier removal
3. Correlation analysis
4. Categorical encoding and feature scaling
5. Dimensionality reduction with PCA
6. Finding the optimal number of clusters (Elbow Method + Silhouette Score)
7. Clustering with **K-Means** and **Agglomerative Clustering**
8. Cluster profiling and business strategy recommendations

---

## 📂 Project Structure

```
smart-cart-clsutering/
│
├── dataset/
│   ├── smartcart_customers.csv        # Raw customer dataset
│   └── SmartCart Clustering System.pdf   # Project brief / problem statement
│
├── smart_cart.ipynb                   # Main analysis notebook (full pipeline)
├── customer_data_updated.csv          # Cleaned & feature-engineered dataset
│
├── 01-pair_plots.png                  # Pairwise feature relationships
├── 02-heatmaps.png                    # Correlation heatmap
├── 03-scatter-pca.png                 # 2D PCA projection
├── 04-3d_projectio.png                # 3D PCA projection
├── 05_elbow_method.png                # Elbow method for optimal K
├── 06_Silhouetter_Score.png           # Silhouette score for optimal K
├── 07_k_value.png                     # Combined elbow + silhouette plot
├── 08-kmeans.png                      # K-Means cluster visualization (3D)
├── 09-agglomerative.png               # Agglomerative cluster visualization (3D)
├── 10_cluster_out.png                 # Cluster size distribution
├── 11_incomevsspending_out.png        # Income vs. Total Spending by cluster
│
└── README.md
```

---

## 📊 Dataset

**Source:** `smartcart_customers.csv`
**Size:** 2,240 customers × 22 original features

| Category | Columns |
|---|---|
| Demographics | `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome` |
| Enrollment | `Dt_Customer`, `Recency` |
| Spending (last 2 yrs) | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| Purchase Channels | `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` |
| Campaign Response | `Complain`, `Response` |

---

## 🧹 Data Preprocessing

- **Missing values:** 24 missing `Income` values filled with the median.
- **Feature engineering:**
  - `Age` — derived from `Year_Birth` (2026 − Year_Birth)
  - `Customer_Tenure` — days since enrollment, relative to the most recent customer
  - `Total_Spending` — sum of all product spending categories
  - `Total_Children` — `Kidhome` + `Teenhome`
  - `Education` — regrouped into `Undergraduate`, `Graduation`, `PostGraduate`
  - `Living_With` — regrouped `Marital_Status` into `Partner` or `Alone`
- **Dropped columns:** raw fields replaced by engineered features (`Year_Birth`, `Marital_Status`, `Kidhome`, `Teenhome`, `Dt_Customer`, and the individual `Mnt*` spending columns)
- **Outlier removal:** customers with `Age ≥ 90` or `Income ≥ 600,000` were excluded.

Cleaned data is saved to `customer_data_updated.csv`.

---

## 🔎 Exploratory Data Analysis

- **Pair plots** to inspect relationships between income, recency, response, age, spending, and children.
- **Correlation heatmap** highlighting the strongest relationships with `Total_Spending`:
  - Income ↔ Total Spending: **0.79**
  - Income ↔ Catalog Purchases: **0.69**
  - Income ↔ Store Purchases: **0.63**
  - Income ↔ Web Visits: **−0.65**
  - Catalog Purchases ↔ Total Spending: **0.78**
  - Store Purchases ↔ Total Spending: **0.68**

---

## ⚙️ Feature Engineering for Modeling

- **Encoding:** `Education` and `Living_With` one-hot encoded with `OneHotEncoder`.
- **Scaling:** All features standardized using `StandardScaler`.
- **Dimensionality reduction:** `PCA` applied to project the scaled feature space down to 2 and 3 components for visualization and clustering.

---

## 🎯 Choosing the Number of Clusters

Two complementary methods were used to select the optimal number of clusters (K):

- **Elbow Method** — inertia (WCSS) plotted across K = 1–10, with the elbow point detected programmatically using `kneed.KneeLocator`.
- **Silhouette Score** — computed across K = 2–10 to validate cluster cohesion and separation.

Both methods pointed to **K = 4** as the optimal number of clusters.

---

## 🧩 Clustering Models

Two clustering algorithms were applied to the PCA-reduced data and compared:

| Model | Algorithm | Parameters |
|---|---|---|
| K-Means | `sklearn.cluster.KMeans` | `n_clusters=4`, `random_state=42` |
| Agglomerative Clustering | `sklearn.cluster.AgglomerativeClustering` | `n_clusters=4`, `linkage="ward"` |

Both produced consistent, well-separated 4-cluster structures, visualized in 3D PCA space.

---

## 🧠 Cluster Insights & Marketing Strategy

| Cluster | Segment | Key Traits | Recommended Strategy |
|---|---|---|---|
| 🔴 **0** | Family Shoppers | More children, poor campaign response, mostly partnered, high web visits but low web/store/catalog purchases | Discounts & promotions to boost engagement |
| 🔵 **1** | Loyalty Members | Fewer children, slightly older, average response, high store/catalog/web purchases | Loyalty programs & reward incentives |
| 🟡 **2** | Digital Browsers | More children, mostly single, high web visits but very low purchases across channels | Heavy discounts & targeted online promotions |
| 🟢 **3** | Golden / Best ROI | Fewer children, slightly older, best campaign response, high store/catalog/web purchases | Premium services & exclusive personalized offers |

**Summary:**
- Clusters **1 & 3** → High income, high spending (retain & reward)
- Clusters **0 & 2** → Low income, low spending (convert & discount)

---

## 🛠️ Tech Stack

- **Python 3**
- **pandas**, **numpy** — data manipulation
- **matplotlib**, **seaborn** — visualization
- **scikit-learn** — `StandardScaler`, `OneHotEncoder`, `PCA`, `KMeans`, `AgglomerativeClustering`, `silhouette_score`
- **kneed** — automatic elbow-point detection

---

## ▶️ How to Run

1. Clone/download this repository and ensure the folder structure above is preserved.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn kneed
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook smart_cart.ipynb
   ```
4. Run all cells in order — outputs (cleaned CSV and all chart PNGs) will be regenerated in the project folder.

---


## 📈 Results at a Glance

- ✅ 2,240 customers reduced to **4 actionable segments**
- ✅ Cleaned dataset exported to `customer_data_updated.csv`
- ✅ 11 visualizations covering EDA, correlation, PCA projections, K selection, and cluster profiling
- ✅ Each segment paired with a concrete, ready-to-use marketing strategy