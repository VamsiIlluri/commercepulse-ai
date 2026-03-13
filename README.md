# CommercePulse AI 🛒
## E-Commerce Sales & Customer Intelligence Platform

A fully-coded (no AI API) e-commerce analytics platform built with **Python + Streamlit**.
Upload any e-commerce CSV and instantly get ML-powered insights.

---

## 🚀 Quick Start

### Step 1: Clone / Download the project
```bash
# If using git:
git clone <your-repo-url>
cd commercepulse-ai

# Or just download and unzip the folder
```

### Step 2: Create a virtual environment (recommended)
```bash
# Create virtual env
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## 📂 Project Structure
```
commercepulse-ai/
├── app.py                        # 🏠 Home page — CSV upload
├── pages/
│   ├── 1_Sales_Dashboard.py      # 📊 Revenue trends, KPIs, charts
│   ├── 2_Customer_Intelligence.py # 👥 RFM segmentation + churn
│   ├── 3_Sales_Forecasting.py    # 📈 30-day revenue forecast
│   ├── 4_Product_Intelligence.py  # 🛒 Market basket analysis
│   └── 5_Smart_Insights.py       # 💡 Auto-generated business report
├── utils/
│   ├── data_loader.py            # CSV loading + column auto-detection
│   ├── metrics.py                # KPI calculations (revenue, growth etc.)
│   ├── segmentation.py           # RFM + K-Means clustering
│   ├── churn_model.py            # Random Forest churn prediction
│   ├── forecasting.py            # Polynomial regression forecasting
│   └── recommendations.py       # Market basket + product scoring
├── sample_data/                  # Place Kaggle datasets here
└── requirements.txt
```

---

## 📊 Datasets to Use

### Option 1: UK Online Retail (Recommended)
1. Go to: https://www.kaggle.com/datasets/carrie1/ecommerce-data
2. Download `data.csv`
3. Rename to `uk_retail.csv`
4. Place in `sample_data/` folder

### Option 2: Superstore Sales
1. Go to: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final
2. Download `Sample - Superstore.csv`
3. Rename to `superstore.csv`
4. Place in `sample_data/` folder

### Option 3: Upload Your Own
- Any CSV with at least a **date** column and **revenue/amount** column
- The app auto-detects column names (see below)

---

## 🔍 Supported CSV Column Names

The app automatically maps your columns — no renaming needed!

| Detected As | Your CSV might call it |
|-------------|----------------------|
| `date`      | OrderDate, purchase_date, InvoiceDate, Date |
| `revenue`   | Amount, Total, UnitPrice, TotalPrice, Sales |
| `customer_id` | CustomerID, client_id, user_id |
| `product`   | Description, ProductName, SKU, StockCode |
| `quantity`  | Qty, Units, Quantity |
| `category`  | Category, Department, Type |
| `region`    | Country, State, City, Region |
| `order_id`  | OrderID, InvoiceNo, TransactionID |

---

## 🧠 ML Algorithms Used

| Feature | Algorithm | Library |
|---------|-----------|---------|
| Customer Segmentation | K-Means Clustering | scikit-learn |
| Churn Prediction | Random Forest Classifier | scikit-learn |
| Sales Forecasting | Polynomial Regression | scikit-learn |
| Market Basket Analysis | Association Rules (custom Apriori) | pandas/numpy |
| Product Scoring | Weighted composite scoring | pandas |
| Auto Insights | Rule-based logic | Pure Python |

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push your code to a GitHub repository
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set main file as `app.py`
5. Click Deploy — your app gets a public URL!

**Note**: Don't include large CSV files in git. Use `.gitignore` for `sample_data/*.csv`

---

## 📚 Understanding the Code

Each file has detailed comments explaining:
- What the function does
- What algorithm is used and why
- What inputs/outputs look like

Start with reading files in this order:
1. `utils/data_loader.py` — how data is loaded
2. `utils/metrics.py` — how KPIs are calculated
3. `app.py` — how the upload UI works
4. `pages/1_Sales_Dashboard.py` — how charts are built
5. `utils/segmentation.py` — how K-Means works
6. `utils/churn_model.py` — how Random Forest works
7. `utils/forecasting.py` — how forecasting works

---

## 🐛 Common Issues

**"ModuleNotFoundError"**
→ Make sure you activated your virtual environment and ran `pip install -r requirements.txt`

**"No module named utils"**
→ Run `streamlit run app.py` from inside the `commercepulse-ai/` folder

**"Sample dataset not found"**
→ Download from Kaggle and place in `sample_data/` folder (see above)

**"Not enough data for clustering"**
→ Your dataset has too few customers. Use the UK Retail dataset for full features.

---

## 🔧 Bug Fixes Applied (v2)

| Bug | Fix |
|-----|-----|
| `infer_datetime_format` crash on pandas 3.x | Removed deprecated argument — uses `errors="coerce"` instead |
| UK Online Retail revenue = unit price only | Added heuristic to detect unit-price datasets and multiply by quantity |
| Market basket shows 0 rules | Slider min lowered from 0.005 → 0.001, default from 0.02 → 0.005 |
| Forecasting R² = 0.002 | Model now fits on weekly-aggregated data (less noise) instead of daily |
| `segmentation.py` docstring said `segment_label` | Corrected to `segment` (actual column name) |

## 📥 UK Online Retail Setup
1. Download from: https://www.kaggle.com/datasets/carrie1/ecommerce-data
2. Rename to exactly: `uk_retail.csv`
3. Place in `sample_data/` folder
4. Load via: Home page → "UK Online Retail" dropdown → Load Sample Dataset

**Note:** The app automatically multiplies UnitPrice × Quantity to get correct line totals.
