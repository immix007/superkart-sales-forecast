"""
SuperKart Sales Forecasting - Streamlit Frontend
Interactive web UI for single and batch sales predictions via the Flask backend.
"""

import io
import requests
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SuperKart Sales Forecaster",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Backend URL - frontend reaches backend via Docker network service name
BACKEND_URL = "http://backend:7860"
PREDICT_URL = f"{BACKEND_URL}/v1/predict"
BATCH_URL = f"{BACKEND_URL}/v1/predictbatch"

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #e94560; font-size: 2.6rem; font-weight: 800; margin: 0; }
    .main-header p { color: #a8b2d8; font-size: 1.05rem; margin: 0.5rem 0 0; }
    .result-card {
        background: linear-gradient(135deg, #0f3460, #1a1a2e);
        border: 1px solid #e94560;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-card .label { color: #a8b2d8; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1px; }
    .result-card .value { color: #e94560; font-size: 2.4rem; font-weight: 800; }
    .section-header {
        color: #e94560; font-size: 1.2rem; font-weight: 700;
        border-left: 4px solid #e94560; padding-left: 0.8rem; margin: 1.5rem 0 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """<div class="main-header">
        <h1>🛒 SuperKart Sales Forecaster</h1>
        <p>AI-powered sales revenue predictions for SuperKart outlets across India</p>
    </div>""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("## About")
    st.info(
        "This tool predicts the **total sales revenue** (Rs.) of a SuperKart product "
        "in a given store using a machine learning model trained on historical sales data."
    )
    st.markdown("---")
    st.markdown("### Key Metrics")
    st.markdown("- 📦 **8,764** product-store records")
    st.markdown("- 🏪 **4** store types")
    st.markdown("- 🌆 **3** city tiers")
    st.markdown("- 🤖 **XGBoost** model (tuned)")
    st.markdown("---")
    st.caption("SuperKart Model Deployment — Great Learning AIML")

# Tabs
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📊 Batch Prediction"])

# ── Tab 1: Single Prediction ──────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-header'>Enter Product & Store Details</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Product Attributes**")
        product_weight = st.number_input("Product Weight (kg)", 1.0, 50.0, 12.66, 0.01, key="pw")
        product_sugar_content = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"], key="psc")
        product_allocated_area = st.number_input("Allocated Area Ratio", 0.001, 1.0, 0.027, 0.001, format="%.3f", key="paa")
        product_mrp = st.number_input("Product MRP (Rs.)", 10.0, 500.0, 117.08, 0.01, key="mrp")
        product_id_char = st.selectbox("Product Category Code", ["FD", "DR", "NC"], key="pic",
                                        help="FD=Food, DR=Drinks, NC=Non-Consumables")
        product_type_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"], key="ptc")

    with col2:
        st.markdown("**Store Attributes**")
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"], key="ss")
        store_location = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"], key="slct")
        store_type = st.selectbox("Store Type",
                                   ["Supermarket Type1", "Supermarket Type2",
                                    "Supermarket Type3", "Departmental Store", "Food Mart"], key="stype")
        store_age = st.number_input("Store Age (Years)", 1, 60, 17, 1, key="say")

    with col3:
        st.markdown("**Run Prediction**")
        if st.button("🚀 Predict Sales", use_container_width=True, key="btn_single"):
            payload = {
                "Product_Weight": product_weight,
                "Product_Sugar_Content": product_sugar_content,
                "Product_Allocated_Area": product_allocated_area,
                "Product_MRP": product_mrp,
                "Store_Size": store_size,
                "Store_Location_City_Type": store_location,
                "Store_Type": store_type,
                "Product_Id_char": product_id_char,
                "Store_Age_Years": int(store_age),
                "Product_Type_Category": product_type_category,
            }
            try:
                with st.spinner("Generating prediction..."):
                    response = requests.post(PREDICT_URL, json=payload, timeout=30)
                if response.status_code == 200:
                    pred = response.json()["prediction"]
                    st.markdown(
                        f'<div class="result-card"><div class="label">Predicted Sales Revenue</div>'
                        f'<div class="value">Rs. {pred:,.2f}</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.success("Prediction successful!")
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to Flask backend. Ensure backend container is running.")
            except Exception as exc:
                st.error(f"Error: {exc}")

        st.markdown("---")
        st.markdown("**Sample Input**")
        st.json({
            "Product_Weight": 12.66, "Product_Sugar_Content": "Low Sugar",
            "Product_Allocated_Area": 0.027, "Product_MRP": 117.08,
            "Store_Size": "Medium", "Store_Location_City_Type": "Tier 2",
            "Store_Type": "Supermarket Type2", "Product_Id_char": "FD",
            "Store_Age_Years": 17, "Product_Type_Category": "Non Perishables",
        })

# ── Tab 2: Batch Prediction ───────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-header'>Upload CSV for Batch Predictions</div>", unsafe_allow_html=True)
    st.info(
        "Upload a CSV with columns: Product_Weight, Product_Sugar_Content, "
        "Product_Allocated_Area, Product_MRP, Store_Size, Store_Location_City_Type, "
        "Store_Type, Product_Id_char, Store_Age_Years, Product_Type_Category"
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_upload")
    if uploaded_file:
        try:
            df_preview = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()

        st.markdown(f"**Uploaded:** {len(df_preview)} rows x {len(df_preview.columns)} columns")
        st.dataframe(df_preview.head(), use_container_width=True)

        if st.button("📊 Run Batch Prediction", use_container_width=True, key="btn_batch"):
            try:
                with st.spinner("Running batch inference..."):
                    files = {"file": ("batch.csv", uploaded_file.read(), "text/csv")}
                    response = requests.post(BATCH_URL, files=files, timeout=60)
                if response.status_code == 200:
                    predictions = response.json()
                    preds = pd.Series(predictions, name="Predicted_Sales (Rs.)")
                    preds.index = preds.index.astype(int)
                    results_df = df_preview.copy()
                    results_df["Predicted_Sales (Rs.)"] = preds.sort_index().values
                    st.success(f"Predictions generated for {len(results_df)} records!")
                    st.dataframe(results_df, use_container_width=True)
                    st.download_button(
                        "⬇️ Download Predictions CSV",
                        results_df.to_csv(index=False).encode("utf-8"),
                        "superkart_predictions.csv", "text/csv",
                        use_container_width=True,
                    )
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to Flask backend.")
            except Exception as exc:
                st.error(f"Error: {exc}")
