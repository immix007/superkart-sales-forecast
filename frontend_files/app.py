"""
SuperKart Sales Forecasting - Streamlit Frontend
Interactive web UI for single and batch sales predictions via the Flask backend.
"""

import io
import requests
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SuperKart Sales Forecaster",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://backend:7860"
PREDICT_URL = f"{BACKEND_URL}/v1/predict"
BATCH_URL = f"{BACKEND_URL}/v1/predictbatch"

st.markdown(
    """
    <style>
    .main-header { background: linear-gradient(135deg,#1a1a2e,#0f3460); padding:2rem;
        border-radius:16px; text-align:center; margin-bottom:2rem; }
    .main-header h1 { color:#e94560; font-size:2.6rem; font-weight:800; margin:0; }
    .main-header p { color:#a8b2d8; margin:0.5rem 0 0; }
    .result-card { background:linear-gradient(135deg,#0f3460,#1a1a2e); border:1px solid #e94560;
        border-radius:12px; padding:1.5rem; text-align:center; margin-top:1.5rem; }
    .result-card .label { color:#a8b2d8; font-size:0.9rem; text-transform:uppercase; }
    .result-card .value { color:#e94560; font-size:2.4rem; font-weight:800; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header"><h1>🛒 SuperKart Sales Forecaster</h1>'
            '<p>AI-powered revenue predictions for SuperKart outlets</p></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## About")
    st.info("Predicts **total sales revenue** (Rs.) for a SuperKart product-store combination.")
    st.markdown("---")
    st.markdown("### Dataset\n- 8,764 records\n- 4 store types\n- 3 city tiers")
    st.markdown("### Model\n- XGBoost (tuned)")
    st.caption("SuperKart Model Deployment — Great Learning AIML")

tab1, tab2 = st.tabs(["🔍 Single Prediction", "📊 Batch Prediction"])

with tab1:
    st.markdown("#### Enter Product & Store Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Product Attributes**")
        pw = st.number_input("Weight (kg)", 1.0, 50.0, 12.66, 0.01, key="pw")
        psc = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"], key="psc")
        paa = st.number_input("Allocated Area Ratio", 0.001, 1.0, 0.027, 0.001, format="%.3f", key="paa")
        mrp = st.number_input("MRP (Rs.)", 10.0, 500.0, 117.08, 0.01, key="mrp")
        pic = st.selectbox("Category Code", ["FD", "DR", "NC"], key="pic")
        ptc = st.selectbox("Type Category", ["Perishables", "Non Perishables"], key="ptc")
    with col2:
        st.markdown("**Store Attributes**")
        ss = st.selectbox("Store Size", ["Small", "Medium", "High"], key="ss")
        slct = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"], key="slct")
        stype = st.selectbox("Store Type",
            ["Supermarket Type1","Supermarket Type2","Supermarket Type3","Departmental Store","Food Mart"],
            key="stype")
        say = st.number_input("Store Age (Years)", 1, 60, 17, 1, key="say")
    with col3:
        st.markdown("**Prediction**")
        if st.button("🚀 Predict Sales", use_container_width=True):
            payload = {"Product_Weight": pw, "Product_Sugar_Content": psc,
                       "Product_Allocated_Area": paa, "Product_MRP": mrp,
                       "Store_Size": ss, "Store_Location_City_Type": slct,
                       "Store_Type": stype, "Product_Id_char": pic,
                       "Store_Age_Years": int(say), "Product_Type_Category": ptc}
            try:
                resp = requests.post(PREDICT_URL, json=payload, timeout=30)
                if resp.status_code == 200:
                    pred = resp.json()["prediction"]
                    st.markdown(f'<div class="result-card"><div class="label">Predicted Revenue</div>'
                                f'<div class="value">Rs. {pred:,.2f}</div></div>', unsafe_allow_html=True)
                    st.success("Prediction successful!")
                else:
                    st.error(f"API Error {resp.status_code}: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to Flask backend.")
            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    st.markdown("#### Upload CSV for Batch Predictions")
    st.info("CSV must contain: Product_Weight, Product_Sugar_Content, Product_Allocated_Area, "
            "Product_MRP, Store_Size, Store_Location_City_Type, Store_Type, "
            "Product_Id_char, Store_Age_Years, Product_Type_Category")
    f = st.file_uploader("Choose CSV", type=["csv"])
    if f:
        df_prev = pd.read_csv(f); f.seek(0)
        st.dataframe(df_prev.head(), use_container_width=True)
        if st.button("📊 Run Batch Prediction", use_container_width=True):
            try:
                resp = requests.post(BATCH_URL, files={"file": ("batch.csv", f.read(), "text/csv")}, timeout=60)
                if resp.status_code == 200:
                    preds = pd.Series(resp.json(), name="Predicted_Sales (Rs.)")
                    preds.index = preds.index.astype(int)
                    res = df_prev.copy()
                    res["Predicted_Sales (Rs.)"] = preds.sort_index().values
                    st.success(f"Done! {len(res)} predictions generated.")
                    st.dataframe(res, use_container_width=True)
                    st.download_button("⬇️ Download CSV", res.to_csv(index=False).encode(), "predictions.csv", "text/csv", use_container_width=True)
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")
