# SuperKart Sales Forecasting - Model Deployment

> **Great Learning AIML Final Project** | Model Deployment Module

An end-to-end machine learning solution to forecast product-level sales revenue for SuperKart outlets. Built with **Random Forest** and **XGBoost**, deployed via **Flask REST API** + **Streamlit UI**, containerized with **Docker**, and hosted on **GitHub Codespaces**.

## Repository Structure

```
superkart-sales-forecast/
|-- backend_files/
|   |-- app.py                  # Flask REST API
|   |-- superkart_model.joblib  # Serialized ML model (added after training)
|   |-- requirements.txt
|   `-- Dockerfile
|-- frontend_files/
|   |-- app.py                  # Streamlit UI
|   |-- requirements.txt
|   `-- Dockerfile
|-- docker-compose.yml          # Orchestrates both containers
|-- .devcontainer/
|   `-- devcontainer.json       # GitHub Codespaces config
`-- README.md
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/v1/predict` | POST | Single record inference (JSON) |
| `/v1/predictbatch` | POST | Batch inference (CSV file) |

## Running on GitHub Codespaces

1. Open repo in GitHub Codespaces
2. In the terminal: `docker compose up --build -d`
3. In the Ports tab: set **7860** and **8501** to **Public**
4. Use the forwarded URL for port 7860 for notebook inference
5. Use the forwarded URL for port 8501 for the Streamlit UI

## Feature Columns

The model expects these 10 features:
- `Product_Weight`, `Product_Sugar_Content`, `Product_Allocated_Area`, `Product_MRP`
- `Store_Size`, `Store_Location_City_Type`, `Store_Type`
- `Product_Id_char`, `Store_Age_Years`, `Product_Type_Category`
