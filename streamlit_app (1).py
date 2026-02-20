import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Mall Customer Clustering",
    page_icon="🛍️",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
h1, h2, h3 {
    color: white;
}
.prediction-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    color: white;
    text-align: center;
}
.cluster-info {
    background-color: rgba(255,255,255,0.95);
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD DATA
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

# -----------------------------------
# PREPROCESS + TRAIN MODEL
# -----------------------------------
df_encoded = df.copy()
df_encoded['Gender'] = (df_encoded['Gender'] == 'Male').astype(int)

features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded[features])

kmeans = KMeans(n_clusters=5, random_state=42)
df_encoded['Cluster'] = kmeans.fit_predict(scaled_data)

# -----------------------------------
# CLUSTER INFO
# -----------------------------------
CLUSTER_INFO = {
    0: "High Value Customers",
    1: "Potential Target",
    2: "Average Customers",
    3: "Loyal Customers",
    4: "Budget Conscious"
}

# -----------------------------------
# TITLE
# -----------------------------------
st.markdown("<h1>🛍️ Mall Customer Clustering Prediction</h1>", unsafe_allow_html=True)

# -----------------------------------
# INPUT SECTION
# -----------------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", int(df.Age.min()), int(df.Age.max()), 30)
    income = st.slider("Annual Income (k$)",
                       int(df['Annual Income (k$)'].min()),
                       int(df['Annual Income (k$)'].max()), 50)
    spending = st.slider("Spending Score (1-100)", 1, 100, 50)

with col2:
    st.metric("Total Customers", len(df))
    st.metric("Average Age", round(df.Age.mean(), 1))
    st.metric("Average Income", round(df['Annual Income (k$)'].mean(), 1))

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.button("🚀 Predict Cluster", use_container_width=True):

    input_data = pd.DataFrame({
        'Gender': [1 if gender == "Male" else 0],
        'Age': [age],
        'Annual Income (k$)': [income],
        'Spending Score (1-100)': [spending]
    })

    scaled_input = scaler.transform(input_data)
    cluster = kmeans.predict(scaled_input)[0]

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
        <div class="prediction-box">
            <h2>Cluster {cluster}</h2>
            <h3>{CLUSTER_INFO[cluster]}</h3>
        </div>
    """, unsafe_allow_html=True)

    # -----------------------------------
    # CLUSTER STATS
    # -----------------------------------
    cluster_data = df_encoded[df_encoded['Cluster'] == cluster]

    st.markdown("### 📊 Cluster Statistics")

    colA, colB, colC = st.columns(3)

    colA.metric("Customers in Cluster", len(cluster_data))
    colB.metric("Avg Income", round(cluster_data['Annual Income (k$)'].mean(), 1))
    colC.metric("Avg Spending", round(cluster_data['Spending Score (1-100)'].mean(), 1))

    # -----------------------------------
    # 3D VISUALIZATION
    # -----------------------------------
    st.markdown("### 📈 3D Cluster Visualization")

    fig = px.scatter_3d(
        df_encoded,
        x='Age',
        y='Annual Income (k$)',
        z='Spending Score (1-100)',
        color=df_encoded['Cluster'].astype(str),
        height=600
    )

    fig.add_scatter3d(
        x=[age],
        y=[income],
        z=[spending],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Your Input'
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------
    # PIE CHART
    # -----------------------------------
    st.markdown("### 🥧 Cluster Distribution")

    cluster_counts = df_encoded['Cluster'].value_counts().sort_index()

    fig_pie = go.Figure(data=[go.Pie(
        labels=[f"Cluster {i}" for i in cluster_counts.index],
        values=cluster_counts.values
    )])

    st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")
st.markdown("<p style='text-align:center;color:white;'>Powered by KMeans | Streamlit Deployment Safe Version</p>", unsafe_allow_html=True)