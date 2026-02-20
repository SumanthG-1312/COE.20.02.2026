K-Means Customer Clustering

A machine learning project that implements K-means clustering to segment mall customers into distinct groups based on their spending behavior and demographics.

📋 Project Overview

This project performs customer segmentation analysis using the K-means clustering algorithm. It includes Jupyter notebooks for experimentation and a Streamlit web application for interactive visualization and prediction of customer clusters.

✨ Features

Customer Segmentation: Clusters mall customers into groups using K-means algorithm

Interactive Web App: Streamlit-based interface for exploring clusters and making predictions

Data Visualization: Multiple visualizations including 2D/3D cluster plots and distributions

Model Persistence: Pre-trained models saved with joblib for quick predictions

Data Processing: Standardized features for optimal clustering performance

Exploratory Analysis: Jupyter notebooks for detailed analysis and experimentation

📁 Project Structure
K-mean/
├── Streamlit_app.py                                    
├── K_mean_clustering.ipynb                             
├── k-mean.ipynb                                        
├── Lab 1 - Classroom Exercise_KMeans_Clustering.ipynb  
├── Mall_Customers.csv                                  
├── clustered_data.csv                                  
├── clustered_mall_customers.csv                        
├── test_fix.py                                         
├── requirements.txt                                    
└── README.md                                           
🛠️ Installation

Clone or download the project

cd "K-mean"

Create a virtual environment (optional but recommended)

python -m venv venv
venv\Scripts\activate  # On Windows

Install dependencies

pip install -r requirements.txt
🚀 Usage
Run the Streamlit Web App
streamlit run Streamlit_app.py

The app will open in your default browser at:
http://localhost:8501

Run Jupyter Notebooks
jupyter notebook

Then select any of the .ipynb files to explore the clustering analysis step-by-step.

📊 Dataset

Mall_Customers.csv contains customer data with the following features:

Customer ID

Gender

Age

Annual Income

Spending Score (1-100)

The data is preprocessed and standardized before applying K-means clustering.

📦 Dependencies

All required packages are listed in requirements.txt:

streamlit – Web application framework

pandas – Data manipulation and analysis

numpy – Numerical computing

scikit-learn – Machine learning library for K-means

joblib – Model serialization

plotly – Interactive visualizations

matplotlib & seaborn – Static plotting libraries

🎯 How It Works

Data Loading: Import customer data from CSV

Preprocessing: Clean data and handle missing values

Feature Scaling: Standardize features using StandardScaler

Clustering: Apply K-means algorithm with optimal number of clusters

Visualization: Create interactive plots showing customer segments

Prediction: Predict cluster for new customer data

📝 Notes

The K-means model is trained on standardized features (Age, Annual Income, Spending Score)

StandardScaler is used to normalize features to have mean 0 and standard deviation 1

Pre-trained models are saved in the working directory for faster predictions

👨‍💻 Course Information

This is a student project from Vignan Institute of Technology and Science Data Science / Machine Learning course on K-means clustering techniques.

📧 Support

For questions or issues, refer to the Jupyter notebooks which contain detailed explanations and comments.

Last Updated: February 2026
