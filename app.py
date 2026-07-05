import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="RetailPulse Dashboard", layout="wide")

st.title(" RetailPulse - AI Powered Customer Analytics")
st.write("Retail Sales Analysis Dashboard")

uploaded_file = st.file_uploader("Upload Retail CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    if len(numeric_columns) > 0:

        selected = st.selectbox("Select Column", numeric_columns)

        fig, ax = plt.subplots()
        ax.hist(df[selected], bins=20)
        ax.set_title(selected)

        st.pyplot(fig)

    if len(numeric_columns) >= 2:

        st.subheader("Customer Segmentation (K-Means)")

        X = df[numeric_columns[:2]].dropna()

        model = KMeans(n_clusters=3, random_state=42)
        clusters = model.fit_predict(X)

        fig2, ax2 = plt.subplots()

        ax2.scatter(
            X.iloc[:,0],
            X.iloc[:,1],
            c=clusters
        )

        ax2.set_xlabel(numeric_columns[0])
        ax2.set_ylabel(numeric_columns[1])

        st.pyplot(fig2)

else:
    st.info("Please upload a retail dataset.")