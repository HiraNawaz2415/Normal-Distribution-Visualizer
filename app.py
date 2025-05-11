import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Inject custom CSS
st.markdown("""
    <style>
    /* Make all slider label texts black */
    .stSlider label, .stSlider span, .stSlider div {
        color: black !important;
    }
    /* Optional: make sidebar text black */
    .stSidebar {
        color: black !important;
    }
    /* Set background color and font */
    body {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #2e86de;
        text-align: center;
    }
    .stSidebar > div:first-child {
        background-color: #eaf2f8;
        padding: 1rem;
        border-radius: 10px;
    }
    .css-1d391kg {
        background-color: #ffffff;
    }
    .stDownloadButton > button {
        background-color: #2e86de;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        transition: 0.3s;
    }
    .stDownloadButton > button:hover {
        background-color: #1b4f72;
        transform: scale(1.02);
    }
    .stSlider > div {
        color: #2e4053;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Normal Distribution Visualizer")

# Sidebar for user inputs
st.sidebar.header("🔢 Input Parameters")
mean = st.sidebar.slider("Mean (μ)", -10.0, 10.0, 0.0)
std_dev = st.sidebar.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0)
size = st.sidebar.slider("Number of Samples", 100, 5000, 1000)

# Generate random normal distribution data
data = np.random.normal(loc=mean, scale=std_dev, size=size)

# Plot Histogram with KDE
st.subheader("📊 Normal Distribution Plot")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data, kde=True, bins=30, color="skyblue", label="Generated Data")
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
y = norm.pdf(x, mean, std_dev) * size * (max(data) - min(data)) / 30
ax.plot(x, y, color="red", label="Theoretical Normal Curve")
ax.legend()
st.pyplot(fig)

# Show statistics
st.write("### 📌 Data Statistics")
st.write(f"**Mean:** `{np.mean(data):.2f}`")
st.write(f"**Standard Deviation:** `{np.std(data):.2f}`")

# Probability Calculator
st.subheader("📉 Probability Calculator")
x_value = st.number_input("Enter a value (X) to find P(X < value):", value=mean)
probability = norm.cdf(x_value, mean, std_dev)
st.success(f"**P(X < {x_value}) = {probability:.4f}**")

# CDF Plot
st.subheader("📈 Cumulative Distribution Function (CDF)")
fig_cdf, ax_cdf = plt.subplots(figsize=(8, 5))
x_cdf = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
y_cdf = norm.cdf(x_cdf, mean, std_dev)
ax_cdf.plot(x_cdf, y_cdf, color="green", label="CDF Curve")
ax_cdf.axvline(x_value, color="red", linestyle="--", label=f"P(X < {x_value})")
ax_cdf.legend()
st.pyplot(fig_cdf)

# Download generated data
st.subheader("📂 Download Data")
df = pd.DataFrame(data, columns=["Generated Values"])
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Data as CSV", csv, "normal_distribution_data.csv", "text/csv")
