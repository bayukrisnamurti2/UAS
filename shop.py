import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv("customer/Customers.csv")

columns_to_update = {
    'Annual Income ($)': 'Annual Income',
    'Spending Score (1-100)': 'Spending Score'
}

df = df.rename(columns=columns_to_update)

st.header("isi dataset")
st.write(X)