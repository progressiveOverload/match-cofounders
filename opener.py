import streamlit as st
import pandas as pd

st.title("YC Companies - Feb 27, 2023")

# Read CSV file
df = pd.read_csv("2023-02-27-yc-companies.csv")

# Display the dataframe
st.write(df)
