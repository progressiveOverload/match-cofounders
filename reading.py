import streamlit as st
import pandas as pd

# Create a file uploader widget using Streamlit
file = st.file_uploader("Upload a CSV file", type="csv")

if file is not None:
    # Load the uploaded file into a Pandas DataFrame
    df = pd.read_csv(file)

    # Display the DataFrame in Streamlit
    st.write(df)
