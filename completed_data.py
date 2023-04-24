import streamlit as st
import pandas as pd

# Set page title
st.set_page_config(page_title='CSV Uploader')

# Create a file uploader widget
csv_file = st.file_uploader('Upload a CSV file', type=['csv'])

# If a file is uploaded
if csv_file is not None:
    # Read the file using Pandas
    df = pd.read_csv(csv_file)

    # Drop rows with missing data in the 'experiences' column
    df = df.dropna(subset=['Experiences'])

    # Display the contents of the file as a table
    st.write(df)

    # Add a download button for the cleaned CSV data
    csv_cleaned = df.to_csv(index=False)
    st.download_button(
        label='Download cleaned CSV data',
        data=csv_cleaned,
        file_name='cleaned_data.csv',
        mime='text/csv'
    )
