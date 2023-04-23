import streamlit as st
import pandas as pd

def main():
    # Set page title
    st.set_page_config(page_title="CSV File Editor")

    # Set page header
    st.title("CSV File Editor")

    # Allow user to upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Load CSV data into a Pandas DataFrame
        data = pd.read_csv(uploaded_file)

        # Display original data
        st.subheader("Original Data")
        st.write(data)

        # Drop selected columns
        cols_to_drop = ['Year Founded', 'Mapping Location', 'Description', 'Categories', 'Y Combinator Year', 'Y Combinator Session', 'Investors', 'Amounts raised in different funding rounds','Office Address', 'Headquarters (City)', 'Headquarters (US State)', 'Headquarters (Country)', 'Logo',
         'Seed-DB / Mattermark Profile', 'Crunchbase / Angel List Profile',]
        data = data.drop(cols_to_drop, axis=1)
        data = data.dropna(subset=['Founders'])

        # Display modified data
        st.subheader("Modified Data")
        st.write(data)

        # Allow user to download modified CSV file
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download Modified CSV",
            data=csv,
            file_name="modified_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
