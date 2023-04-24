import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define a function to load the data
@st.cache_data
def load_data():
    data = pd.read_csv("founders.csv")
    return data

# Define a function to rank the potential co-founders
def rank_founders(founders_df, input_founder):
    # Create a new column in the DataFrame that combines the skills and experiences columns
    founders_df['skills_experiences'] = founders_df['Skills'] + " " + founders_df['Experiences']

    # Define the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the vectorizer on the skills and experiences column
    tfidf = vectorizer.fit_transform(founders_df['skills_experiences'])

    # Calculate the cosine similarity between the input founder and all the other founders
    input_index = founders_df[founders_df['Founder'] == input_founder].index[0]
    cosine_similarities = cosine_similarity(tfidf[input_index], tfidf)

    # Sort the cosine similarities in descending order and extract the top 10 most similar founders
    similar_indices = cosine_similarities.argsort()[0][-11:-1]
    similar_founders = founders_df.iloc[similar_indices]['Founder'].tolist()
    similarities = cosine_similarities[0][similar_indices].tolist()

    # Create a dictionary of the ranked co-founders and their similarity scores
    ranked_founders = {}
    for founder, similarity in zip(similar_founders, similarities):
        ranked_founders[founder] = round(similarity*100, 2)

    return ranked_founders

# Load the data
founders_df = load_data()

# Define the sidebar inputs
input_founder = st.sidebar.selectbox("Select a founder", founders_df['Founder'].unique())


# Rank the potential co-founders based on the input
ranked_founders = rank_founders(founders_df, input_founder)

# Display the ranked co-founders in the main area
st.title("Ranking of potential co-founders")
for founder, similarity in ranked_founders.items():
    st.write("{}: {}% similarity".format(founder, similarity))
