import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define a function to load the data
@st.cache
def load_data():
    data = pd.read_csv("founders.csv")
    return data

# Define a function to rank the potential co-founders based on multiple factors
def rank_founders(founders_df, input_founder):
    # Extract the motivation, values, skills, and experience of the input founder
    input_motivation = founders_df[founders_df['Founder'] == input_founder]['Motivation'].iloc[0]
    input_values = founders_df[founders_df['Founder'] == input_founder]['Values'].iloc[0]
    input_skills = founders_df[founders_df['Founder'] == input_founder]['Skills'].iloc[0]
    input_experience = founders_df[founders_df['Founder'] == input_founder]['Experiences'].iloc[0]

    # Create a new column in the DataFrame that combines the skills, experience, motivation, and values columns
    founders_df['skills_experience_motivation_values'] = founders_df['Skills'] + " " + founders_df['Experiences'] + " " + founders_df['Motivation'] + " " + founders_df['Values']

    # Define the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the vectorizer on the skills, experience, motivation, and values column
    tfidf = vectorizer.fit_transform(founders_df['skills_experience_motivation_values'])

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

    # Calculate the complementary matches based on the input
    complementary_matches = []
    input_vector = vectorizer.transform([input_skills + " " + input_experience + " " + input_motivation + " " + input_values])
    for i, founder in enumerate(similar_founders):
        founder_motivation = founders_df[founders_df['Founder'] == founder]['Motivation'].iloc[0]
        founder_values = founders_df[founders_df['Founder'] == founder]['Values'].iloc[0]
        founder_skills = founders_df[founders_df['Founder'] == founder]['Skills'].iloc[0]
        founder_experience = founders_df[founders_df['Founder'] == founder]['Experience'].iloc[0]
        founder_vector = vectorizer.transform([founder_skills + " " + founder_experience + " " + founder_motivation + " " + founder_values])
        similarity = cosine_similarity(input_vector, founder_vector)[0][0]
        complementary_match = (founder, round(similarity*100, 2))
        complementary_matches.append(complementary_match)

    # Sort the complementary matches by similarity score in descending order
    complementary_matches.sort(key=lambda x: x[1], reverse=True)

    # Print the complementary matches
    print("Complementary Matches:")
    for match in complementary_matches:
        print("- Founder:", match[0])
        print("  Similarity Score:", match[1], "%")

