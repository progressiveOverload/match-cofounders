import pandas as pd
import streamlit as st
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
    input_indices = founders_df[founders_df['Founder'] == input_founder].index
    if len(input_indices) == 0:
        return {}
    else:
        input_index = input_indices[0]
        cosine_similarities = cosine_similarity(tfidf[input_index], tfidf)

        # Sort the cosine similarities in descending order and extract the top 10 most similar founders
        similar_indices = cosine_similarities.argsort()[0][-11:-1]
        similar_founders = founders_df.iloc[similar_indices]['Founder'].tolist()
        similarities = cosine_similarities[0][similar_indices].tolist()

        # Create a dictionary of the ranked co-founders and their similarity scores, sorted by similarity in descending order
        ranked_founders = {}
        for founder, similarity in zip(similar_founders, similarities):
            ranked_founders[founder] = round(similarity * 100, 2)

        ranked_founders = dict(sorted(ranked_founders.items(), key=lambda x: x[1], reverse=True))

        return ranked_founders


# Load the data
founders_df = load_data()

# Split the data into training and testing sets
train_data = founders_df.sample(frac=0.8, random_state=1)
test_data = founders_df.drop(train_data.index)
# Create tabs
tabs = st.tabs(["Ranking", "Test Data"])

# Define the sidebar inputs
input_founder = tabs[0].selectbox("Select a founder", founders_df['Founder'].unique())

# Rank the potential co-founders based on the input
ranked_founders = rank_founders(founders_df, input_founder)

# Display the ranked co-founders in the main area
if not ranked_founders:
    tabs[0].write("No similar founders found.")
else:
    for founder, similarity in ranked_founders.items():
        tabs[0].write("{}: {}% complementary to the pick".format(founder, similarity))

# Display test data in a separate tab
tabs[1].write("Test Data")
test_data = pd.read_csv("test_founders.csv")
tabs[1].write(test_data)

# Load the test data into a DataFrame
test_data = pd.read_csv("test_founders.csv", error_bad_lines=False)


# Split the test data into input and expected output
input_founders = test_data['Founder'].tolist()
expected_output = test_data['Expected Output'].tolist()

# Test the function on the input and compare with expected output
for i in range(len(input_founders)):
    tabs[1].write("Test case #" + str(i + 1))
    tabs[1].write("Input founder: " + input_founders[i])
    tabs[1].write("Expected output: " + str(expected_output[i]))
    actual_output = rank_founders(train_data, input_founders[i])
    tabs[1].write("Actual output: " + str(actual_output))
    tabs[1].write("Test result: " + ("Pass" if expected_output[i] == actual_output else "Fail"))
