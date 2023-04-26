import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the founders data
founders_df = pd.read_csv('founders.csv')

# Combine the text data for each founder into a single column
founders_df['text'] = founders_df['Skills'] + ' ' + founders_df['Experiences'] + ' ' + founders_df['Motivation'] + ' ' + founders_df['Values']

# Create a TF-IDF vectorizer to convert text data into numerical vectors
vectorizer = TfidfVectorizer()

# Vectorize the text data for each founder
founder_vectors = vectorizer.fit_transform(founders_df['text'])
import streamlit as st

# Define the app title and a brief description
st.title('Startup Founder Matcher')
st.write('Enter your details below and we will match you with potential co-founders based on your skills, experience, motivation, and values.')

# Define the input fields for the user to enter their details
input_skills = st.text_input('Skills')
input_experience = st.text_input('Experiences')
input_motivation = st.text_input('Motivation')
input_values = st.text_input('Values')

# Define the number of top matches to display
num_matches = st.slider('Number of Matches', min_value=1, max_value=10, value=5)

# Calculate the matches based on the input
input_vector = vectorizer.transform([input_skills + " " + input_experience + " " + input_motivation + " " + input_values])
similarities = cosine_similarity(input_vector, founder_vectors)
similarities_list = similarities[0].tolist()
similar_founders = founders_df.loc[:, 'Founder'].tolist()
match_dict = dict(zip(similar_founders, similarities_list))
sorted_matches = sorted(match_dict.items(), key=lambda x: x[1], reverse=True)[1:num_matches+1]

# Display the top matches to the user
st.write('Here are your top matches:')
for match in sorted_matches:
    st.write(match[0], 'with a match score of', round(match[1]*100, 2), '%')

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

# Sort the complementary matches by their similarity score
sorted_complementary_matches = sorted(complementary_matches, key=lambda x: x[1], reverse=True)

# Display the complementary matches to the user
st.write('Here are your complementary matches:')
for match in sorted_complementary_matches:
    st.write(match[0], 'with a match score of', match[1], '%')

