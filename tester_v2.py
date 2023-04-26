import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np


@st.cache_data
def load_data():
    data = pd.read_csv("founders.csv")
    return data


# Load the data
founders_df = load_data()

# Define the number of folds for cross-validation
num_folds = 5

# Shuffle the data
founders_df = founders_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the data into folds
kf = KFold(n_splits=num_folds)
fold_indices = kf.split(founders_df)

# Define the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Define the list of similarity scores for each fold
scores = []

# Perform cross-validation
for fold, (train_indices, test_indices) in enumerate(fold_indices):
    print("Fold", fold + 1)

    # Split the data into training and testing sets
    train_data = founders_df.iloc[train_indices]
    test_data = founders_df.iloc[test_indices]

    # Fit and transform the vectorizer on the training data
    tfidf_train = vectorizer.fit_transform(train_data['skills_experiences'])

    # Calculate the cosine similarity between the input founder and all the other founders in the test data
    for input_founder in test_data['Founder']:
        input_index = train_data[train_data['Founder'] == input_founder].index[0]
        cosine_similarities = cosine_similarity(tfidf_train[input_index],
                                                vectorizer.transform(test_data['skills_experiences']))

        # Sort the cosine similarities in descending order and extract the top 10 most similar founders
        similar_indices = cosine_similarities.argsort()[0][-11:-1]
        similar_founders = test_data.iloc[similar_indices]['Founder'].tolist()
        similarities = cosine_similarities[0][similar_indices].tolist()

        # Create a dictionary of the ranked co-founders and their similarity scores, sorted by similarity in descending order
        ranked_founders = {}
        for founder, similarity in zip(similar_founders, similarities):
            ranked_founders[founder] = round(similarity * 100, 2)

        ranked_founders = dict(sorted(ranked_founders.items(), key=lambda x: x[1], reverse=True))

        # Print the ranked co-founders and their similarity scores
        print("Input founder:", input_founder)
        for founder, similarity in ranked_founders.items():
            print("{}: {}% complementary to the pick".format(founder, similarity))

        # Calculate the accuracy of the model by comparing the predicted co-founders to the actual co-founders
        actual_founders = test_data[test_data['Founder'] == input_founder]['Co-founders'].tolist()[0].split(", ")
        predicted_founders = list(ranked_founders.keys())
        num_correct = len(set(actual_founders).intersection(set(predicted_founders)))
        accuracy = num_correct / len(actual_founders)
        print("Accuracy:", accuracy)

        # Add the accuracy to the list of scores for this fold
        scores.append(accuracy)

# Calculate the overall accuracy of the model
mean_accuracy = np.mean(scores)
print("Mean accuracy:", mean_accuracy)
