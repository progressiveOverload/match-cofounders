import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Load data
data = pd.read_csv("founders.csv")

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Vectorize training data
tfidf_train = vectorizer.fit_transform(train_data['skills_experiences'])

# Train logistic regression model
model = LogisticRegression()
model.fit(tfidf_train, train_data['label'])

# Vectorize test data
tfidf_test = vectorizer.transform(test_data['skills_experiences'])

# Predict test data
y_pred = model.predict(tfidf_test)

# Calculate F1-score
f1 = f1_score(test_data['label'], y_pred, average='weighted')

print(f"F1-score: {f1}")
