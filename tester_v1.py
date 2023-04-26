import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the data
data = pd.read_csv("founders.csv")

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create the multi-label binarizer
mlb = MultiLabelBinarizer()

# Create the X and y matrices
X_train = train_data[['Skills', 'Experiences']].values
y_train = mlb.fit_transform(train_data['Labels'])

X_test = test_data[['Skills', 'Experiences']].values
y_test = mlb.transform(test_data['Labels'])

# Define the multi-label classifier
classifier = OneVsRestClassifier(LogisticRegression())

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
