import pandas as pd

# Load the data from CSV file
data = pd.read_csv("founders.csv")

from sklearn.model_selection import train_test_split

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print(train_data.shape)
print(test_data.shape)
