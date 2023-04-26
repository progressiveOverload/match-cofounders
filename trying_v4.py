import pandas as pd

# Load the test data into a DataFrame
test_data = pd.read_csv('founder.csv')

# Split the test data into input and expected output
input_founders = test_data['Founder'].tolist()
expected_output = [rank_founders(train_data, founder) for founder in input_founders]

# Add the expected output to the test data DataFrame
test_data['Expected Output'] = expected_output

# Export the test data to a CSV file
test_data.to_csv('test_founders.csv', index=False)
