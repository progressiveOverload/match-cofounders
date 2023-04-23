import pandas as pd

# read the original CSV file
df = pd.read_csv('2023-02-27-yc-companies.csv')

# select only the desired column
new_df = df['founders_names']

# save the new dataframe as CSV file
new_df.to_csv('new_file.csv', index=False)
