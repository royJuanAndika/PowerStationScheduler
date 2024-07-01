import pandas as pd

# Create a sample DataFrame
data = {'Column A': [10, 15, 12], 'Column B': [20, 25, 18]}
df = pd.DataFrame(data)

# Get the row count
row_count = df.shape[0]
column_count = df.shape[1]
print(df)
print(f"Number of rows in the DataFrame: {row_count}")
print(f"Number of columns in the DataFrame: {column_count}")
