import pandas as pd

data = pd.read_csv("dataset/student_data.csv")

print(data)

print("\nDataset Information:")
print(data.info())

print("\nFirst 5 Students:")
print(data.head())