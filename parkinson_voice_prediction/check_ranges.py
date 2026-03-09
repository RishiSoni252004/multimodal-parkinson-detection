import pandas as pd
df = pd.read_csv('dataset/parkinsons.data')
df = df.drop(columns=['name', 'status'])
print(df.describe().T[['mean', 'std', 'min', 'max']])
