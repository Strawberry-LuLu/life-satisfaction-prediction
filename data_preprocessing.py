import pandas as pd

with open('life_satisfaction.csv', 'r', encoding='utf-8') as f:
 content = f.read().replace('"', '')
 
lines = [line.strip() for line in content.split('\n') if line.strip()]

header = lines[0].split(',')

data = []
for line in lines[1:]:
 values = line.split(',')
 if len(values) == len(header):
  data.append([v.strip() for v in values])
 
df = pd.DataFrame(data, columns=header)

numeric_cols = ['2013', '2018', '2022']
for col in numeric_cols:
 df[col] = pd.to_numeric(df[col], errors='coerce')
 
df.to_excel('cleaned_data.xlsx', index=False, engine='openpyxl')