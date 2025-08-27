import pandas as pd

df = pd.read_excel('cleaned_data.xlsx')

print("Исходные столбцы в датасете:")
print(df.columns.tolist())

technical_columns = ['freq', 'statinfo', 'unit']

df_clean = df.drop(columns=technical_columns, errors='ignore')

print("\nСтолбцы после удаления:")
print(df_clean.columns.tolist())

columns_to_fill_numeric = ['2013', '2018', '2022']

for column in columns_to_fill_numeric:
 if column in df_clean.columns:
  median_value = df_clean[column].median()
  df_clean[column] = df_clean[column].fillna(median_value)
  print(f"Столбец '{column}': Пропуски заполнены медианой 
({median_value:.2f})")
 else:
  print(f"Предупреждение: Столбец '{column}' не найден в DataFrame.")
 
df_clean.to_excel('cleaned_data_without_technical.xlsx', index=False)