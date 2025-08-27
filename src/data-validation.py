import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, kendalltau
from sklearn.preprocessing import OneHotEncoder

df = pd.read_excel('cleaned_data_without_technical.xlsx')

def age_to_numeric(age_range):
 try:
  age_range = age_range.replace("Y","")
  parts = age_range.split('-')
  lower = int(parts[0])
  upper = int(parts[1])
  return (lower + upper) / 2
 except:
  return np.nan

df['age_numeric'] = df['age'].apply(age_to_numeric)

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(df[['life_sat']])
life_sat_encoded = ohe.transform(df[['life_sat']])
life_sat_df = pd.DataFrame(life_sat_encoded,
columns=ohe.get_feature_names_out(['life_sat']))
df = pd.concat([df, life_sat_df], axis=1)
df = df.drop('life_sat', axis=1)

print("\n--- Анализ корреляции (числовые переменные) ---")
numeric_cols = ['age_numeric', '2013', '2018', '2022']
numeric_df = df[numeric_cols].dropna()

plt.figure(figsize=(10, 8))
corr_matrix = numeric_df.corr(method='pearson')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Корреляционная матрица числовых переменных")
plt.savefig('Корреляционная матрица.png', dpi=300, bbox_inches='tight')
plt.close()
print("Корреляционная матрица:\n", corr_matrix)

print("\n--- Проверка баланса классов (категориальные переменные) ---")
categorical_cols = ['isced11', 'sex', 'age', 'geo']
for col in categorical_cols:
 plt.figure(figsize=(8, 6))
 df[col].value_counts().plot(kind='bar')
 plt.title(f"Распределение классов в '{col}'")
 plt.xlabel(col)
 plt.ylabel("Количество, ед.")
 plt.savefig(f'{col}_Распределение классов.png', dpi=300,
bbox_inches='tight')
 plt.close()
 
 class_balance = df[col].value_counts(normalize=True)
 print(f"Баланс классов для '{col}':\n{class_balance}")
68
numeric_cols = ['2013', '2018', '2022', 'age_numeric']
categorical_cols = ['isced11', 'sex', 'age', 'geo']

print("\n=== Статистический анализ ===")
print("\n* Географическое положение (geo) и категории life_sat:")
life_sat_categories = [col for col in df.columns if
col.startswith('life_sat_')]
for cat in life_sat_categories:
 contingency_table = pd.crosstab(df['geo'], df[cat])
 stat, p, dof, expected = chi2_contingency(contingency_table)
 print(f" Chi-squared test ({cat}): p-value={p:.4f}")
 n = len(df)
 cramers_v = np.sqrt(stat / (n * (min(contingency_table.shape) - 1)))
 print(f" Cramer's V: {cramers_v:.3f}")
 
print("\n* Пол (sex) и категории life_sat:")
for cat in life_sat_categories:
 contingency_table = pd.crosstab(df['sex'], df[cat])
 stat, p, dof, expected = chi2_contingency(contingency_table)
 print(f" Chi-squared test ({cat}): p-value={p:.4f}")
 n = len(df)
 cramers_v = np.sqrt(stat / (n * (min(contingency_table.shape) - 1)))
 print(f" Cramer's V: {cramers_v:.3f}")
 
print("\n* Возрастные группы (age) и категории life_sat:")
df['age_rank'] = df['age'].astype('category').cat.codes
for cat in life_sat_categories:
 tau, p = kendalltau(df['age_rank'], df[cat])
 print(f" Kendall's tau ({cat}): τ={tau:.3f}, p={p:.4f}")
