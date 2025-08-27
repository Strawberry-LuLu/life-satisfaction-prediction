import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('cleaned_data.xlsx')

numeric_cols = ['2013', '2018', '2022']
for col in numeric_cols:
 df[col] = pd.to_numeric(df[col], errors='coerce')
 
categorical_cols = ['freq', 'life_sat', 'statinfo', 'isced11', 'sex', 'age',
'unit', 'geo']

plt.figure(figsize=(20, 25))
plt.suptitle('Распределение переменных', fontsize=20, y=0.98)
for i, col in enumerate(numeric_cols, 1):
 plt.subplot(4, 3, i)
 sns.histplot(df[col].dropna(), bins=20, color='skyblue')
 plt.title(f'{col}')
 plt.xlabel('Значение')
 plt.ylabel('Частота')
for i, col in enumerate(categorical_cols, len(numeric_cols)+1):
 plt.subplot(4, 3, i)

 if df[col].nunique() > 15:
  top_categories = df[col].value_counts().nlargest(15).index
  filtered_df = df[df[col].isin(top_categories)]
  sns.countplot(data=filtered_df, y=col, order=top_categories,
 palette='viridis')
  plt.title(f'Топ-15 категорий: {col}')
 else:
  sns.countplot(data=df, y=col, palette='viridis')
  plt.title(f'{col}')

 plt.xlabel('Количество наблюдений, ед.')
 
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Распределение всех переменных.png', bbox_inches='tight',
dpi=300)
plt.show()

var_counts = pd.DataFrame({
 'Variable': numeric_cols + categorical_cols,
 'Count': [df[col].count() for col in numeric_cols + categorical_cols],
 'Type': ['Числовая']*len(numeric_cols) +
['Категориальная']*len(categorical_cols)
})

var_counts = var_counts.sort_values('Count', ascending=False)

bar = sns.barplot(data=var_counts, x='Count', y='Variable', hue='Type',
dodge=False)

plt.title('Общее количество наблюдений по переменным')
plt.xlabel('Количество наблюдений, ед.')
65
plt.ylabel('Переменная')
plt.legend(title='Тип переменной')
plt.grid(axis='x', alpha=0.3)

for p in bar.patches:
 width = p.get_width()
 plt.text(width + 50, p.get_y() + p.get_height()/2, f'{int(width)}',
 ha='left', va='center')
 
plt.tight_layout()
plt.savefig('Общее количество наблюдейний по переменным.png',
bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(12, 6))

var_counts['Pct'] = var_counts['Count'] / len(df) * 100

bar = sns.barplot(data=var_counts, x='Pct', y='Variable', hue='Type',
dodge=False)

plt.title('Доля заполненных значений по переменным')
plt.xlabel('Процент заполненных значений, %')
plt.ylabel('Переменная')
plt.legend(title='Тип переменной')
plt.grid(axis='x', alpha=0.3)

for p in bar.patches:
 width = p.get_width()
 plt.text(width + 1, p.get_y() + p.get_height()/2, f'{width:.1f}%',
 ha='left', va='center')
plt.tight_layout()
plt.savefig('Доля заполненных значений по переменным.png',
bbox_inches='tight', dpi=300)
plt.show()
