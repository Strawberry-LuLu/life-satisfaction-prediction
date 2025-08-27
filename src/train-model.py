import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

print("ПОДГОТОВКА ДАННЫХ ДЛЯ СЦЕНАРНОГО ПРОГНОЗИРОВАНИЯ")
df = pd.read_excel('cleaned_data_without_technical.xlsx')
print(f"Загружено данных: {df.shape[0]} строк")

for year in ['2013', '2018', '2022']:
 df[year] = df[year].astype(str).str.replace(',', '.').astype(float)
df['satisfaction'] = df[['2013', '2018', '2022']].mean(axis=1)
df = df.rename(columns={
 '2013': 'value_2013',
 '2018': 'value_2018',
 '2022': 'value_2022'
})

df = pd.get_dummies(df, columns=['life_sat', 'isced11', 'sex', 'age'],
 prefix=['aspect', 'edu', 'gender', 'age'],
dtype=np.float32)
geo_categories = df['geo'].unique()
geo_to_id = {geo: idx for idx, geo in enumerate(geo_categories)}
df['geo_id'] = df['geo'].map(geo_to_id).astype(np.int32)
print(f"Созданы ID для {len(geo_categories)} стран")

X = df.drop(['satisfaction', 'geo'], axis=1)
y = df['satisfaction'].astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42, stratify=df['geo_id']
)
print(f"Обучающая выборка: {X_train.shape[0]}, тестовая: {X_test.shape[0]} записей")

scaler = StandardScaler()
numeric_cols = ['value_2013', 'value_2018', 'value_2022']
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols]).astype(np.float32)
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols]).astype(np.float32)
train_numeric = X_train[numeric_cols].values.astype(np.float32)
test_numeric = X_test[numeric_cols].values.astype(np.float32)
categorical_cols = [col for col in X_train.columns
 if col.startswith(('aspect_', 'edu_', 'gender_',
'age_'))]
train_categorical = X_train[categorical_cols].values.astype(np.float32)
test_categorical = X_test[categorical_cols].values.astype(np.float32)
train_geo = X_train['geo_id'].values.astype(np.int32)
test_geo = X_test['geo_id'].values.astype(np.int32)

joblib.dump(geo_to_id, 'geo_mapping.pkl')
joblib.dump(scaler, 'standard_scaler.pkl')
print("Артефакты сохранены: geo_mapping.pkl, standard_scaler.pkl")

print("\nПОСТРОЕНИЕ МОДЕЛИ ДЛЯ СЦЕНАРНОГО ПРОГНОЗА")
num_geo = len(geo_categories)
embedding_dim = min(15, num_geo // 3)
numeric_input = Input(shape=(3,), name='numeric_input', dtype=tf.float32)
categorical_input = Input(shape=(len(categorical_cols),),
name='categorical_input', dtype=tf.float32)
geo_input = Input(shape=(1,), name='geo_input', dtype=tf.int32)
geo_embed = Embedding(input_dim=num_geo,
 output_dim=embedding_dim,
 name='geo_embedding')(geo_input)
geo_flat = Flatten()(geo_embed)
merged = Concatenate()([numeric_input, categorical_input, geo_flat])

x = Dense(128, activation='relu')(merged)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)

output = Dense(1, activation='linear')(x)
model = Model(inputs=[numeric_input, categorical_input, geo_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001),
 loss='mse',
 metrics=['mae',
tf.keras.metrics.RootMeanSquaredError(name='rmse')])
model.summary()

print("\nОБУЧЕНИЕ МОДЕЛИ")
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = model.fit(
 x=[train_numeric, train_categorical, train_geo],
 y=y_train.values,
 epochs=200,
 batch_size=256,
 validation_split=0.2,
 callbacks=[early_stop],
 verbose=1
)
model.save('life_satisfaction_regression_model.keras')

print("\nОЦЕНКА МОДЕЛИ НА ТЕСТОВЫХ ДАННЫХ")
test_loss, test_mae, test_rmse = model.evaluate(
 [test_numeric, test_categorical, test_geo],
 y_test.values,
 verbose=0
)

y_pred = model.predict([test_numeric, test_categorical, test_geo])
r2 = r2_score(y_test, y_pred)
print(f"Результаты на тестовых данных:")
print(f" MSE: {test_loss:.4f}")
print(f" MAE: {test_mae:.4f}")
print(f" RMSE: {test_rmse:.4f}")
print(f" R²: {r2:.4f}")

print("\nВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ РЕГРЕССИИ")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('Фактические vs Предсказанные значения удовлетворенности')
plt.xlabel('Фактическая удовлетворённость жизнью')
plt.ylabel('Предсказанная удовлетворённость')
plt.grid(True)
plt.savefig('regression_results.png')
plt.show()

print("\nАНАЛИЗ ВЛИЯНИЯ ФАКТОРОВ НА ПРОГНОЗ")
def plot_impact(model, country, aspect, education, gender, age_group):
 try:
  country_id = geo_to_id[country]
  base_values = X_train[numeric_cols].mean().values.astype(np.float32)
  aspect_vector = np.zeros(len(categorical_cols), dtype=np.float32)
  edu_vector = np.zeros(len(categorical_cols), dtype=np.float32)
  gender_vector = np.zeros(len(categorical_cols), dtype=np.float32)
  age_vector = np.zeros(len(categorical_cols), dtype=np.float32)
  aspect_col = f"aspect_{aspect}"
  edu_col = f"edu_{education}"
  gender_col = f"gender_{gender}"
  age_col = f"age_{age_group}"
  if aspect_col in categorical_cols:
   aspect_vector[categorical_cols.index(aspect_col)] = 1
  if edu_col in categorical_cols:
   edu_vector[categorical_cols.index(edu_col)] = 1
  if gender_col in categorical_cols:
   gender_vector[categorical_cols.index(gender_col)] = 1
  if age_col in categorical_cols:
   age_vector[categorical_cols.index(age_col)] = 1
  combined_vector = aspect_vector + edu_vector + gender_vector + age_vector
  current_values = np.linspace(
    df['value_2022'].min() * 0.9,
    df['value_2022'].max() * 1.1, 50,
    dtype=np.float32
  )

  economic_vals = np.tile(base_values, (50, 1))
  economic_vals[:, 2] = current_values
  categorical_data = np.tile(combined_vector, (50, 1))
  geo_data = np.full((50, 1), country_id, dtype=np.int32)
  predictions = model.predict([ economic_vals, categorical_data, geo_data]).flatten()

  plt.figure(figsize=(12, 7))
  plt.plot(current_values, predictions, lw=2.5)
  plt.title(f'Прогноз удовлетворённости для {country}\n({gender},{age_group}, {education}, {aspect})')
  plt.xlabel('Текущий уровень удовлетворенности (2022)')
  plt.ylabel('Предсказанный общий уровень удовлетворенности')
  plt.grid(True)
  plt.savefig(f'impact_{country}_{aspect}.png')
  plt.show()
  return current_values, predictions

 except Exception as e:
  print(f"Ошибка при построении графика влияния: {e}")
  return [], []

print("Генерация прогнозных сценариев для Германии (DE)...")
current_values, predictions = plot_impact(
 model,
 country='DE',
 aspect='ACCOM',
 education='ED0-2',
 gender='F',
 age_group='Y16-19'
)
scenarios_df = pd.DataFrame({
 'current_value': current_values,
 'predicted_satisfaction': predictions
})
scenarios_df.to_csv('germany_scenarios.csv', index=False)
print("Сценарии сохранены в germany_scenarios.csv")

print("\nГЕНЕРАЦИЯ ПРОГНОЗОВ ДЛЯ БУДУЩИХ СЦЕНАРИЕВ")
def predict_future_scenario(model, scaler, geo_to_id, categorical_cols,
 country, aspect, education, gender, age_group,
value_2013, value_2018, value_2022):
 try:
  country_id = geo_to_id[country]
  economic_vals = scaler.transform([[value_2013, value_2018,value_2022]])[0].astype(np.float32)
  aspect_vector = np.zeros(len(categorical_cols), dtype=np.float32)
  edu_vector = np.zeros(len(categorical_cols), dtype=np.float32)
  gender_vector = np.zeros(len(categorical_cols), dtype=np.float32)
  age_vector = np.zeros(len(categorical_cols), dtype=np.float32)
  aspect_col = f"aspect_{aspect}"
  edu_col = f"edu_{education}"
  gender_col = f"gender_{gender}"
  age_col = f"age_{age_group}"
  if aspect_col in categorical_cols:
   aspect_vector[categorical_cols.index(aspect_col)] = 1
  if edu_col in categorical_cols:
   edu_vector[categorical_cols.index(edu_col)] = 1
  if gender_col in categorical_cols:
   gender_vector[categorical_cols.index(gender_col)] = 1
  if age_col in categorical_cols:
   age_vector[categorical_cols.index(age_col)] = 1
  combined_vector = aspect_vector + edu_vector + gender_vector + age_vector
  input_data = {
   'numeric_input': np.array([economic_vals], dtype=np.float32),
   'categorical_input': np.array([combined_vector], dtype=np.float32),
   'geo_input': np.array([[country_id]], dtype=np.int32)
 }

  prediction = model.predict(input_data)[0][0]
  return prediction
 except Exception as e:
  print(f"Ошибка при прогнозировании сценария: {e}")
  return None

print("Прогноз для Германии (DE) в 2026 году:")
prediction = predict_future_scenario(
 model, scaler, geo_to_id, categorical_cols,
 country='DE',
 aspect='ACCOM',
 education='ED0-2',
 gender='F',
 age_group='Y16-19',
 value_2013=8.7,
 value_2018=7.2,
 value_2022=7.5
)
print(f"Предсказанная общая удовлетворённость жизнью: {prediction:.2f}")
countries = ['DE', 'FR', 'IT', 'ES']
predictions_2026 = []

print("\nСРАВНИТЕЛЬНЫЙ ПРОГНОЗ ДЛЯ 2026 ГОДА:")
for country in countries:
 country_data = df[df['geo'] == country]
 pred = predict_future_scenario(
 model, scaler, geo_to_id, categorical_cols,
 country=country,
 aspect='ACCOM',
 education='ED0-2',
 gender='F',
 age_group='Y16-19',
 value_2013=country_data['value_2013'].mean(),
 value_2018=country_data['value_2018'].mean(),
 value_2022=country_data['value_2022'].mean() * 1.1
 )
 predictions_2026.append(pred)
 print(f" {country}: {pred:.2f}")
 
plt.figure(figsize=(10, 6))
plt.bar(countries, predictions_2026, color='skyblue')
plt.title('Прогноз удовлетворённости жизнью на 2026 год\n(Сценарий улучшения экономики на 10%)')
plt.ylabel('Удовлетворённость жизнью')
plt.ylim(min(predictions_2026)*0.95, max(predictions_2026)*1.05)
plt.grid(axis='y')
plt.savefig('country_comparison_2026.png')
plt.show()

print("\nСОХРАНЕНИЕ ФИНАЛЬНЫХ РЕЗУЛЬТАТОВ")
test_predictions = model.predict([test_numeric, test_categorical, test_geo])
results_df = X_test.copy()
results_df['actual_satisfaction'] = y_test.values
results_df['predicted_satisfaction'] = test_predictions
results_df.to_csv('model_predictions.csv', index=False)
print("Все этапы завершены.")
