import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Leer el archivo CSV original
df = pd.read_csv('output/Prediction.csv')

# Calcular la media del RatingPrediccion por apartment_id
mean_ratings = df.groupby('apartment_id')['RatingPrediccion'].mean().reset_index()

# Escalar la columna de medias al rango [0, 1]
scaler = MinMaxScaler()
mean_ratings['RatingPrediccion_scaled'] = scaler.fit_transform(mean_ratings[['RatingPrediccion']])

# Guardar el nuevo CSV
mean_ratings[['apartment_id', 'RatingPrediccion_scaled']].to_csv('RatingsEscalados.csv', index=False)

print("CSV generado como 'RatingsEscalados.csv'")
