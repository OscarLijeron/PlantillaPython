import pandas as pd
import json
from multiprocessing import Pool, cpu_count
import csv
import ast

def safe_json_loads(raw):
    if pd.isna(raw):
        return None
    try:
        return ast.literal_eval(raw)
    except Exception as e:
        print(f"Error evaluando JSON con ast: {e} → {raw}")
        return None

def procesar_fila(row):
    reviews = row['reviews']
    
    if not isinstance(reviews, list):
        return []

    resultado = []
    for review in reviews:
        apt_id = review.get('listing_id')
        comentario = review.get('comments', '')

        if apt_id and comentario:
            resultado.append({
                'apartment_id': apt_id,
                'review': comentario
            })
        else:
            print(f"⚠️ Reseña sin 'listing_id' o 'comments': {review}")

    return resultado

if __name__ == '__main__':
    df = pd.read_csv('Turkey_translated.csv')

    # Parsear JSON de forma segura
    df['reviews'] = df['reviews'].apply(safe_json_loads)

    # Filtrar registros válidos
    df_validos = df[df['reviews'].notnull()].copy()
    print(f"Total filas con JSON válido: {len(df_validos)}")

    # Guardar archivo intermedio
    df_validos.to_csv('archivo_limpio.csv', index=False)

    # Mostrar algunos ejemplos
    print("\nEjemplo de datos en la columna 'reviews':")
    for i, rev in enumerate(df_validos['reviews'].head(5)):
        print(f"Fila {i}:")
        print(rev)
        print("---")

    # Multiproceso
    rows = df_validos.to_dict(orient='records')
    with Pool(processes=min(6, cpu_count())) as pool:
        resultados = pool.map(procesar_fila, rows)

    all_reviews = [item for sublist in resultados for item in sublist]
    print(f"Total de reseñas extraídas: {len(all_reviews)}")

    if all_reviews:
        pd.DataFrame(all_reviews).to_csv('reviews_expandidas.csv', index=False, quoting=csv.QUOTE_ALL)
        print("✅ ¡Archivo reviews_expandidas.csv generado correctamente!")
    else:
        print("⚠️ No se extrajeron reseñas válidas. Revisa el archivo de entrada.")



