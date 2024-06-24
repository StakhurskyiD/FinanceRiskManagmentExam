import pandas as pd

# Завантажуємо результати з файлу
vawi_output_file_path = 'output/vawi_k_ratio_output.csv'
vawi_df = pd.read_csv(vawi_output_file_path)

# Знаходимо портфель з найвищим K-коефіцієнтом
highest_k_ratio_portfolio = vawi_df.loc[vawi_df['K-ratio'].idxmax()]

# Виводимо результати
print("Портфель з найвищим K-коефіцієнтом:")
print(f"Кластер: {highest_k_ratio_portfolio['Cluster']}")
print(f"K-коефіцієнт: {highest_k_ratio_portfolio['K-ratio']}")