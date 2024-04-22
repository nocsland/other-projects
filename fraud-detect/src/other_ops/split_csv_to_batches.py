import pandas as pd


def split_csv_to_batches(csv_file, batch_size):
    # Загрузка данных из CSV файла
    data = pd.read_csv(csv_file)

    # Разбиение данных на батчи
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]
        batches.append(batch)

    return batches


# Путь к вашему CSV файлу
csv_file = "../../data/dataset/creditcard.csv"

# Размер батча
batch_size = 10000

# Разбиение CSV на батчи
batches = split_csv_to_batches(csv_file, batch_size)

# Сохранение батчей в отдельные файлы
for i, batch in enumerate(batches):
    batch.to_csv(f"../../data/batches/batch_{i + 1}.csv", index=False)
    print(f"Сохранен batch_{i + 1}.csv")