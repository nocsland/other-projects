import pandas as pd

def add_batches_to_dataset(dataset, batches):
    for batch in batches:
        dataset = pd.concat([dataset, batch], ignore_index=True)
    return dataset

# Путь к вашему исходному CSV файлу (датасету)
original_dataset_file = "original_dataset.csv"

# Загрузка исходного датасета
dataset = pd.read_csv(original_dataset_file)

# Пути к файлам с батчами
batch_files = ["batch_1.csv", "batch_2.csv", "batch_3.csv"]  # и так далее

# Загрузка батчей и добавление их к датасету
for batch_file in batch_files:
    batch = pd.read_csv(batch_file)
    dataset = add_batches_to_dataset(dataset, [batch])
