import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# Функция для загрузки данных батчами
def load_data_in_batches(folder_path, batch_size):
    batches = []
    file_names = sorted(os.listdir(folder_path))  # Получаем имена файлов в папке и сортируем их
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        for chunk in pd.read_csv(file_path, chunksize=batch_size):
            batches.append(chunk)
    return batches

# Путь к вашему CSV файлу (датасету)
file_path = "../../data/batches/"

# Размер батча
batch_size = 10000

# Загрузка данных батчами
batches = load_data_in_batches(file_path, batch_size)

# Выбор признаков для обучения
selected_features = ['V1', 'V4', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V20', 'V21', 'V26']

# Объединение всех батчей в один DataFrame
data = pd.concat(batches, ignore_index=True)

# Разделение на признаки и целевую переменную
X = data[selected_features]
y = data['Class']

# Применение синтетического создания данных с помощью SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Применение стандартизации к признакам
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Инициализация модели Gradient Boosting Machines
clf = GradientBoostingClassifier(random_state=42)

# Обучение модели с прогресс-баром
with tqdm(desc="Обучение модели", unit="батч", total=len(batches)) as progress_bar:
    for batch in batches:
        X_batch = batch[selected_features]
        y_batch = batch['Class']

        # Применение стандартизации к батчу
        X_batch_scaled = scaler.transform(X_batch)

        # Обучение модели на текущем батче
        clf.fit(X_batch_scaled, y_batch)

        # Обновление индикатора прогресса
        progress_bar.update(1)

# Сохранение обученной модели
model_file = "../../data/model/trained_model.pkl"
joblib.dump(clf, model_file)

# Оценка модели
y_pred = clf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
