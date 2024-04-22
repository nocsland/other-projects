import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Загрузите свой датасет
data = pd.read_csv("../../data/dataset/source_creditcard.csv")

# Подготовьте данные
X = data.drop("Class", axis=1)
y = data["Class"]

# Разделите данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создайте модель классификатора (может быть любой моделью)
clf = RandomForestClassifier(n_estimators=25, class_weight='balanced', n_jobs=8)

# Создайте объект RFE
rfe = RFE(estimator=clf, step=1, n_features_to_select=5)

# Обучите RFE на обучающем наборе данных и преобразуйте данные
X_train_rfe = rfe.fit_transform(X_train, y_train)

# Оцените производительность модели на тестовом наборе данных с разным количеством признаков
scores = []
for n_features in tqdm(range(1, X.shape[1] + 1), desc="Выбор признаков"):
    rfe.n_features_to_select = n_features
    X_test_rfe = rfe.transform(X_test)
    clf.fit(X_train_rfe, y_train)
    y_pred = clf.predict(X_test_rfe)
    accuracy = accuracy_score(y_test, y_pred)
    scores.append((n_features, accuracy))

# Найдите оптимальное количество признаков
best_n_features, best_accuracy = max(scores, key=lambda x: x[1])

# Получите список выбранных признаков
selected_features = np.array(X.columns)[rfe.support_]

print("Оптимальное количество признаков:", best_n_features)
print("Точность модели с оптимальным количеством признаков:", best_accuracy)
print("Список выбранных признаков:", selected_features)

