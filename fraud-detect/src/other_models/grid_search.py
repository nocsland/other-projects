import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from random_forest_model import X_train, y_train

# Установка seed для генератора случайных чисел в numpy
# np.random.seed(42)

# Создание экземпляра классификатора
rf_classifier = RandomForestClassifier()

# Сетка параметров для перебора
param_grid = {
    'n_estimators': [50, 55, 60, 65, 70, 75, 85]  # Перебираем разное количество деревьев
}

# Создание экземпляра GridSearchCV
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=8)

# Запуск поиска
grid_search.fit(X_train, y_train)

# Вывод лучших параметров и оценки
print("Лучшие параметры:", grid_search.best_params_)
print("Лучшая оценка точности:", grid_search.best_score_)
