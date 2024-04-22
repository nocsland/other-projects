import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from random_forest_model import X_train, y_train

# Создание экземпляра классификатора
rf_classifier = RandomForestClassifier()

# Сетка параметров для перебора
param_grid = {
    'n_estimators': [60,70,80]  # Перебираем разное количество деревьев , 55, 60, 65, 70, 75, 85, 90, 95, 100
}

# Создание экземпляра GridSearchCV
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=8)

# Запуск поиска
grid_search.fit(X_train, y_train)

# Получение результатов кросс-валидации
cv_results = grid_search.cv_results_

# Извлечение значений оценки точности и количества деревьев
accuracy = cv_results['mean_test_score']
n_estimators = param_grid['n_estimators']

# Построение кривой валидации
plt.figure()
plt.plot(n_estimators, accuracy)
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Validation Curve')
plt.show()
