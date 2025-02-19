import numpy as np
import matplotlib.pyplot as plt

# Функция для считывания данных из текстового файла
def read_data(file_name):
    x = []
    y = []
    with open(file_name, 'r') as file:
        next(file)  # Пропускаем заголовок
        for line in file:
            values = line.strip().split(',')
            x.append(float(values[0]))
            y.append(float(values[1]))
    return np.array(x), np.array(y)

# Функция для выполнения линейной регрессии
def linear_regression(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Вычисляем коэффициенты
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    return slope, intercept

# Функция для предсказания значений
def predict(x, slope, intercept):
    return slope * x + intercept

# Основная программа
if __name__ == "__main__":
    # Считываем данные
    x, y = read_data('data.txt')

    # Выполняем линейную регрессию
    slope, intercept = linear_regression(x, y)

    print(f'Коэффициент наклона: {slope}')
    print(f'Свободный член: {intercept}')

    # Предсказание значений
    y_pred = predict(x, slope, intercept)

    # Визуализация результатов
    plt.scatter(x, y, color='blue', label='Исходные данные')
    plt.plot(x, y_pred, color='red', label='Линейная регрессия')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Линейная регрессия')
    plt.legend()
    plt.show()
