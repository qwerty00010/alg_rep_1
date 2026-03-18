import numpy as np

def linear_regression(X, theta):
    # h = X * theta: Obliczanie przewidywań modelu (hipoteza)
    return np.dot(X, theta)


def cost_function(X, Y, theta):
    m = X.shape[0]  # Liczba przykładów w zbiorze danych
    h = linear_regression(X, theta)
    # Obliczanie błędu średniokwadratowego (MSE) według wzoru J(theta)
    cost = (1 / (2 * m)) * np.sum((h - Y) ** 2)
    return cost


def gradient_descent(X, Y, theta, alpha, num_iters):
    m = X.shape[0]  # Liczba przykładów
    J_history = []  # Tablica do przechowywania wartości funkcji kosztu
    X_T = np.transpose(X)  # Transpozycja macierzy X potrzebna do wzoru na gradient

    for i in range(num_iters):
        h = linear_regression(X, theta)
        # Aktualizacja parametrów theta zgodnie ze wzorem: theta := theta - alpha * gradient
        gradient = (1 / m) * np.dot(X_T, (h - Y))
        theta = theta - alpha * gradient

        # Zapisujemy koszt w każdej iteracji, aby sprawdzić czy algorytm działa poprawnie
        J_history.append(cost_function(X, Y, theta))

    return theta, J_history

def predict_Yy(X_test, theta, y_pred):
    y_pred = np.dot(X_test, theta)