import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import funkcje # Importowanie własnych funkcji z pliku funkcje.py

# 1. Wczytywanie parametrów z pliku JSON
with open("model.json", "r", encoding="utf-8") as read_file:
    config = json.load(read_file)
    alpha = config["model"]["alpha"] # Współczynnik uczenia (learning rate)
    num_iters = config["model"]["num_iters"] # Liczba iteracji algorytmu

# 2. Wczytywanie danych z pliku CSV
data = pd.read_csv("insurance.csv")

# 3. Mapowanie zmiennych kategorycznych na wartości liczbowe
data['sex'] = data['sex'].map({'male': 1, 'female': 0}) # Mężczyzna: 1, Kobieta: 0
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0}) # Palacz: 1, Niepalący: 0

# 4. Wybór cech (X) oraz zmiennej celu (Y)
# Pomijamy kolumnę 'region' i 'charges' w macierzy X
X = data[['age', 'sex', 'bmi', 'children', 'smoker']].copy()
Y = data[['charges']].copy() # Koszt ubezpieczenia, który chcemy przewidzieć

# 5. Podział na dane treningowe (80%) i testowe (20%) 0.2 to 20 proc
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 6. Normalizacja danych (Standardization)
mean_values = X_train.mean() # Obliczanie średniej dla cech
std_values = X_train.std() # Obliczanie odchylenia standardowego

# Wykonujemy normalizację według wzoru: (X - mean) / std
X_train_norm = (X_train - mean_values) / std_values
X_test_norm = (X_test - mean_values) / std_values

# 7. Konwersja na tablice NumPy i zmiana kształtu wektorów Y
X_train_final = X_train_norm.values
X_test_final = X_test_norm.values
Y_train_final = Y_train.values.reshape(-1, 1) # reshape(-1,1) tworzy wektor kolumnowy
Y_test_final = Y_test.values.reshape(-1, 1)

# 8. Dodanie kolumny jedynek (bias/theta_0)
X_train_final = np.c_[np.ones(X_train_final.shape[0]), X_train_final]
X_test_final = np.c_[np.ones(X_test_final.shape[0]), X_test_final]

# 9. Inicjalizacja parametrów theta zerami
theta = np.zeros((X_train_final.shape[1], 1))

# 10. Trenowanie modelu przy użyciu Gradient Descent
theta, J_history = funkcje.gradient_descent(X_train_final, Y_train_final, theta, alpha, num_iters)

# 11. Testowanie modelu i przewidywanie wartości
Y_pred = funkcje.linear_regression(X_test_final, theta)

# 12. Wizualizacja wyników: Rzeczywiste vs Przewidywane
plt.scatter(Y_test_final, Y_pred) # Rysowanie punktów danych
max_val = max(Y_test_final.max(), Y_pred.max())
plt.plot([0, max_val], [0, max_val], color="black") # Linia idealnego dopasowania
plt.xlabel("Wartości rzeczywiste")
plt.ylabel("Wartości przewidywane")
plt.title("Porównanie wyników modelu")
plt.show()

# 13. Obliczanie współczynnika determinacji R2
ss_res = np.sum((Y_test_final - Y_pred) ** 2) # Suma kwadratów reziduów
ss_tot = np.sum((Y_test_final - np.mean(Y_test_final)) ** 2) # Całkowita suma kwadratów
r2 = 1 - (ss_res / ss_tot)

print(f"Końcowe parametry theta: \n{theta}")
print(f"Współczynnik R2: {r2}")