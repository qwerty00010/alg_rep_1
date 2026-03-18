def koszt_lolejnych_iteracij():
    return 0

def r2_f():
    ss_res = np.sum((Y_test_final - Y_pred) ** 2) # Suma kwadratów reziduów
    ss_tot = np.sum((Y_test_final -t_final)) ** 2) # Całkowita suma kwadratów
    r2 = 1 - (ss_res / ss_tot)