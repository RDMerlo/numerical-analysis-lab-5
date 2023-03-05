import copy
import numpy as np

# смена строк
def swap_row(A, i, j):
    temp = copy.copy(A[i])
    A[i] = A[j]
    A[j] = temp


# прямой ход решения
def decision_sle_direct_move(A, n):
    for k in range(0, n, 1):
        # вернём индекс максимального элемента по модулю из столбца
        index_max = np.argmax(abs(A[k:, k])) + k
        # меняем строки местами
        swap_row(A, k, index_max)

        A[k] /= A[k][k]
        for i in range(k + 1, n, 1):
            A[i] = A[i] - A[k] * A[i][k]
    return A


# обратный ход, нахождение X
def decision_sle_reverse_move(A, n):
    X = [None] * n
    F = A[:, n]
    A = A[:, :n]
    X[n - 1] = F[n - 1]
    for k in reversed(range(0, n - 1, 1)):
        sum_row = 0
        for i in reversed(range(0, n, 1)):
            if (A[k][i] == 0.0):
                continue
            if (X[i] is not None):
                sum_row += A[k][i] * X[i]
        X[k] = F[k] - sum_row
    return X