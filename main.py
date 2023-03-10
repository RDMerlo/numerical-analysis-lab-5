import math
from scipy import integrate
from slae_gauss import *
from numpy_print import *

N = 14
n = 4

a = 0.6 - N / 13
b = 2 - N / 13
dl = (b - a) / 5
mu1 = 15 / (N + 3)
mu2 = -6 * N / 21

eps = 0.000001

def kx(x):
    return (4 - 0.1 * x) / (x ** 2 + N / 16)

def qx(x):
    return (x + 5) / (x ** 2 + 0.9 * N)

def fx(x):
    return (N + x) / 3.5

# другой вариант phi
# def phi0(x):
#     return mu1 + (mu2 - mu1) * math.cos(math.pi/2 + math.pi * (x - a) / (2 * (b - a)))
# def derPhi0(x):
#     return (mu2-mu1) * math.pi / (2 * (b - a)) * math.sin(math.pi/2 + (math.pi * (x - a) / (2 * (b - a))))
# def phiK(x, k):
#     return math.cos(math.pi/2 + math.pi * k * (x - a) / (b - a))
# def derPhiK(x, k):
#     return k * math.pi / (b - a) * math.sin(math.pi/2 + k * math.pi * (x - a) / (b - a))

def phi0(x):
    return mu1 + (mu2 - mu1) * math.cos(math.pi/2 + math.pi * (x - a) / (2 * (b - a)))
def derPhi0(x):
    return (mu2-mu1) * math.pi / (2 * (b - a)) * math.sin((math.pi * (x - a) / (2 * (b - a))))
def phiK(x, k):
    return math.cos(math.pi/2 + math.pi * k * (x - a) / (b - a))
def derPhiK(x, k):
    return k * math.pi / (b - a) * math.sin(k * math.pi * (x - a) / (b - a))

############################################################################

def thetaA(x, i, j):
    "Элемент матрицы А"
    return kx(x) * derPhiK(x, i) * derPhiK(x, j) + qx(x) * phiK(x, i) * phiK(x, j)

def thetaB(x, i, j=0):
    "Элемент вектора B"
    return fx(x) * phiK(x, i) - qx(x) * phi0(x) * phiK(x, i) - kx(x) * derPhiK(x, i) * derPhi0(x)


def compulation_Aij(i = 1, j = 1):
    "Вычисление элементов матрицы А"
    a_temp, b_temp = a, b
    resultPrev = GaussIntegral(i + 1, j + 1, thetaA, a_temp, b_temp) #основное вычисление
    #делим на два отрезка и суммируем
    resultLast = GaussIntegral(i + 1, j + 1, thetaA, a_temp, (b_temp - a_temp) / 2) + GaussIntegral(i + 1, j + 1, thetaA, (b_temp - a_temp) / 2, b_temp)

    count = 3 #три деления
    while abs(resultLast - resultPrev) > eps:
        delta = (b_temp - a_temp) / count #шаг по отрезкам
        resultPrev = resultLast
        resultLast = 0
        for l in range(count):
            resultLast += GaussIntegral(i + 1, j + 1, thetaA, a_temp + l * delta, a_temp + (l + 1) * delta)
        count += 1
    return resultLast

def compulation_Bj(i = 1):
    "Вычисление элементов вектора В"
    a_temp, b_temp = a, b
    resultPrev = GaussIntegral(i + 1, 0, thetaB, a_temp, b_temp) #основное вычисление
    # делим на два отрезка и суммируем
    resultLast = GaussIntegral(i + 1, 0, thetaB, a_temp, (b_temp - a_temp) / 2) + GaussIntegral(i + 1, 0, thetaB, (b_temp - a_temp) / 2, b_temp)

    count = 3 #три деления
    while abs(resultLast - resultPrev) > eps:
        delta = (b_temp - a_temp) / count #шаг по отрезкам
        resultPrev = resultLast
        resultLast = 0
        for l in range(count):
            resultLast += GaussIntegral(i + 1, 0, thetaB, a_temp + l * delta, a_temp + (l + 1) * delta)
        count += 1
    return resultLast

def GaussIntegral(i, j, f, a0, b0):
    "Интегрирование методом Гаусса"
    "заменить своими значениями относительно n=4 или n=3"
    c = [0.34785484, 0.65214516, 0.65214516, 0.34785484] # из книжки
    t = [-0.86113631, -0.33998104, 0.33998104, 0.86113631] # из книжки
    x_points = [(b0 + a0) / 2 + (b0 - a0) / 2 * t[k] for k in range(n)]
    return (b0 - a0) / 2 * sum([c[k] * f(x_points[k], i, j) for k in range(n)])

def ChebyshevIntegral(i, j, f, a0, b0):
    "Интегрирование методом Чебышева"
    "заменить своими значениями относительно n=4 или n=3"
    t = [-0.577350, 0.577350] # из книжки
    x_points = [(b0 + a0) / 2 + (b0 - a0) / 2 * t[k] for k in range(n)]
    return (b0 - a0) / n * sum([f(x_points[k], i, j) for k in range(n)])

A = [[0] * n for _ in range(n)]
ATrue = [[0] * n for _ in range(n)]

B = [0] * n
BTrue = [0] * n

"Собираем матрицу А и В"
for i in range(n):
    for j in range(n):
        A[i][j] = compulation_Aij(i, j)
        ATrue[i][j] = integrate.quad(thetaA, a, b, args=(i + 1, j + 1))[0]
    B[i] = compulation_Bj(i)
    BTrue[i] = integrate.quad(thetaB, a, b, args=(i + 1,))[0]

chislo_obuslovnosti = np.linalg.cond(A)
print("Число обусловности: ", chislo_obuslovnosti, end = "\n\n")

print_array(A, "Метод Гауса\nA:")
print("---" * 20)
print_array(ATrue, "Библиотека Scipy\nA:")
print("---" * 20)
print_vector(B, "Метод Гауса\nB:")
print("---" * 20)
print_vector(BTrue, "Библиотека Scipy\nB:")

# решаем СЛАУ
A = np.column_stack((A, B)) #расширенная матрица
# прямой ход
A = decision_sle_direct_move(A, n)
# обратный ход
C = decision_sle_reverse_move(A, n)

print_vector(C, "C: ")

# выводим значения в точках
durX = a
for i in range(6):
    unx = phi0(durX)
    for j in range(n):
        unx += C[j] * phiK(durX, j + 1)
    print(f"u{i + 1}(x) = {unx:.4f}")
    durX += dl
