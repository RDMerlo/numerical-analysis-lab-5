import math
from scipy import integrate
from slae_gauss import *
from numpay_print import *

N = 3
a = 0.6 - 3 / N
b = 2 - N / 13
dl = (b - a) / 5
mu1 = 15 / (N + 3)
mu2 = -6 * N / 21
VarInt = 4
n = 4
eps = 0.000001


def kx(x):
    return (4 - 0.1 * x) / (x ** 2 + N / 16)


def qx(x):
    return (x + 5) / (x ** 2 + 0.9 * N)


def fx(x):
    return (N + x) / 3.5


def phi0(x):
    return mu1 + (mu2 - mu1) * math.sin(math.pi * (x - a) / (2 * (b - a)))


def derPhi0(x):
    return (mu2 - mu1) * math.pi * 2 * (b - a) * math.cos((math.pi * (x - a) / (2 * (b - a))))


def phiK(x, k):
    return math.sin(math.pi * k * (x - a) / (b - a))


def derPhiK(x, k):
    return k * math.pi * (b - a) * math.cos(k * math.pi * (x - a) / (b - a))


def thetaA(x, i, j):
    return kx(x) * derPhiK(x, i) * derPhiK(x, j) + qx(x) * phiK(x, i) * phiK(x, j)


# посчитать с помощью интеграла встроенного
def thetaB(x, i, j=0):
    return fx(x) * phiK(x, i) - qx(x) * phi0(x) * phiK(x, i) - kx(x) * derPhiK(x, i) * derPhi0(x)


def GaussIntegral(i, j, f, a0, b0):
    # поделить на части и для каждой пересчитать
    c = [0.34785484, 0.65214516, 0.65214516, 0.34785484]
    t = [-0.86113631, -0.33998104, 0.33998104, 0.86113631]
    points = [(b0 + a0) / 2 + (b0 - a0) / 2 * t[i] for i in range(VarInt)]
    return (b0 - a0) / 2 * (
                c[0] * f(points[0], i, j) + c[1] * f(points[1], i, j) + c[2] * f(points[2], i, j) + c[3] * f(points[3],
                                                                                                             i, j))

A = [[0] * VarInt for _ in range(VarInt)]
ATrue = [[0] * VarInt for _ in range(VarInt)]

print(ATrue)
# for i in range(VarInt):
#     for j in range(VarInt):
#         ATrue[i][j] = integrate.quad(thetaA, a, b, args=(i+1, j+1))[0]
# print(ATrue)
# exit()
B = [0] * VarInt
for i in range(VarInt):
    for j in range(VarInt):
        a0, b0 = a, b
        resultPrev = GaussIntegral(i + 1, j + 1, thetaA, a0, b0)
        resultLast = GaussIntegral(i + 1, j + 1, thetaA, a0, (b0 - a0) / 2) + GaussIntegral(i + 1, j + 1, thetaA,
                                                                                            (b0 - a0) / 2, b0)
        delta = (b0 - a0) / 3
        count = 3
        while abs(resultLast - resultPrev) > eps:
            resultPrev = resultLast
            resultLast = 0
            for l in range(count):
                resultLast += GaussIntegral(i + 1, j + 1, thetaA, a0 + l * delta, a0 + (l + 1) * delta)
            count += 1
            delta = (b0 - a0) / count

        A[i][j] = resultLast
        # ATrue[i][j] = integrate.quad(thetaA, a, b, args=(i, j))
        ATrue[i][j] = integrate.quad(thetaA, a, b, args=(i + 1, j + 1))[0]

    a0, b0 = a, b
    resultPrev = GaussIntegral(i + 1, 0, thetaB, a0, b0)
    resultLast = GaussIntegral(i + 1, 0, thetaB, a0, (b0 - a0) / 2) + GaussIntegral(i + 1, 0, thetaB, (b0 - a0) / 2, b0)
    delta = (b0 - a0) / 3
    count = 3
    while abs(resultLast - resultPrev) > eps:
        resultPrev = resultLast
        resultLast = 0
        for l in range(count):
            resultLast += GaussIntegral(i + 1, 0, thetaB, a0 + l * delta, a0 + (l + 1) * delta)
        count += 1
        delta = (b0 - a0) / count
    B[i] = resultLast

A = copy.copy(A)
B = np.array(B)
C = np.array([None, None, None, None])

print_array(A)
print("----")
print_array(ATrue)
print("----")
exit()

print_vector(B)

# решаем СЛАУ
A = np.column_stack((A, B))
A = decision_sle_direct_move(A)
# print("Полученная трапецевидная (расширенная) матрица A:")
# print_array(A)

B = A[:, 4]
A = A[:, :4]
# обратное ход
C = decision_sle_reverse_move(A, C, B)
print("\nПолученное решение:", end="")
print("\n\nX = ", C, "\n\n")

# выводим значения в точках
durX = a
for i in range(6):
    unx = phi0(durX)
    for j in range(VarInt):
        unx += C[j] * phiK(durX, j + 1)
    print(f"u{i + 1}(x) = {unx}")
    durX += dl
