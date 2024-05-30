import numpy as np

import matplotlib.pyplot as plt

dane = np.loadtxt('138305.txt')
stala = 0.2
x = dane[dane[:, 1] == stala, 0]
f = dane[dane[:, 1] == stala, 2]

def interpolacja_gaussa(P, R):
    n = P.shape[0]
    M = np.zeros([n, n+1])
    M[:, :n] = P
    M[:, n] = R
    x = np.zeros(n)

    for s in range(n-1):
        for i in range(s+1, n):
            for j in range(s+1, n+1):
                M[i, j] = M[i, j] - ((M[i, s] / M[s, s]) * M[s, j])
    x[n-1] = M[n-1, n] / M[n-1, n-1]

    for i in range(n-2, -1, -1):
        sum = 0
        for s in range(i+1, n):
            sum += M[i, s] * x[s]
        x[i] = (M[i, n] - sum) / M[i, i]
    return x

def aproksymacja_liniowa():
    n = x.shape[0]
    M = np.zeros((2, 2))
    P = np.zeros(2)
    M[0, 0] = n
    for i in range(n):
        M[0, 1] += x[i]
        M[1, 0] += x[i]
        M[1, 1] += (x[i] ** 2.0)
        P[0] += f[i]
        P[1] += x[i] * f[i]
    a = interpolacja_gaussa(M, P)
    return a

def aproksymacja_kwadratowa():
    n = x.shape[0]
    M = np.zeros((3, 3))
    P = np.zeros(3)
    M[0, 0] = n
    for i in range(n):
        M[0, 1] += x[i]
        M[0, 2] += (x[i] ** 2.0)
        M[1, 0] += x[i]
        M[1, 1] += (x[i] ** 2.0)
        M[1, 2] += (x[i] ** 3.0)
        M[2, 0] += (x[i] ** 2.0)
        M[2, 1] += (x[i] ** 3.0)
        M[2, 2] += (x[i] ** 4.0)
        P[0] += f[i]
        P[1] += x[i] * f[i]
        P[2] += (x[i] ** 2.0) * f[i]
    a = interpolacja_gaussa(M, P)
    return a

liniowaWspolczynniki = aproksymacja_liniowa()
kwadratowaWspolczynniki = aproksymacja_kwadratowa()

def true_function(x):
    return liniowaWspolczynniki[0] + liniowaWspolczynniki[1] * x

def true_function_kwadrat(x):
    return kwadratowaWspolczynniki[0] + kwadratowaWspolczynniki[1] * x + kwadratowaWspolczynniki[2] * x * x


def macierze_aproksymacjaLiniowa():
    macierz_y = np.zeros(100 * (int(21 / 3)))
    macierz_x = np.linspace(x[0], x[x.shape[0] - 1], 100 * (int(21 / 3)))
    for i in range(100 * (int(21 / 3))):
        macierz_y[i] = true_function(macierz_x[i])
    return macierz_x, macierz_y


def macierze_aproksymacjaKwadratowa():
    macierz_y = np.zeros(100 * (int(21 / 3)))
    macierz_x = np.linspace(x[0], x[x.shape[0] - 1], 100 * (int(21 / 3)))
    for i in range(100 * (int(21 / 3))):
        macierz_y[i] = true_function_kwadrat(macierz_x[i])
    return macierz_x, macierz_y


def rysunek():
    liniowa = macierze_aproksymacjaLiniowa()
    kwadratowa = macierze_aproksymacjaKwadratowa()

    plt.scatter(x, f, label="Punkty", color='darkviolet')
    plt.plot(liniowa[0], liniowa[1], label="Aproksymacja liniowa", color='red')
    plt.plot(kwadratowa[0], kwadratowa[1], label="Aproksymacja kwadratowa", color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Wykres aproksymacji dla stałej y = 0.2")
    plt.legend()
    plt.show()

rysunek()
