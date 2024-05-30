import numpy as np

import matplotlib.pyplot as plt

dane = np.loadtxt('138305.txt')
x = dane[:, 0]
y = 0.3
f = dane[:, 2]

def interpolacja_gaussa(P, R):
    n = P.shape[0]
    M = np.zeros([n, n+1])
    M[:, :n] = P
    M[:, n] = R
    x = np.zeros(n)

    for s in range(n-1):
        for i in range(s+1, n):
            for j in range(s+1, n+1):
                M[i, j] -= (M[i, s] / M[s, s]) * M[s, j]
    x[n-1] = M[n-1, n] / M[n-1, n-1]

    for i in range(n-2, -1, -1):
        sum = 0
        for s in range(i+1, n):
            sum += M[i, s] * x[s]
        x[i] = (M[i, n] - sum) / M[i, i]
    return x

def interpolacja_wielomianowa(P, R):
    n = len(P)
    A = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            A[i, j] = P[i] ** j
    return interpolacja_gaussa(A, R)

def wartosc_y(P, R):
    n = len(R)
    n1 = len(P)
    y = np.zeros(n)
    for i in range(n):
        for j in range(n1):
            y[i] += (R[i] ** j) * P[j]
    return y

def rysunek():
    xs = np.split(x, 6)
    fs = np.split(f, 6)

    w1 = interpolacja_wielomianowa(xs[0][8:23], fs[0][8:23])
    w2 = interpolacja_wielomianowa(xs[0][18:27], fs[0][18:27])
    w3 = interpolacja_wielomianowa(xs[0][28:37], fs[0][28:37])
    w4 = interpolacja_wielomianowa(xs[0][:12], fs[0][:12])
    w5 = interpolacja_wielomianowa(xs[0][33:], fs[0][33:])
    w6 = interpolacja_wielomianowa(xs[0][23:32], fs[0][23:32])

    x1 = np.linspace(1, 2, 1000)
    x2 = np.linspace(2, 2.5, 200)
    x3 = np.linspace(3, 3.5, 300)
    x4 = np.linspace(0, 1, 1000)
    x5 = np.linspace(3.5, 4, 300)
    x6 = np.linspace(2.5, 3, 200)

    f_interp_wiel1 = wartosc_y(w1, x1)
    f_interp_wiel2 = wartosc_y(w2, x2)
    f_interp_wiel3 = wartosc_y(w3, x3)
    f_interp_wiel4 = wartosc_y(w4, x4)
    f_interp_wiel5 = wartosc_y(w5, x5)
    f_interp_wiel6 = wartosc_y(w6, x6)

    plt.plot(xs[0], fs[0], 'o', color='b', label="punkty oryginalne")

    plt.plot(x1, f_interp_wiel1, color='r', label="interpolacja wielomianowa")
    plt.plot(x2, f_interp_wiel2, color='r')
    plt.plot(x3, f_interp_wiel3, color='r')
    plt.plot(x4, f_interp_wiel4, color='r')
    plt.plot(x5, f_interp_wiel5, color='r')
    plt.plot(x6, f_interp_wiel6, color='r')

    plt.title("Funkcja interpolacyjna wielomianowa")
    plt.xlabel('X')
    plt.ylabel('F(x, y = 0.0)')
    plt.legend()
    plt.grid(True)
    plt.show()

rysunek()
