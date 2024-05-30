import numpy as np

import matplotlib.pyplot as plt

dane = np.loadtxt('138305.txt')

x=[] 
y=[] 
z=[]
stala = 0.3
for i in range(dane.shape[0]): 
    if(dane[i][1] == 0.3):
        x.append(dane[i][0]) 
        y.append(dane[i][1]) 
        z.append(dane[i][2])

x = np.array(x) 
y = np.array(y)
z = np.array(z)

def interpolacja_Lag(x, y, f, p):
    Xi = np.linspace(x[0], x[-1], 10001)
    Zi = np.zeros(Xi.shape[0])
    i = np.where(y == p)[0][0]
    Z = np.copy(f)
    print("Współczynniki wielomianów interpolacyjnych Lagrange'a:")
    xp = 0
    xk = 4
    while xp < len(x) - 1:
        xk = min(xk, len(x) - 1)  # Upewnij się, że nie przekraczamy granic
        A = np.zeros(xk - xp + 1)  # Dostosuj rozmiar tablicy A
        for k in range(A.shape[0]):
            mianownik = 1
            for j in range(xp, xk + 1):
                if xp + k != j:
                    mianownik *= x[xp + k] - x[j]
            A[k] = Z[xp + k] / mianownik
        for j in range(Xi.shape[0]):
            if x[xp] <= Xi[j] and Xi[j] <= x[xk]:
                suma = 0
                for k in range(A.shape[0]):
                    iloczyn = 1
                    for l in range(xp, xk + 1):
                        if xp + k != l:
                            iloczyn *= Xi[j] - x[l]
                    suma += A[k] * iloczyn
                Zi[j] = suma
        xp = xk
        xk += 4
        print(A)
    return (Xi, Zi, x, Z)

def metoda_eliminacji_gaussa(M, d):
    n = M.shape[0]
    c = np.zeros((n, n + 1))
    c[:, :n] = M
    c[:, n] = d
    X = np.zeros(n)
    for s in range(0, n - 1):
        for i in range(s + 1, n):
            for j in range(s + 1, n + 1):
                c[i, j] -= c[i, s] / c[s, s] * c[s, j]
    X[n - 1] = c[n - 1, n] / c[n - 1, n - 1]
    for i in range(n - 1, -1, -1):
        suma = 0
        for s in range(i + 1, n):
            suma += c[i, s] * X[s]
        X[i] = (c[i, n] - suma) / c[i, i]
    return X

def interp_splajn(x, f):
    n = len(x)
    M = np.zeros((n + 2, n + 2))
    B = np.zeros(n + 2)
    h = x[1] - x[0]
    M[0][0] = -3 / h
    M[0][2] = 3 / h
    M[n + 1][n + 1] = 3 / h
    M[n + 1][n - 1] = -3 / h
    for i in range(1, n + 1):
        M[i][i] = 4
        M[i][i - 1] = 1
        M[i][i + 1] = 1
    B[n + 1] = -1
    B[0] = 1
    for i in range(1, n + 1):
        B[i] = f[i - 1]
    K = metoda_eliminacji_gaussa(M, B)
    return K

def rysuj_interpolacje(x, y, f, p):
    L = interpolacja_Lag(x, y, f, p)
    S = interp_splajn(x, f)
    plt.subplots()
    plt.plot(L[0], L[1], 'r-', linewidth=2.0, label="funckja interpolacyjna Lagrange'a")
    plt.plot(L[2], L[3], 'bo', label='[x,z]')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Funkcje interpolacyjne dla y = 0.3")
    plt.grid()
    plt.legend()
    plt.show()
#
rysuj_interpolacje(x, y, z, stala)