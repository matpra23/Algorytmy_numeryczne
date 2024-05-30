import numpy as np

dane = np.loadtxt('138305.txt')
stala = 0.2

x = [] 
f = []
for i in range(dane.shape[0]): 
    if dane[i][1] == stala:
        x.append(dane[i][0])
        f.append(dane[i][2])
x = np.array(x) 
f = np.array(f)

def metoda_eliminacji_gaussa(M, d):
    n = M.shape[0]
    c = np.zeros((n, n+1))
    c[:, 0:n] = M  # Poprawiono błąd indeksowania
    c[:, n] = d
    X = np.zeros(n)
    for s in range(0, n-1):
        for i in range(s+1, n):
            for j in range(s+1, n+1):
                c[i, j] -= c[i, s] / c[s, s] * c[s, j]
    X[n-1] = c[n-1, n] / c[n-1, n-1]
    for i in range(n-1, -1, -1):
        suma = 0
        for s in range(i+1, n):
            suma += c[i, s] * X[s]
        X[i] = (c[i, n] - suma) / c[i, i]
    return X

def a_lag(a, xi, x): 
    wynik = 0
    m = 1
    n = len(a) 
    nn = len(xi)
    for i in range(n):
        for j in range(nn): 
            if i != j:
                m *= (x - xi[j])
        wynik += m * a[i]
        m = 1 
    return wynik

def interp_lag(x, y, n): 
    a = np.zeros(n)
    d = 1
    for i in range(n):
        for j in range(n): 
            if i != j:
                d *= (x[i] - x[j]) 
        a[i] = y[i] / d
        d = 1 
    return a

def macierze_lag(): 
    I = 0
    macierz_x = np.zeros(100 * (int(21/3))) 
    macierz_y = np.zeros(100 * (int(21/3))) 
    for i in range(0, 21, 3):
        przedzial_x = x[i:i+4]
        przedzial_y = f[i:i+4]
        n = len(przedzial_x) 
        a = interp_lag(przedzial_x, przedzial_y, n) 
        xx = np.linspace(przedzial_x[0], przedzial_x[-1], 100) 
        yy = np.zeros(100)
        for j in range(len(xx)):
            yy[j] = a_lag(a, przedzial_x, xx[j]) 
        macierz_x[I:I+100] = xx 
        macierz_y[I:I+100] = yy
        I += 100
    return macierz_x, macierz_y

def interp_splajn(x, f): 
    n = x.shape[0]
    M = np.zeros((n+2, n+2)) 
    B = np.zeros(n+2) 
    h = x[1] - x[0] 
    M[0][0] = -3 / h 
    M[0][2] = 3 / h 
    M[n+1][n+1] = 3 / h 
    M[n+1][n-1] = -3 / h
    for i in range(1, n+1):
        M[i][i] = 4 
        M[i][i-1] = 1 
        M[i][i+1] = 1
    B[n+1] = -1
    B[0] = 1
    for i in range(1, n+1):
        B[i] = f[i-1] 
    K = metoda_eliminacji_gaussa(M, B) 
    return K

def funkcja(x, h): 
    if -2 * h <= x <= -h: 
        return (1/h**3) * (x - (-2 * h))**3 
    if -h <= x <= 0: 
        return (1/h**3) * (h**3 + 3 * h**2 * (x + h) + 3 * h * (x + h)**2 - 3 * (x + h)**3) 
    if 0 <= x <= h: 
        return (1/h**3) * (h**3 + 3 * h**2 * (h - x) + 3 * h * (h - x)**2 - 3 * (h - x)**3) 
    if h <= x <= 2 * h: 
        return (1/h**3) * (2 * h - x)**3 
    else: 
        return 0

def interp_bsplajn(macierz_x, x, K): 
    n = len(macierz_x) 
    h = macierz_x[1] - macierz_x[0] 
    B = np.zeros(n+2)
    suma = 0 
    B[0] = macierz_x[0] - h 
    B[n+1] = macierz_x[n-1] + h 
    for i in range(1, n+1):
        B[i] = macierz_x[i-1] 
    for i in range(n+2):
        suma += K[i] * funkcja(x - B[i], h) 
    return suma
    
def macierze_bsplajn(): 
    n = len(x)
    K = interp_splajn(x, f) 
    macierz_y = np.zeros(100 * (int(21/3))) 
    macierz_x = np.linspace(x[0], x[n-1], 100 * (int(21/3))) 
    for i in range(100 * (int(21/3))):
        macierz_y[i] = interp_bsplajn(x, macierz_x[i], K) 
    return macierz_x, macierz_y

def aproksymacja_f1zm(x, f):
    w = np.zeros((2, 2))
    b = np.zeros((2, 1))
    suma = 0
    
    for i in range(len(x)):
        suma += x[i]
    w[1][0] = suma
    w[0][1] = suma
    suma = 0
    for i in range(len(x)):
        suma += x[i]**2
    w[1][1] = suma
    w[0][0] = len(x)
    suma = 0
    for j in range(len(x)):
        suma += f[j]
    b[0][0] = suma
    suma = 0
    for i in range(len(x)):
        suma += x[i] * f[i]
    b[1][0] = suma
    a = np.linalg.solve(w, b) 
    return a

def aproksymacja_kwadratowa(): 
    n = x.shape[0] 
    M = np.zeros((3, 3)) 
    P = np.zeros(3)
    for i in range(n):
        M[0][1] += x[i] 
        M[0][2] += x[i]**2.0
        M[1][0] += x[i] 
        M[1][1] += x[i]**2.0 
        M[1][2] += x[i]**3.0
        M[2][0] += x[i]**2.0 
        M[2][1] += x[i]**3.0 
        M[2][2] += x[i]**4.0
        P[0] += f[i] 
        P[1] += x[i] * f[i] 
        P[2] += x[i]**2.0 * f[i]
    a = np.linalg.solve(M, P) 
    return a

liniowaWspolczynniki = aproksymacja_f1zm(x, f) 

def true_function(x):
    return liniowaWspolczynniki[0] + liniowaWspolczynniki[1]*x

kwadratowaWspolczynniki = aproksymacja_kwadratowa() 

def true_function_kwadrat(x):
    return kwadratowaWspolczynniki[0] + kwadratowaWspolczynniki[1]*x + kwadratowaWspolczynniki[2]*x*x

def macierze_aproksymacjaLiniowa(): 
    macierz_y = np.zeros(100 * (int(21/3))) 
    macierz_x = np.linspace(x[0], x[x.shape[0]-1], 100 * (int(21/3))) 
    for i in range(100 * (int(21/3))):
        macierz_y[i] = true_function(macierz_x[i]) 
    return macierz_x, macierz_y

def macierze_aproksymacjaKwadratowa(): 
    macierz_y = np.zeros(100 * (int(21/3))) 
    macierz_x = np.linspace(x[0], x[x.shape[0]-1], 100 * (int(21/3))) 
    for i in range(100 * (int(21/3))):
        macierz_y[i] = true_function_kwadrat(macierz_x[i]) 
    return macierz_x, macierz_y

lagrange = macierze_lag()
bsplajn = macierze_bsplajn()
aproksymacjaLiniowa = macierze_aproksymacjaLiniowa() 
aproksymacjaKwadratowa = macierze_aproksymacjaKwadratowa()

def calkPr(x, y): 
    n = len(x)
    calka = 0
    for i in range(n-1):
        h = x[i+1] - x[i]
        calka += h * y[i] 
    return calka

calkaAproksymacjaLiniowa = calkPr(aproksymacjaLiniowa[0], aproksymacjaLiniowa[1]) 
print("Wynik całki metodą prostokątów z aproksymacji liniowej: ", calkaAproksymacjaLiniowa) 
calkaAproksymacjaKwadratowa = calkPr(aproksymacjaKwadratowa[0], aproksymacjaKwadratowa[1])
print("Wynik całki metodą prostokątów z aproksymacji kwadratowej: ", calkaAproksymacjaKwadratowa)
calkaLag = calkPr(lagrange[0], lagrange[1])
print("Wynik całki metodą prostokątów z interpolacji Lagrange'a: ", calkaLag) 
calkaSplajn = calkPr(bsplajn[0], bsplajn[1])
print("Wynik całki metodą prostokątów z interpolacji B-splajn: ", calkaSplajn) 
#