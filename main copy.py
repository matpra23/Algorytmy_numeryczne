import matplotlib.pyplot as plt
import numpy as np

# #wizualizacja danych w układzie x, F(x,y) dla każdej linii y=const

# Definicja funkcji F(x, y)
dane = np.loadtxt('138305.txt')
x = dane[:,0]
y = dane[:,1]
f = dane[:,2]
def z1_wizualizacja():
    xn = np.split(x,6)
    yn = np.split(y,6)
    fn = np.split(f,6)

    for i in range(6):
        plt.plot(xn[i], fn[i], '-o', label=f"y = {yn[i][0]}")
    plt.xlabel('x')
    plt.ylabel('F(x,y)')
    plt.title('Wizualizacja danych F(x, y) dla każdej linii y=const')
    plt.legend()
    plt.grid(True)
    plt.show()

#1_wizualizacja()

#- wyznaczyć średnią, medianę, odchylenie standardowe z podziałem na współrzędne y, prezentacja na wykresie słupkowym (obowiązkowo)z    

def z2_srednia_mediana_odchylenie():
    
    xn = np.split(x,6)
    yn = np.split(y,6)
    fn = np.split(f,6)

    srednia = []
    mediana = []
    odchylenie_standardowe = []

    for i in fn: 
        srednia.append(np.mean(i))
        mediana.append(np.median(i))
        odchylenie_standardowe.append(np.std(i))

    
    xn = np.arange(len(yn))
    width = 0.2

    plt.bar(xn, srednia, width, color='red', label='Średnia')
    plt.bar(xn + width, mediana, width, label='Mediana')
    plt.bar(xn + 2*width, odchylenie_standardowe, width, label='Odchylenie standardowe')
    plt.title('Analiza danych: srednia, meidana, odchylanie standardowe')
    plt.legend()
    plt.show()

#z2_srednia_mediana_odchylenie()

#- wyznaczyć jedną funkcję interpolacyjną wielomianową dla wybranej współrzędnej y  z siatki (obowiązkowo)

def interpolacja_gaussa(P, R):
    n = P.shape[0]
    M = np.zeros([n, n+1])
    M[:,:n] = P
    M[:,n] = R
    x = np.zeros(n)

    for s in range(n-1):
        for i in range(s+1, n):
            for j in range(s+1, n+1):
                M[i, j] = M[i,j] - ((M[i,s] / M[s,s]) * M[s,j])
    x[n-1] = M[n-1,n]/M[n-1,n-1]

    for i in range(n-2, -1, -1):
        sum = 0
        for s in range(i+1, n):
            sum+=M[i,s]*x[s]
        x[i] = (M[i,n] - sum) / M[i,i]
    return x

def interpolacja_wielomianowa(P,R):
    n = len(P)
    A = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            A[i,j] = P[i] ** j
    return interpolacja_gaussa(A,R)

def wartosc_y(P, R):
    n = len(R)
    n1 = len(P)
    y = np.zeros(n)
    for i in range(n):
        for j in range(n1):
            y[i] += (R[i]**j) * P[j]
    return y

def z3_interpolacja():
    xs = np.split(x, 6)
    fs = np.split(f, 6)

    #Podzial wykresu na podwykresy
    w1 = interpolacja_wielomianowa(xs[0][8:23],fs[0][8:23])
    w2 = interpolacja_wielomianowa(xs[0][18:27],fs[0][18:27])
    w3 = interpolacja_wielomianowa(xs[0][28:37],fs[0][28:37])
    w4 = interpolacja_wielomianowa(xs[0][:12],fs[0][:12])    
    w5 = interpolacja_wielomianowa(xs[0][33:],fs[0][33:])
    w6 = interpolacja_wielomianowa(xs[0][23:32],fs[0][23:32])

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

    plt.plot(xs[0], fs[0],'o', color = 'b', label="punkty oryginalne")

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

z3_interpolacja()

#- wyznaczyć funkcję interpolacyjną sklejaną dla wybranej współrzędnej y  z siatki (obowiązkowo) 
def f_macierzy(X, xi, h):
    xmacierz1 = xi - h
    xmacierz2 = xi - (2*h)

    xi_pkt1 = xi + h
    xi_pkt2 = xi + h * 2

    if(X>xmacierz2 and X <= xmacierz1):
        return (1/h**3) * (X - xmacierz2) ** 3
    elif(X >= xmacierz1 and X<= xi):
        return (1/h**3) * ((h**3) + (3*(h*h) * (X - xmacierz1)**2 ) - (3 * h * (x - xmacierz1)**2) - (3 * (X - xmacierz1) ** 3))
    elif(X >= xi and X <= xi_pkt1):
        return (1/h**3) * ((h**3) + (3*(h*h) * (xi_pkt1 - X)**2 ) - (3 * h * (xi_pkt1 - X)**2) - (3 * (xi_pkt1 - X) ** 3))
    elif(X >= xi_pkt2 and X <= xi_pkt2):
        return (1/h**3) * ((xi_pkt2 - X) ** 3)
    else:
        return 0

def znajdz_wartosc(x, f):
    h = x[1] - x[0]
    len2 = len(x)+2
    w = np.zeros([len2, len2])
    for i in range(1, len2-1):
        w[i,i] = 4
        w[i,i-1] = 1
        w[i,i+1] = 1
    w[0,0] = -3/h
    w[len2-1, len2-1] = 3/h
    w[0,2] = -3/h
    w[len2-1, len2-3] = -3/h

    y1 = np.zeros(len2)
    y1[1:len2-1] = f[:]
    y1[0] = 1
    y1[len2-1] = -1
    return interpolacja_gaussa(w, y1)

def s(x, k , X):
    rez = 0
    x2 = np.zeros(len(x)+2)
    x2[1:len(x)+1] = x[:]
    x2[0] = -0.1
    x2[len(x2)-1] = 4.1
    for i in range(len(x2)):
        rez += f_macierzy(X, x2[i], 0.1) * k[i]
    return rez

def z4_interpolacja_sklejana():
    k = znajdz_wartosc(x[0], f[0])
    x_s = np.linspace(0, 4, 1000)
    y1 = np.zeros(len(x_s))
    for i in range(len(x_s)):
        y1[i] = s(x[0], k, x_s[i])

    plt.plot(x[0], f[0], 'o', label = "pkt. oryginalne")
    plt.plot(x_s, y1, label = "Interpolacja splajn")
    plt.xlabel('X')
    plt.ylabel('F(x)')
    plt.title("F. interpolacyjna sklejana")
    plt.legend()
    plt.grid(True)
    plt.show()

#z4_interpolacja_sklejana()

#- dokonaj porównania funkcji interpolacyjnych 

def z5_porownanie_funkcji_interpolacyjnych():
    chosen_y = y[0]
    
    wybrane_pkt = dane[dane[:, 1] == chosen_y]
    wybrane_x = wybrane_pkt[:, 0]
    F_wybrane = wybrane_pkt[:, 2]

    n = len(wybrane_x)
    B = np.zeros((n, n))
    
    for i in range(0,n):
        for j in range(0,n):
            B[i, j] = wybrane_x[i] ** j
    
    A = np.linalg.solve(B, F_wybrane)

    def eval_polynomial(x, coeffs):
        return sum(coeffs[j] * x ** j for j in range(len(coeffs)))
    
    # //////////////////////////////////////
    X = np.zeros((len(wybrane_pkt)+2,len(wybrane_pkt)+2))
    Y = np.zeros((len(wybrane_pkt)+2))
    K = np.zeros((len(wybrane_pkt)+2))

    def wypelnianieMacierzyX():

        h = wybrane_x[1]-wybrane_x[0]
    
        for i in range(len(wybrane_pkt)+1):
            X[i,i] = 4
            X[i,i+1] = 1
            X[i+1,i] = 1
            
        X[0,0]=-3/h
        X[0,2]=3/h
        X[0,1]=0
        X[len(wybrane_pkt)+1,len(wybrane_pkt)+1]=3/h
        X[len(wybrane_pkt)+1,len(wybrane_pkt)-1]=-3/h
        X[len(wybrane_pkt)+1,len(wybrane_pkt)]=0
     
    def wypelnianieMacierzyY():
        for i in range (0,len(wybrane_pkt)):
            Y[i+1] = F_wybrane[i]
        Y[0] = 1
        Y[len(wybrane_pkt)+1] = -1
        
    wypelnianieMacierzyX()
    wypelnianieMacierzyY()

    K = np.linalg.solve(X,Y)
    print(K)
    x_values = np.linspace(np.min(wybrane_x), np.max(wybrane_x), 1000)
    wartosci_interp = []
    for x in x_values:       # obliczanie wartosci interpolowanej dla kazdego punktu x, ze wzoru na interpolacje kubiczna
        for i in range(len(wybrane_pkt)):
            if x >= wybrane_pkt[i, 0] and x <= wybrane_pkt[i+1, 0]:
                h = wybrane_pkt[i+1, 0] - wybrane_pkt[i, 0]
                t = (x - wybrane_pkt[i, 0]) / h
                interpolated_value = (1 - t) * wybrane_pkt[i, 2] + t * wybrane_pkt[i+1, 2] + (t * (1 - t) * ((h ** 2) / 6) * ((1 - t) * K[i] + t * K[i+1]))
                wartosci_interp.append(interpolated_value)
                break
        
    plt.plot(wybrane_x, F_wybrane, 'bo', label='Dane')
    plt.plot(x_values, wartosci_interp, 'r-', label='Interpolacja splajn')
    plt.grid(True)
    # ////////////////////////////////////

    # Wykres
    # plt.scatter(x_chosen, F_chosen,)

    y_interp = [eval_polynomial(x, A) for x in x_values]
    plt.plot(x_values, y_interp, label='Interpolacja wielomianowa', color='green')
    plt.xlabel('x')
    plt.ylabel('F(x,y)')
    plt.title('Porownanie interpolacji')
    plt.legend()
    plt.show()

#z5_porownanie_funkcji_interpolacyjnych()

#- wyznaczyć dwie funkcje aproksymacyjne dla wybranej jednej współrzędnej y  z siatki - obowiązkowo jedna funkcja

#- dokonaj oceny jakości aproksymacji

#- obliczyć całkę z funkcji interpolacyjnych i aproksymacyjnych - obowiązkowo jedna metoda

#- wyznaczyć pochodne cząstkowe dla wybranej linii punktów (y=const) (obowiązkowo)

#- określić monotoniczność  dla wybranej linii punktów (y=const) (obowiązkowo)