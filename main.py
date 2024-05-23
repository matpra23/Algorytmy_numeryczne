import matplotlib.pyplot as plt
import numpy as np

# #wizualizacja danych w układzie x, F(x,y) dla każdej linii y=const

# Definicja funkcji F(x, y)
dane = np.loadtxt('138305.txt')
y = dane[:,1]
def z1_wizualizacja():
    for i in y:
        points = dane[dane[:, 1] == i]  #wybieranie punktu dla konkretnej wartosci y 
        x = points[:, 0]
        F_xy = points[:, 2]
        plt.plot(x, F_xy)

    plt.xlabel('x')
    plt.ylabel('F(x,y)')
    plt.title('Wizualizacja danych F(x, y) dla każdej linii y=const')
    plt.legend()
    plt.grid(True)
    plt.show()

z1_wizualizacja()

#- wyznaczyć średnią, medianę, odchylenie standardowe z podziałem na współrzędne y, prezentacja na wykresie słupkowym (obowiązkowo)
def z2_srednia_mediana_odchylenie():
    # obliczanie:
    mean = []
    median = []
    std_deviation = []

    for i in y:
        points = dane[dane[:, 1] == i]
        F_xy = points[:, 2]
    
        mean.append(np.mean(F_xy))
        median.append(np.median(F_xy))
        std_deviation.append(np.std(F_xy))

    # wykres słupkowy:
    x_values = np.arange(len(y))
    width = 0.2

    plt.bar(x_values, mean, width, color='red', label='Średnia')
    plt.bar(x_values + width, median, width, label='Mediana')
    plt.bar(x_values + 2*width, std_deviation, width, label='Odchylenie standardowe')
    plt.legend()

    plt.show()

z2_srednia_mediana_odchylenie()

#- wyznaczyć jedną funkcję interpolacyjną wielomianową dla wybranej współrzędnej y  z siatki (obowiązkowo)

def z3_interpolacja_wielomianowa():
    chosen_y = y[0]

    #punkty dla wybranego y
    chosen_points = dane[dane[:, 1] == chosen_y]
    x_chosen = chosen_points[:, 0]
    F_chosen = chosen_points[:, 2]
    #interpolacja wielomianowa
    poly_interp = np.polyfit(x_chosen, F_chosen, deg=len(x_chosen)-1)
    poly_func = np.poly1d(poly_interp)

    #wykres oryginalnych danych i interpolacji wielomianowej
    plt.scatter(x_chosen, F_chosen, label='Dane')
    x_range = np.linspace(min(x_chosen), max(x_chosen), 100)
    plt.plot(x_range, poly_func(x_range), label='Interpolacja wielomianowa', color='red')
    plt.xlabel('x')
    plt.ylabel('F(x,y)')
    plt.title('Interpolacja wielomianowa dla y = ' + str(chosen_y))
    plt.legend()
    plt.show()

z3_interpolacja_wielomianowa()

#- wyznaczyć funkcję interpolacyjną sklejaną dla wybranej współrzędnej y  z siatki (obowiązkowo) 

def z4_interpolacja_splajn():
    wybrany_y = 0.2
    
    wybrane_pkt = dane[dane[:, 1] == wybrany_y]
    wybrany_x = wybrane_pkt[:, 0]
    F_wybrane = wybrane_pkt[:, 2]
    
    X = np.zeros((len(wybrane_pkt)+2,len(wybrane_pkt)+2))
    Y = np.zeros((len(wybrane_pkt)+2))
    K = np.zeros((len(wybrane_pkt)+2))

    def macierzX():

        h = wybrany_x[1]-wybrany_x[0]
    
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
     
    def macierzY():
        for i in range (0,len(wybrane_pkt)):
            Y[i+1] = F_wybrane[i]
        Y[0] = 1
        Y[len(wybrane_pkt)+1] = -1
        
    macierzX()
    macierzY()

    K = np.linalg.solve(X,Y)
    
    x_values = np.linspace(np.min(wybrany_x), np.max(wybrany_x), 100)
    wartosc_interpolowane = []
    for x in x_values:       # obliczanie wartosci interpolowanej dla kazdego punktu x, ze wzoru na interpolacje kubiczna
        for i in range(len(wybrane_pkt)):
            if x >= wybrane_pkt[i, 0] and x <= wybrane_pkt[i+1, 0]:
                h = wybrane_pkt[i+1, 0] - wybrane_pkt[i, 0]
                t = (x - wybrane_pkt[i, 0]) / h
                interpolated_value = (1 - t) * wybrane_pkt[i, 2] + t * wybrane_pkt[i+1, 2] + (t * (1 - t) * ((h ** 2) / 6) * ((1 - t) * K[i] + t * K[i+1]))
                wartosc_interpolowane.append(interpolated_value)
                break
        
    plt.plot(wybrany_x, F_wybrane, 'bo', label='Dane')
    plt.plot(x_values, wartosc_interpolowane, 'r-', label='Interpolacja')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Funkcja Sklejana - Interpolacja')
    plt.legend()
    plt.grid(True)
    plt.show()

z4_interpolacja_splajn()

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

z5_porownanie_funkcji_interpolacyjnych()

#- wyznaczyć dwie funkcje aproksymacyjne dla wybranej jednej współrzędnej y  z siatki - obowiązkowo jedna funkcja

#- dokonaj oceny jakości aproksymacji

#- obliczyć całkę z funkcji interpolacyjnych i aproksymacyjnych - obowiązkowo jedna metoda

#- wyznaczyć pochodne cząstkowe dla wybranej linii punktów (y=const) (obowiązkowo)

#- określić monotoniczność  dla wybranej linii punktów (y=const) (obowiązkowo)