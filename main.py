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



#- wyznaczyć funkcję interpolacyjną sklejaną dla wybranej współrzędnej y  z siatki (obowiązkowo) 

#- dokonaj porównania funkcji interpolacyjnych 

#- wyznaczyć dwie funkcje aproksymacyjne dla wybranej jednej współrzędnej y  z siatki - obowiązkowo jedna funkcja

#- dokonaj oceny jakości aproksymacji

#- obliczyć całkę z funkcji interpolacyjnych i aproksymacyjnych - obowiązkowo jedna metoda

#- wyznaczyć pochodne cząstkowe dla wybranej linii punktów (y=const) (obowiązkowo)

#- określić monotoniczność  dla wybranej linii punktów (y=const) (obowiązkowo)