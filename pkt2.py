import numpy as np

import matplotlib.pyplot as plt

dane = np.loadtxt('138305.txt')
x = dane[:, 0]
y = dane[:, 1]
f = dane[:, 2]

def srednia_mediana_odchylenie():

    xn = np.split(x, 6)
    yn = np.split(y, 6)
    fn = np.split(f, 6)

    srednia = []
    mediana = []
    odchylenie_standardowe = []

    for i in fn:
        srednia.append(np.mean(i))
        mediana.append(np.median(i))
        odchylenie_standardowe.append(np.std(i))

    xn = np.arange(len(yn))
    width = 0.2

    print("Wartosci sredniej: ")
    print(srednia)
    print("")
    print("Wartosci mediany: ")
    print(mediana)
    print("")
    print("Wartosci odchylenia standardowego: ")
    print(odchylenie_standardowe)

    plt.bar(xn, srednia, width, color='red', label='Średnia')
    plt.bar(xn + width, mediana, width, label='Mediana')
    plt.bar(xn + 2 * width, odchylenie_standardowe, width, label='Odchylenie standardowe')
    plt.title('Analiza: średnia, mediana, odchylenie standardowe')
    plt.legend()
    plt.show()
#
srednia_mediana_odchylenie()