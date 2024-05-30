import numpy as np

import matplotlib.pyplot as plt

dane = np.loadtxt('138305.txt')
x = dane[:, 0]
y = dane[:, 1]
f = dane[:, 2]

xn = np.split(x, 6)
yn = np.split(y, 6)
fn = np.split(f, 6)

def rysunek():
    for i in range(6):
        plt.plot(xn[i], fn[i], '-o', label=f"y = {yn[i][0]}")

    plt.xlabel('x')
    plt.ylabel('F(x,y)')
    plt.title('Wizualizacja danych F(x, y) dla każdej linii y = const')
    plt.legend()
    plt.grid(True)
    plt.show()

rysunek()