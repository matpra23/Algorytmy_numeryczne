import numpy as np

import matplotlib.pyplot as plt

dane = np.loadtxt('138305.txt')

stala = 0.3
x, y, f = [], [], []

for i in range(dane.shape[0]):
    if dane[i][1] == stala:
        x.append(dane[i][0])
        y.append(dane[i][1])
        f.append(dane[i][2])

x = np.array(x)
y = np.array(y)
f = np.array(f)
n = len(x)

pochodna = np.zeros(n)
monotonicznosc = np.zeros(n)

for i in range(n):
    if i == 0:
        pochodna[i] = (f[i] - f[i+1]) / (x[i] - x[i+1])
    elif i == n-1:
        pochodna[i] = (f[i] - f[i-1]) / (x[i] - x[i-1])
    else:
        pochodna[i] = (f[i+1] - f[i-1]) / (x[i+1] - x[i-1])
    
    monotonicznosc[i] = np.sign(pochodna[i])

lin = np.linspace(x[0], x[-1], y.shape[0])

print("Przedziały monotoniczności funkcji:", monotonicznosc)
plt.title("Przedziały monotoniczności dla stałej y = 0.3")
plt.plot(lin, monotonicznosc, '-', label="monotoniczność funkcji", color='b')
plt.plot(lin, y, label='y', color='r')
plt.xlabel("X")
plt.ylabel("Wartości monotoniczności")
plt.legend()
plt.grid(True)
plt.show()

print("Wartości pochodnej:", pochodna)
plt.title("Wykres pochodnej dla stałej y = 0.3")
plt.plot(lin, pochodna, '-o', label="Wykres pochodnej", color='b')
plt.xlabel("X")
plt.ylabel("Wartości pochodnej")
plt.legend()
plt.grid(True)
plt.show()
#
