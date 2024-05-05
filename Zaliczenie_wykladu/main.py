import matplotlib.pyplot as plt
import numpy as np

#wizualizacja danych w układzie x, F(x,y) dla każdej linii y=const

# Definicja funkcji F(x, y)
x = np.loadtxt('x.txt')
y = np.loadtxt('y.txt')
def F(x, y):
    return np.sin(x) + np.cos(y)

# Zakres x i y
x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.linspace(-2*np.pi, 2*np.pi, 100)

# Tworzenie siatki danych X i Y
X, Y = np.meshgrid(x, y)

# Wartości funkcji F(x, y) dla siatki danych
Z = F(X, Y)

# Wizualizacja danych dla każdej linii y=const
plt.figure(figsize=(10, 6))
for i in range(len(y)):
    plt.plot(x, Z[i, :], label=f"y = {y[i]:.2f}")

plt.xlabel('x')
plt.ylabel('F(x,y)')
plt.title('Wizualizacja danych F(x, y) dla każdej linii y=const')
plt.legend()
plt.grid(True)
plt.show()

#- wyznaczyć średnią, medianę, odchylenie standardowe z podziałem na współrzędne y, prezentacja na wykresie słupkowym (obowiązkowo)

#- wyznaczyć jedną funkcję interpolacyjną wielomianową dla wybranej współrzędnej y  z siatki (obowiązkowo)

#- wyznaczyć funkcję interpolacyjną sklejaną dla wybranej współrzędnej y  z siatki (obowiązkowo) 

#- dokonaj porównania funkcji interpolacyjnych 

#- wyznaczyć dwie funkcje aproksymacyjne dla wybranej jednej współrzędnej y  z siatki - obowiązkowo jedna funkcja

#- dokonaj oceny jakości aproksymacji

#- obliczyć całkę z funkcji interpolacyjnych i aproksymacyjnych - obowiązkowo jedna metoda

#- wyznaczyć pochodne cząstkowe dla wybranej linii punktów (y=const) (obowiązkowo)

#- określić monotoniczność  dla wybranej linii punktów (y=const) (obowiązkowo)