import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('138305.txt')

y_values = data[:, 1]

def wizualizacja_danych():
    for y in y_values:
        points = data[data[:, 1] == y]  #wybieranie punktu dla konkretnej wartosci y 
        x = points[:, 0]
        F_xy = points[:, 2]
    
        plt.plot(x, F_xy, label=f'y={y}')   

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('F(x,y)')
    plt.title('Wizualizacja danych dla różnych linii y=const')
    plt.show()

def srednia_mediana_odchylenie():
    # obliczanie:
    mean = []
    median = []
    std_deviation = []

    for y in y_values:
        points = data[data[:, 1] == y]
        F_xy = points[:, 2]
    
        mean.append(np.mean(F_xy))
        median.append(np.median(F_xy))
        std_deviation.append(np.std(F_xy))

    # wykres słupkowy:
    x_values = np.arange(len(y_values))
    width = 0.2

    plt.bar(x_values, mean, width, color='red', label='Średnia')
    plt.bar(x_values + width, median, width, label='Mediana')
    plt.bar(x_values + 2*width, std_deviation, width, label='Odchylenie standardowe')
    plt.legend()
    plt.show()

def interpolacja():
    chosen_y = y_values[0]
    
    chosen_points = data[data[:, 1] == chosen_y]
    x_chosen = chosen_points[:, 0]
    F_chosen = chosen_points[:, 2]

    n = len(x_chosen)
    B = np.zeros((n, n))
    
    for i in range(0,n):
        for j in range(0,n):
            B[i, j] = x_chosen[i] ** j
    
    A = np.linalg.solve(B, F_chosen)

    def eval_polynomial(x, coeffs):
        return sum(coeffs[j] * x ** j for j in range(len(coeffs)))

    # Wykres
    plt.scatter(x_chosen, F_chosen, label='Dane')
    x_range = np.linspace(min(x_chosen), max(x_chosen), 1000)
    y_interp = [eval_polynomial(x, A) for x in x_range]
    plt.plot(x_range, y_interp, label='Interpolacja wielomianowa', color='red')
    plt.xlabel('x')
    plt.ylabel('F(x,y)')
    plt.title('Interpolacja wielomianowa dla y = ' + str(chosen_y))
    plt.legend()
    plt.show()
    
def interpolacja_splajn():
    chosen_y = y_values[0]
    
    chosen_points = data[data[:, 1] == chosen_y]
    x_chosen = chosen_points[:, 0]
    F_chosen = chosen_points[:, 2]
    
    X = np.zeros((len(chosen_points)+2,len(chosen_points)+2))
    Y = np.zeros((len(chosen_points)+2))
    K = np.zeros((len(chosen_points)+2))

    def wypelnianieMacierzyX():

        h = x_chosen[1]-x_chosen[0]
    
        for i in range(len(chosen_points)+1):
            X[i,i] = 4
            X[i,i+1] = 1
            X[i+1,i] = 1
            
        X[0,0]=-3/h
        X[0,2]=3/h
        X[0,1]=0
        X[len(chosen_points)+1,len(chosen_points)+1]=3/h
        X[len(chosen_points)+1,len(chosen_points)-1]=-3/h
        X[len(chosen_points)+1,len(chosen_points)]=0
     
    def wypelnianieMacierzyY():
        for i in range (0,len(chosen_points)):
            Y[i+1] = F_chosen[i]
        Y[0] = 1
        Y[len(chosen_points)+1] = -1
        
    wypelnianieMacierzyX()
    wypelnianieMacierzyY()

    K = np.linalg.solve(X,Y)
    
    x_values = np.linspace(np.min(x_chosen), np.max(x_chosen), 1000)
    interpolated_values = []
    for x in x_values:       # obliczanie wartosci interpolowanej dla kazdego punktu x, ze wzoru na interpolacje kubiczna
        for i in range(len(chosen_points)):
            if x >= chosen_points[i, 0] and x <= chosen_points[i+1, 0]:
                h = chosen_points[i+1, 0] - chosen_points[i, 0]
                t = (x - chosen_points[i, 0]) / h
                interpolated_value = (1 - t) * chosen_points[i, 2] + t * chosen_points[i+1, 2] + (t * (1 - t) * ((h ** 2) / 6) * ((1 - t) * K[i] + t * K[i+1]))
                interpolated_values.append(interpolated_value)
                break
        
    plt.plot(x_chosen, F_chosen, 'bo', label='Dane')
    plt.plot(x_values, interpolated_values, 'r-', label='Interpolacja')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Interpolacja funkcją sklejaną')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def porownanie_interpolacji():
    chosen_y = y_values[0]
    
    chosen_points = data[data[:, 1] == chosen_y]
    x_chosen = chosen_points[:, 0]
    F_chosen = chosen_points[:, 2]

    n = len(x_chosen)
    B = np.zeros((n, n))
    
    for i in range(0,n):
        for j in range(0,n):
            B[i, j] = x_chosen[i] ** j
    
    A = np.linalg.solve(B, F_chosen)

    def eval_polynomial(x, coeffs):
        return sum(coeffs[j] * x ** j for j in range(len(coeffs)))
    
    # //////////////////////////////////////
    X = np.zeros((len(chosen_points)+2,len(chosen_points)+2))
    Y = np.zeros((len(chosen_points)+2))
    K = np.zeros((len(chosen_points)+2))

    def wypelnianieMacierzyX():

        h = x_chosen[1]-x_chosen[0]
    
        for i in range(len(chosen_points)+1):
            X[i,i] = 4
            X[i,i+1] = 1
            X[i+1,i] = 1
            
        X[0,0]=-3/h
        X[0,2]=3/h
        X[0,1]=0
        X[len(chosen_points)+1,len(chosen_points)+1]=3/h
        X[len(chosen_points)+1,len(chosen_points)-1]=-3/h
        X[len(chosen_points)+1,len(chosen_points)]=0
     
    def wypelnianieMacierzyY():
        for i in range (0,len(chosen_points)):
            Y[i+1] = F_chosen[i]
        Y[0] = 1
        Y[len(chosen_points)+1] = -1
        
    wypelnianieMacierzyX()
    wypelnianieMacierzyY()

    K = np.linalg.solve(X,Y)
    print(K)
    x_values = np.linspace(np.min(x_chosen), np.max(x_chosen), 1000)
    interpolated_values = []
    for x in x_values:       # obliczanie wartosci interpolowanej dla kazdego punktu x, ze wzoru na interpolacje kubiczna
        for i in range(len(chosen_points)):
            if x >= chosen_points[i, 0] and x <= chosen_points[i+1, 0]:
                h = chosen_points[i+1, 0] - chosen_points[i, 0]
                t = (x - chosen_points[i, 0]) / h
                interpolated_value = (1 - t) * chosen_points[i, 2] + t * chosen_points[i+1, 2] + (t * (1 - t) * ((h ** 2) / 6) * ((1 - t) * K[i] + t * K[i+1]))
                interpolated_values.append(interpolated_value)
                break
        
    plt.plot(x_chosen, F_chosen, 'bo', label='Dane')
    plt.plot(x_values, interpolated_values, 'r-', label='Interpolacja splajn')
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

def aproksymacja_f_liniowa():
    chosen_y = y_values[0]
    chosen_points = data[data[:, 1] == chosen_y]
    x_chosen = chosen_points[:, 0]
    F_chosen = chosen_points[:, 2]
    
    W = np.zeros((2,2))
    b = np.zeros((2,1))
    
    suma = 0
    
    for i in range(0,len(x_chosen)):
        suma += x_chosen[i]
    W[1][0] = suma
    W[0][1] = suma
    suma = 0
    for i in range(0,len(x_chosen)):
        suma += (x_chosen[i])**2
    W[1][1] = suma
    W[0][0] = len(x_chosen)
    suma = 0
    for i in range (0, len(F_chosen)):
        suma += F_chosen[i]
        
    b[0][0] = suma
    
    suma = 0
    
    for i in range  (0, len(F_chosen)):
        suma += x_chosen[i] * F_chosen[i]
    
    
    b[1][0] = suma
    Wynik = np.linalg.solve(W,b)
    plt.plot(x_chosen,F_chosen,"ro")
    plt.plot(x_chosen,Wynik[0]+Wynik[1]*x_chosen)
    plt.title("Aproksymacja_f_liniowa")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def aproksymacja_f_kwadratowa():
    chosen_y = y_values[0]
    chosen_points = data[data[:, 1] == chosen_y]
    x_chosen = chosen_points[:, 0]
    F_chosen = chosen_points[:, 2]
    
    W = np.zeros((3,3))
    b = np.zeros((3,1))
    
    suma = 0
    
    for i in range(0,len(x_chosen)):
        suma += x_chosen[i]
    W[1][0] = suma
    W[0][1] = suma
    suma = 0
    
    for i in range(0,len(x_chosen)):
        suma += (x_chosen[i])**2
    W[1][1] = suma
    W[2][0] = suma
    W[0][2] = suma
    W[0][0] = len(x_chosen)
    suma = 0
    
    for i in range(0,len(x_chosen)):
        suma += (x_chosen[i])**3
    W[2][1] = suma    
    W[1][2] = suma    
    suma = 0
    
    for i in range(0,len(x_chosen)):
        suma += (x_chosen[i])**4
    W[2][2] = suma
    suma = 0
    
    
    for i in range (0, len(F_chosen)):
        suma += F_chosen[i]
    b[0][0] = suma
    
    suma = 0
    
    for i in range  (0, len(F_chosen)):
        suma += x_chosen[i] * F_chosen[i]
    b[1][0] = suma
    suma = 0
    
    for i in range  (0, len(F_chosen)):
        suma += (x_chosen[i]**2) * F_chosen[i]
    b[2][0] = suma
    


    Wynik = np.linalg.solve(W,b)
    plt.plot(x_chosen,F_chosen,"ro")
    plt.plot(x_chosen,Wynik[0]+Wynik[1]*x_chosen+Wynik[2]*x_chosen*x_chosen)
    plt.title("Aproksymacja_f_kwadratowa")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()    






# wizualizacja_danych()
srednia_mediana_odchylenie()
# interpolacja()
# interpolacja_splajn()
#porownanie_interpolacji()
# aproksymacja_f_liniowa()
# aproksymacja_f_kwadratowa()
