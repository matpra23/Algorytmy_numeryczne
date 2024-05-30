import numpy as np
import matplotlib.pyplot as plt

dane = np.loadtxt('138305.txt')
xn = np.array(sorted(list(set(dane[:,0]))))
print('X:',xn)
yn = np.array(sorted(list(set(dane[:,1])))) 
print('Y:',yn)
fn = np.zeros((yn.shape[0],xn.shape[0])) 
for i in range (yn.shape[0]):
    for j in range (xn.shape[0]):
        for k in range (dane.shape[0]):
            if dane[k,0] == xn[j] and dane[k,1] == yn[i]: 
                fn[i][j] = dane[k,2]
print('F:',fn)
stala = 0.2

def interpolacja_Lag(x,y,f,p): 
    Xi=np.linspace(x[0],x[-1],10001) 
    Zi=np.zeros(Xi.shape[0]) 
     
    i = np.where(y == p)[0][0] 
    Z = np.copy(f[i,:])
 
    print("Współczynniki wielomianów interpolacyjnych Lagrange'a:") 
     
    xp=0 
    xk=4 
    while xp < len(x)-1: 
        A=np.zeros(5)
        for k in range(A.shape[0]): 
            mianownik = 1 
            for j in range(xp,xk+1): 
                if xp+k!=j: 
                    mianownik*=x[xp+k]-x[j] 
            A[k] = Z[xp+k]/mianownik 
         
        for i in range(Xi.shape[0]): 
            if x[xp] <= Xi[i] and Xi[i] <= x[xk]: 
                sum=0 
                for k in range(A.shape[0]): 
                    iloczyn=1 
                    for j in range(xp,xk+1): 
                        if xp+k!=j: 
                            iloczyn*=Xi[i]-x[j]       
                    sum+=A[k]*iloczyn 
                Zi[i]=sum 
         
        xp=xk 
        xk+=4 
        print(A) 
 
    return (Xi,Zi,x,Z) 
 
def B(x): 
    h=x[1]-x[0] 
    if(-2*h<= x <=-h): 
        return (1/h**3)*(x-(-2*h))**3 
    if(-h<= x <=0): 
        return (1/h**3)*(h**3+3*h**2*(x+h)+3*h*(x+h)**2-3*(x+h)**3) 
    if(0<= x <=h): 
        return (1/h**3)*(h**3+3*h**2*(h-x)+3*h*(h-x)**2-3*(h-x)**3) 
    if(h<= x <=2*h): 
        return (1/h**3)*(2*h-x)**3 
    else: 
        return 0 

def metoda_eliminacji_gaussa(M, d):
    n = M.shape[0]
    c = np.zeros((n, n+1))
    c[:][0:n] = M 
    c[:][n] = d
    X = np.zeros(n)
    for s in range(0, n-1):
        for i in range(s+1, n):
            for j in range(s+1, n+1):
                c[i, j] = (c[i, j] - c[i, s]/c[s,s] * c[s,j])
    X[n-1] = c[n-1, n] / c[n-1, n-1]

    for i in range(n-1, -1, -1):
        sum = 0
        for s in range(i+1, n):
            sum = sum + c[i, s] * X[s]
        X[i] = (c[i, n] - sum) / c[i,i]
    return X
 
def interp_splajn(x,f): 
    n = len(x)
    M=np.zeros((n+2,n+2)) 
    B=np.zeros(n+2, dtype = object) 
    h=x[1]-x[0] 
    M[0][0]=-3/h 
    M[0][2]=3/h 
    M[n+1][n+1]=3/h 
    M[n+1][n-1]=-3/h
    for i in range(1,n+1):
        M[i][i]=4 
        M[i][i-1]=1 
        M[i][i+1]=1
    B[n+1] = -1
    B[0] = 1
    for i in range(1, n+1): #index out of bounds
        B[i] = f[i-1]
    K = metoda_eliminacji_gaussa(M,B) 
    return K 
         
def rysuj_intrpolacje(xn,yn,fn,p): 
    L = interpolacja_Lag(xn,yn,fn,p) 
    S = interp_splajn(xn, fn) 
    plt.subplots() 
    plt.plot(L[0],L[1],'r-',linewidth=2.0, label="f(x) Lagrange'a") 
    plt.plot(L[2],L[3],'bo',label='[x,z]') 
    plt.xlabel('x') 
    plt.ylabel('z') 
    plt.title("Funkcje interpolacyjne dla y = 0.5") 
    plt.grid() 
    plt.legend() 
    plt.show() 

def rysuj_interpolacje(x, y, f, p):
    L = interpolacja_Lag(x, y, f, p)
    S = interp_splajn(xn, fn)
    xp = 0
    xk = 4
    for i in range(len(x) // 5 + 1):
        if xk >= len(x): 
            xk = len(x) - 1
        mask = (L[0] >= x[xp]) & (L[0] <= x[xk])
        plt.plot(L[0][mask], L[1][mask], color='r', linewidth=1.5, label=f"{i + 1}")
        xp = xk 
        xk += 5
    plt.plot(S[0],S[1],'g-',linewidth=2.0, label='f(x) splajn') 
    plt.plot(L[2], L[3], 'bo', label='punkty')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('F(x,y)', fontsize=14)
    plt.title("Interpolacja Lagrange'a dla y = 0.5", 
    fontsize=16) 
    plt.grid()
    plt.legend(fontsize=12) 
    plt.show()

print("X", xn)
print()
print("Y", yn)
print()
print("F", fn)
print()
print(stala)
print()

rysuj_intrpolacje(xn, yn, fn, stala)