# -*- coding: utf-8 -*-

# modules
import numpy as np
import time
from matplotlib import pyplot as plt
import math


def ReductionGauss(Aaug):
    n, m = Aaug.shape
    for i in range(n-1):
        if Aaug[i, i] == 0:
            print(
                "L'un des pivots est nul, on ne peut pas continuer avec la méthode de Gauss")
            return

        else:
            for k in range(i+1, n):
                g = Aaug[k, i]/Aaug[i, i]
                Aaug[k] = Aaug[k]-g*Aaug[i]
    return(Aaug)


def ResolutionSystTriSup(Taug):
    n, m = np.shape(Taug)
    print(n)
    if m != n+1:
        print('pas une matrice augmentée')
        return
    X = np.zeros(n)
    for i in range(n-1, -1, -1):
        somme = 0
        for k in range(i, n):
            somme = somme + X[k]*Taug[i, k]
        if Taug[i, i] == 0:
            X[i] = 0
        else:
            X[i] = (Taug[i, -1]-somme)/Taug[i, i]
    return X


def ResolutionSystTriInf(Taug):
    n, m = Taug.shape
    if m != n+1:
        print('pas une matrice augmentée')
        return
    X = np.zeros(n)
    for i in range(0, n, 1):
        somme = 0
        for k in range(n):
            somme = somme + X[k]*Taug[i, k]
        if Taug[i, i] == 0:
            X[i] = 0
        else:
            X[i] = (Taug[i, -1]-somme)/Taug[i, i]
    return X


def Gauss(A, B):
    Aaug = np.concatenate((A, B), axis=1)
    Taug = ReductionGauss(Aaug)
    X = ResolutionSystTriSup(Taug)
    return X


def DecompositionLU(A):
    n, m = np.shape(A)
    U = np.copy(A)
    L = np.eye(n)
    for i in range(n-1):
        for k in range(i+1, n):
            pivot = U[i, i]

            pivot = U[k, i]/pivot
            L[k, i] = pivot
            for j in range(i, n):
                U[k, j] = U[k, j]-(pivot*U[i, j])
    return L, U


def ResolutionLU(A, B):
    n, m = np.shape(A)
    X = np.zeros(n)
    L, U = DecompositionLU(A)
    Y = ResolutionSystTriInf(np.concatenate((L, B), axis=1))
    Y1 = np.asarray(Y).reshape(n, 1)
    X = ResolutionSystTriSup(np.concatenate((U, Y1), axis=1))
    return (X)


print("-----------")
print("\n")
print("Question 1 :")
print("\n")
print("-----------")


def Cholesky(A):
    n, m = np.shape(A)
    L = np.zeros((n, m))
    L[0][0] = math.sqrt(A[0][0])
    S = 0
    S2 = 0
    for i in range(0, n):
        for j in range(0, i+1):
            L[i][0] = A[i][0]/L[0][0]
            S = 0
            S2 = 0
            for k in range(j):
                S += L[j][k]**2
                S2 += L[i][k]*L[j][k]
                if S == 0:
                    return "Erreur: Matrice non définie positive!"
    return L


M = np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])

while np.linalg.det(M) == 0:
    M = np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])
A = M.dot(M.T)
print(Cholesky(A))

B = np.random.randint(low=1, high=100, size=(3, 1))

print("-----------")
print("\n")
print("Question 2 :")
print("\n")
print("-----------")


def ResolCholesky(A, B):
    L = Cholesky(A)
    Taug = np.hstack((L, B))
    Y = ResolutionSystTriInf(Taug)
    Y = Y[:, np.newaxis]
    LT = np.transpose(L)
    Baug = np.hstack((LT, Y))
    X = ResolutionSystTriSup(Baug)
    return (X)


print("La solution du système AX=B avec la méthode de cholesky, en sachant que A= \n",
      A, "\n", "B=\n", B, "\n", "est:\n", ResolCholesky(A, B))

M = np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])
while np.linalg.det(M) == 0:
    M = np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])
A = M.dot(M.T)
B = np.array([[7, 12, 3]])


# courbes temps

temps = []
indices = []
tempslog = []
indiceslog = []
tempsLU = []
indicesLU = []
tempsLUlog = []
indicesLUlog = []
tempscholesky = []
indicescholesky = []
tempscholeskylog = []
indicescholeskylog = []
tempslinalgsolve = []
indiceslinalgsolve = []
tempslinalgsolvelog = []
indiceslinalgsolvelog = []
tempslinalgcholesky = []
indiceslinalgcholesky = []
tempslinalgcholeskylog = []
indiceslinalgcholeskylog = []


for n in range(50, 500, 50):
    A = np.random.rand(n, n)
    B = np.random.rand(n, 1)
    t1 = time.time()
    X = Gauss(A, B)
    t2 = time.time()
    t_operation = t2 - t1
    temps.append(t_operation)
    indices.append(n)
    indiceslog.append(math.log(n))
    tempslog.append(math.log(t_operation))

for n in range(50, 500, 50):
    A = np.random.rand(n, n)
    B = np.random.rand(n, 1)
    t3 = time.time()
    X = ResolutionLU(A, B)
    t4 = time.time()
    t_operation = t4 - t3
    tempsLU.append(t_operation)
    indicesLU.append(n)
    indicesLUlog.append(math.log(n))
    tempsLUlog.append(math.log(t_operation))

for n in range(50, 500, 50):
    A = np.random.rand(n, n)
    B = np.random.rand(n, 1)
    t5 = time.time()
    X = ResolCholesky(A, B)
    t6 = time.time()
    t_operation = t6 - t5
    tempscholesky.append(t_operation)
    indicescholesky.append(n)
    indicescholeskylog.append(math.log(n))
    tempscholeskylog.append(math.log(t_operation))

for n in range(50, 500, 50):
    A = np.random.rand(n, n)
    B = np.random.rand(n, 1)
    t7 = time.time()
    X = np.linalg.solve(A, B)
    t8 = time.time()
    t_operation = t8 - t7
    tempslinalgsolve.append(t_operation)
    indiceslinalgsolve.append(n)
    indiceslinalgsolvelog.append(math.log(n))
    tempslinalgsolvelog.append(math.log(t_operation))

for n in range(50, 500, 50):
    M = np.random.rand(n, n)
    A = np.dot(M, np.transpose(M))
    B = np.random.rand(n, 1)
    t9 = time.time()
    L = np.linalg.cholesky(A)
    Taug = np.hstack((L, B))
    Y = ResolutionSystTriInf(Taug)
    Y = Y[:, np.newaxis]
    LT = np.transpose(L)
    Baug = np.hstack((LT, Y))
    X = ResolutionSystTriSup(Baug)
    t10 = time.time()
    t_operation = t10 - t9
    tempslinalgcholesky.append(t_operation)
    indiceslinalgcholesky.append(n)
    indiceslinalgcholeskylog.append(math.log(n))
    tempslinalgcholeskylog.append(math.log(t_operation))



plt.plot(indices, temps, color='red', label="Gauss")
plt.plot(indicesLU,tempsLU,color='blue', label="LU")
plt.plot(indicescholesky, tempscholesky, color='orange', label="Cholesky")
plt.plot(indiceslinalgsolve, tempslinalgsolve, color='green', label="Linalgsolve")
plt.plot(indiceslinalgcholesky, tempslinalgcholesky, color='black', label="Numpy Cholesky")
plt.xlabel("taille de la matrice (n)")
plt.ylabel("temps d'execution (en s)")
plt.title("Temps d'execution en fonction de la taille de la matrice")
plt.legend()
plt.show()     

plt.plot(indiceslog, tempslog, color='red', label="Gauss")
plt.plot(indicesLUlog, tempsLUlog, color='blue', label="LU")
plt.plot(indicescholeskylog, tempscholeskylog,
         color='orange', label="Cholesky")
plt.plot(indiceslinalgsolvelog, tempslinalgsolvelog,
         color='green', label="Linalgsolve")
plt.plot(indiceslinalgcholeskylog, tempslinalgcholeskylog,
         color='black', label="Numpy Cholesky")
plt.xlabel("log n")
plt.ylabel("log t")
plt.title("Temps d'execution en fonction de la taille de la matrice")
plt.legend()
plt.show()


# courbes erreurs
erreur = []
indices = []
erreurLU = []
indicesLU = []
erreurcholesky = []
indicescholesky = []
erreurlinalgsolve = []
indiceslinalgsolve = []
erreurlinalgcholesky = []
indiceslinalgcholesky = []


for n in range(50, 500, 50):
    A = np.random.rand(n, n)
    B = np.random.rand(n, 1)
    X = Gauss(A, B)
    e = np.linalg.norm(np.dot(A, X)-np.ravel(B))
    erreur.append(e)
    indices.append(n)


for n in range(50, 500, 50):
    A = np.random.rand(n, n)
    B = np.random.rand(n, 1)
    X = ResolutionLU(A, B)
    e1 = np.linalg.norm(np.dot(A, X)-np.ravel(B))
    erreurLU.append(e1)
    indicesLU.append(n)


for n in range(50, 500, 50):
    M = np.random.rand(n, n)
    A = np.dot(M, np.transpose(M))
    B = np.random.rand(n, 1)
    X = ResolCholesky(A, B)
    e2 = np.linalg.norm(np.dot(A, X)-np.ravel(B))
    erreurcholesky.append(e2)
    indicescholesky.append(n)

for n in range(50, 500, 50):
    A = np.random.rand(n, n)
    B = np.random.rand(n, 1)
    X = np.linalg.solve(A, B)
    e3 = np.linalg.norm(np.dot(A, X)-np.ravel(B))
    erreurlinalgsolve.append(e3)
    indiceslinalgsolve.append(n)

for n in range(50, 500, 50):
    M = np.random.rand(n, n)
    A = np.dot(M, np.transpose(M))
    B = np.random.rand(n, 1)
    L = np.linalg.cholesky(A)
    Taug = np.hstack((L, B))
    Y = ResolutionSystTriInf(Taug)
    Y = Y[:, np.newaxis]
    LT = np.transpose(L)
    Baug = np.hstack((LT, Y))
    X = ResolutionSystTriSup(Baug)
    e4 = np.linalg.norm(np.dot(A, X)-np.ravel(B))
    erreurlinalgcholesky.append(e4)
    indiceslinalgcholesky.append(n)


plt.plot(indices, erreur, color='red', label="Gauss")
plt.plot(indicesLU,erreurLU,color='blue', label="LU")
plt.plot(indicescholesky, erreurcholesky, color='orange', label="Cholesky")
plt.plot(indiceslinalgsolve, erreurlinalgsolve, color='green', label="Linalgsolve")
plt.plot(indiceslinalgcholesky, erreurlinalgcholesky, color='black', label="Numpy Cholesky")
plt.xlabel("taille de la matrice (n)")
plt.ylabel("erreurs")
plt.title("Estimation de l'erreur commise")
plt.legend()
plt.show()
