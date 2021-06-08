import numpy as np
import math
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import random

def exp1(x, y):
    a = 70/math.log(10/7)

    b = 20/math.log(10/7)

    temp = -(x-12)**2/a - (y-12)**2 / b

    resul = 100 * math.e ** temp

    return resul

def exp2(x, y):
    a = 70/math.log(10/7)

    b = 20/math.log(10/7)

    temp = -(x-38)**2/a - (y-12)**2 / b

    resul = 85 * math.e ** temp

    return resul

def zona2(x, y):
    temp = 75 * math.e ** (-(((x-12)**2)/1800)-(((y-12)**2)/500))

    return temp

def tran(x, y):

    resul = -(((x-12)**(2))/(37)) - (((y-12)**(2))/(40)) + 70

    return resul

def nula(x, y):
    resul = -(((x-12)**(2))/(35))-(((y-12)**(2))/(28))+70

    return resul

def funcionIzquierda(x, y):
    # Dentro de la mejor zona

    z = 0

    elip1 = ((x-12)**2)/70 + ((y-12)**2)/20
    elip2 = ((x-38)**2)/70 + ((y-12)**2)/20

    tran1 = ((x+4)**2)/1000 + ((y-3)**2)/500
    tran2 = ((x-54)**2)/1000 + ((y-3)**2)/500

    nul1 = ((x+4)**2)/2000 + ((y-3)**2)/1000
    nul2 = ((x-54)**2)/2000 + ((y-3)**2)/1000

    if elip1 <= 1:
        z = exp1(x, y)
    elif elip2 <= 1:
        z = exp2(x, y)
    elif tran1 <= 1 and tran2 <= 1:
        z = 70
    elif tran1 <= 1:
        z = zona2(x, y)
    elif nul1 <= 1:
        z = tran(x, y)
    else:
        z = nula(x, y)

    return z


x = np.linspace(0, 49, 50)
y = np.linspace(0, 39, 40)

x1, y1 = np.meshgrid(x, y)

resultados = []
for i in range(len(x1)):
    temp = []
    for j in range(len(y1[i])):
        temp.append(funcionIzquierda(x1[i][j], y1[i][j]))

    resultados.append(temp)

resultados = np.array(resultados)

# Creamos la figura
fig = plt.figure(figsize=(12,7))
# Creamos el plano 3D
ax = fig.gca(projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("f(x, y)")

# Agregamos los puntos en el plano 3D
ax.plot_surface(x1, y1, resultados, cstride=1, rstride=1)

# Mostramos el gráfico
plt.show()


for i in range(len(resultados)):
    for j in range(len(resultados[i])):
        num = (random.random() * 2 - 1) * 0.05
        resultados[i][j] = resultados[i][j] * (1 + num)

# Creamos la figura
fig = plt.figure(figsize=(12,7))
# Creamos el plano 3D
ax = fig.gca(projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("f(x, y)")

# Agregamos los puntos en el plano 3D
ax.plot_surface(x1, y1, resultados, cstride=1, rstride=1)

# Mostramos el gráfico
plt.show()

lista = []

for i in range(len(x1)):
    for j in range(len(x1[i])):
        temp = []
        temp.append(x1[i][j])
        temp.append(y1[i][j])
        temp.append(resultados[i][j])
        lista.append(temp)

lista = np.array(lista)

print(len(lista))

print(lista)

np.savetxt('datos.txt', lista)
