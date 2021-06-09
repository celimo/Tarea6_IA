import numpy as np
import math
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import random

# Las siguientes funciones están pensadas para las puntuaciones del brazo izquierdo

# Se define la función exponencial para la zona preferencial izquierda
# Función de dos parámetros f(x, y) que retorna un valor real
def exp1(x, y):
    a = 70/math.log(10/7)

    b = 20/math.log(10/7)

    temp = -(x-12)**2/a - (y-12)**2 / b

    resul = 100 * math.e ** temp

    return resul

# Se define la función exponencial para la zona preferencial derecha
# Función de dos parámetros f(x, y) que retorna un valor real
def exp2(x, y):
    a = 70/math.log(10/7)

    b = 20/math.log(10/7)

    temp = -(x-38)**2/a - (y-12)**2 / b

    resul = 85 * math.e ** temp

    return resul

# Se define el paraboloide que describe la puntuación en la zona exterior preferente
def zona2(x, y):
    temp = 75 * math.e ** (-(((x-12)**2)/1800)-(((y-12)**2)/500))

    return temp

# Se define el paraboloide que describe la puntuación en la zona de transición
def tran(x, y):

    resul = -(((x-12)**(2))/(37)) - (((y-12)**(2))/(40)) + 70

    return resul

# Se decribe el paraboloide que describe la pntuación en la puntuación nula
def nula(x, y):
    resul = -(((x-12)**(2))/(35))-(((y-12)**(2))/(28))+70

    return resul

# Puntuación del brazo izquierdo en función de x y y
def funcionIzquierda(x, y):
    # Dentro de la mejor zona

    z = 0

    # Se obtienen constanes que son utilizadas para determinar la región donde se encuentran
    # los puntos
    # Zonas prefernetes
    elip1 = ((x-12)**2)/70 + ((y-12)**2)/20
    elip2 = ((x-38)**2)/70 + ((y-12)**2)/20

    # Zona de transición
    tran1 = ((x+4)**2)/1000 + ((y-3)**2)/500
    tran2 = ((x-54)**2)/1000 + ((y-3)**2)/500

    # Zona nula
    nul1 = ((x+4)**2)/2000 + ((y-3)**2)/1000
    nul2 = ((x-54)**2)/2000 + ((y-3)**2)/1000

    if elip1 <= 1: # Determinar si está en la zona preferente izquierda
        z = exp1(x, y)
    elif elip2 <= 1: # Determinar si está en la zona preferente derecha
        z = exp2(x, y)
    elif tran1 <= 1 and tran2 <= 1: # Determinar si está en la zona de intersección
        z = 70
    elif tran1 <= 1: # Determinar si está en la zona 2
        z = zona2(x, y)
    elif nul1 <= 1: # Determinar si está en la zona de transición
        z = tran(x, y)
    else: # Al final se evalua en la zona nula
        z = nula(x, y)

    return z

# Se crea la superficie para visualizar el comportamiento de mejor manera
x = np.linspace(0, 49, 50)
y = np.linspace(0, 39, 40)

# Se ponen en el formato que utiliza matplotlib
x1, y1 = np.meshgrid(x, y)

resultados = []
for i in range(len(x1)): # Se calculan los resultados
    temp = []
    for j in range(len(y1[i])):
        temp.append(funcionIzquierda(x1[i][j], y1[i][j]))

    resultados.append(temp)

# Se obtienen los resultados en un formato numpy
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

# Se guardan los valores para ser graficados en otro sofware
f = open("supSinRuido.txt", "w")

for i in range(len(x1)):
    for j in range(len(x1[i])):
        f.write(str(x1[i][j]) + " ")
        f.write(str(y1[i][j]) + " ")
        f.write(str(resultados[i][j]) + "\n")
    f.write("\n")

# Se agrega el ruido a los datos
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

# Se generan los datos en el formato que trabaja la libreria deap
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

# Se guardan los valores creados
np.savetxt('datos.txt', lista)

# Se guardan los puntos con ruido en formato .txt para ser graficados en otro sofware
f = open("supTeorica.txt", "w")

for i in range(len(x1)):
    for j in range(len(x1[i])):
        f.write(str(x1[i][j]) + " ")
        f.write(str(y1[i][j]) + " ")
        f.write(str(resultados[i][j]) + "\n")
    f.write("\n")
