import numpy as np
import matplotlib.pyplot as plt

h = 54
k = 3
a = np.sqrt(2000)
b = np.sqrt(1000)

cantDatos = 50

cnt = 2/cantDatos
t = 0

f = open("Frontera/nul2.txt", "w")

for i in range(cantDatos+1):
    x = h + a*np.cos(np.pi*t)
    y = k + b*np.sin(np.pi*t)
    f.write(str(round(x, 2)) + " " + str(round(y, 2)) + "\n")
    t += cnt
    t = round(t, 2)

f.close()

x = h + a*np.cos(np.pi*t)
y = k + b*np.sin(np.pi*t)

print(x, y)
