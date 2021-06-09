import operator
import math
import random
import operator
import pygraphviz as pgv

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

data = np.loadtxt('final.txt')

# Función división protegida
def protectedDiv(left, right):
    if right == 0:
        return 1
    else:
        return left / right

def potencia2(n):
    return n ** 2

data = np.loadtxt('final.txt')

pset= gp.PrimitiveSet("main", 2)  # main es el nombre de la funcion y 2 las entradas

#aca creamos los operandos a usar

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
#agregar div con cuidado de evitar division por cero
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(protectedDiv, 2)
#pset.addPrimitive(potencia2, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)

#aca creamos las variables a usar
pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="y")


#aca creamos las constantes a usar
#pset.addTerminal(3)                                           #agrega un 3
pset.addEphemeralConstant("rand101", lambda: random.uniform(-10, 10))    #agrega un valor de -10a10

# aca se crean los arboles
#genGrow():Generate an expression where each leaf might have a different depth between min and max.
#genFull():Generate an expression where each leaf has the same depth between min and max.
#genHalfAndHalf():Generate an expression with a PrimitiveSet pset. Half the time, the expression is generated with genGrow(), the other half, the expression is generated with genFull().

expr = gp.genGrow(pset, min_=1, max_=5)    # la altura de los arboles varia de 1 a 5
tree = gp.PrimitiveTree(expr)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
               pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

#funcion que evalua a los individuos

#datos contiene x,y, puntuacion
#dx=x
#dy=y
#tag=puntuacion

def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    #Transforma el arbol en una expresion calculable

    func = toolbox.compile(expr=individual)
    sqerrors = 0
    for i in range(len(data)):
        sqerrors += (func(data[i][0],data[i][1]) - data[i][2]) ** 2

    sqerrors /= len(data)

    return sqerrors,

toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=60)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.7, 0.3, 200, stats,
                                   halloffame=hof, verbose=True)

    return pop, log, hof

x = main2(pobTemp)

poblacion = x[0]
history = x[1]
best = x[2][0]

print(best)

#arbol=gp.PrimitiveTree(x[1])
print("===========")
ejeX = []
for i in range(len(history)):
    ejeX.append(i)

ejeY = []
for i in history:
    ejeY.append(i['min'])

fig = plt.figure(figsize=(12,7))
plt.plot(ejeX, ejeY)
plt.show()

funct = toolbox.compile(best)

xx = np.linspace(0, 49, 50)
yy = np.linspace(0, 39, 40)

x1, y1 = np.meshgrid(xx, yy)

resultados = []
for i in range(len(x1)):
    temp = []
    for j in range(len(y1[i])):
        temp.append(funct(50 - x1[i][j], y1[i][j]))

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
ax.view_init(30,60)

# Mostramos el gráfico
plt.show()

f = open("Datos/evolution.txt", "w")

for i in range(len(ejeX)):
    f.write(str(ejeX[i]) + " ")
    f.write(str(ejeY[i]) + "\n")

f.close()

f = open("Datos/superficie.txt", "w")

for i in range(len(x1)):
    for j in range(len(x1[i])):
        f.write(str(x1[i][j]) + " ")
        f.write(str(y1[i][j]) + " ")
        f.write(str(resultados[i][j]) + "\n")
    f.write("\n")

f.close()

nodes, edges, labels = gp.graph(best)

# Recortar la cantidad de decimales en números del labels
def isfloat(letra):
    try:
        float(letra)
        return True
    except ValueError:
        return False

for i in range(len(labels)):
  if isfloat(str(labels[i])):
    numTemp = round(labels[i], 3)
    labels[i] = numTemp

### Graphviz Section ###

g = pgv.AGraph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
g.layout(prog="dot")

for i in nodes:
    n = g.get_node(i)
    n.attr["label"] = labels[i]

g.draw("tree.png")
