"""Travelling salesman problem for member states of European Union."""

import ga_tsp as ga
import matplotlib.pyplot as plt
import numpy as np
import time
import handies as hf
from collections import OrderedDict

start_time = time.time()

# city: (latitude, longtitude) +/- 0.5 degree
# geocenter for EU: 50 9
cities = {
    'Vienna': (48, 16),
    'Brussels': (51, 4),
    'Sofia': (43, 23),
    'Zagreb': (46, 16),
    'Nicosia': (35, 33),
    'Prague': (50, 14),
    'Copenhagen': (55, 13),
    'Tallinn': (59, 25),
    'Helsinki': (60, 25),
    'Paris': (49, 2),
    'Berlin': (53, 13),
    'Athens': (38, 24),
    'Budapest': (48, 19),
    'Dublin': (53, -6),
    'Rome': (42, 13),
    'Riga': (57, 24),
    'Vilnius': (55, 25),
    'Luxembourg': (50, 6),
    'Valletta': (36, 15),
    'Amsterdam': (52, 5),
    'Warsaw': (52, 21),
    'Lisbon': (39, -9),
    'Bucharest': (44, 26),
    'Bratislava': (48, 17),
    'Ljubljana': (46, 15),
    'Madrid': (40, -4),
    'Stockholm': (59, 18),
    'London': (52, 0)
}

cities = OrderedDict(sorted(cities.items(), key=lambda t: t[0]))
cities_names = [key for key in cities.keys()]
cities_indices = [x for x in range(len(cities))]

for key, value in cities.items():
    nu_v = hf.equirectangular_projection(value[0], value[1], phi_0=50, l_0=9)
    cities[key] = nu_v

ga.cities = cities
# ga.cities_names = cities_names
# ga.cities_indices = cities_indices

starters = ga.mfp(200)
n = 200
pm = 0.02
pc = 0.9
tournsize = 4
# f = open("/home/luke/TSP2.data", "a")
# f.write("size\tsteps\tpc\tpm\tpath[km]\n")

salesmen = starters

# path_s = hf.pathlength(ga.findbest(salesmen).city_seq) * ga.EARTH_RADIUS
path_s = 1.0 / ga.findbest(salesmen).fitness
print('first population best: ', path_s)

for i in range(n):
    salesmen = ga.evolution(salesmen, pm, pc, tournsize)
    # path = 1.0 / ga.findbest(salesmen).fitness
    # path = hf.pathlength(ga.findbest(
    #     salesmen).city_seq) * ga.EARTH_RADIUS
    # print(i + 1, path)

path = 1.0 / ga.findbest(salesmen).fitness
print(str(n) + '-th population best: ', path)
print("Time elapsed: " + str(time.time() - start_time) + 's')

# plot:
fig, ax = plt.subplots(1)

starters_best_seq = ga.findbest(starters).city_seq
starters_best_seq += [starters_best_seq[0]]
starters_best_seq = np.asarray(starters_best_seq)
plt.plot(starters_best_seq[:, 0], starters_best_seq[:, 1], 'b-', alpha=0.2)

labels = cities_indices
cities = np.asarray(list(ga.cities.values()))

plt.scatter(cities[:, 0], cities[:, 1])
for label, x, y in zip(labels, cities[:, 0], cities[:, 1]):
    plt.annotate(label, xy=(x, y), xytext = (-6, -12),
                 textcoords = 'offset points')
poland_c = hf.equirectangular_projection(52, 19, 50, 9)
poland = plt.Circle(poland_c, .047, color='r', alpha=0.3)
ax.add_artist(poland)

best_seq = ga.findbest(salesmen).city_seq
best_seq = best_seq + [best_seq[0]]
best_seq = np.asarray(best_seq)

plt.plot(best_seq[:, 0],
         best_seq[:, 1], 'g-')

legend = "Legend:\n"
legend += "\n".join([str(ii) + ': ' + name
                     for ii, name in enumerate(cities_names)])
# legend += "\nPath length: " + str(round(path, 1)) + ' km'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(-0.15, 0.95, legend,
        transform=ax.transAxes,
        fontsize=14, verticalalignment='top', bbox=props)

plt.axis('off')
plt.show()

# save to file:
# savepic(name=str(size) + 'TS_' + str(n) + 'steps_' +
#         str(pc) + 'pc_' + str(pm) + 'pm_' + str(path) + 'km.png')
# output.write(str(len(salesmen)) + '\t' + str(n) + '\t' +
#              str(pc) + '\t' + str(pm) + '\t' + str(path) + '\n')

# f.close()