"""Travelling salesman problem for member states of European Union."""

import ga_tsp as ga
import matplotlib.pyplot as plt
import numpy as np
import time
import handies as hf
from collections import OrderedDict


def main():
    """run genetic algorithm for a Travelling Salesman Problem.

    Salesman travels through all European Union states' capitals.
    Assuming velocity in all countries except for Poland to be 70km/h
    and periodically changing velocity in Poland between 50 and 80 km/h
    we minimize the time of visitting all the cities.
    Program prints runs both haploidal and diploidal algorithm,
    prints out results and shows a graph with cities and calculated routes.
    Parameters of algorithm are taken from params.txt.
    File that should look like:
    50
    80
    100
    300
    0.02
    0.9
    4
    200

    Where the values are in that order: v1, v2, t, n, pm, pc, tournsize, size.
    v1, v2 - velocities in Poland
    (for 50 and 80 there is no visible effect of diploidal solution adapting
    more swiftly than haploidal so these are not constant, good results are
    seen for v1 = 0.1 and v2 = 80000 (for example))
    t - period of change of velocity in Poland (in generations)
    n - total number of generations
    pm - probability of mutation (per gene)
    pc - prob. of crossingover
    tournsize - no. of individuals taken for the tournament (selection method)
    size - total size of population
    """
    # start_time = time.time()

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
    cities_indices = [x for x in range(len(cities))]
    cities_names = [key for key in cities.keys()]

    for key, value in cities.items():
        nu_v = hf.equirectangular_projection(
            value[0], value[1], phi_0=50, l_0=9)
        cities[key] = nu_v

    decoder = {value: key for (key, value) in cities.items()}

    ga.cities = cities
    # ga.cities_names = cities_names
    # ga.cities_indices = cities_indices
    param_names = ['v1', 'v2', 't', 'n', 'pm', 'pc', 'tournsize', 'size']
    f = open('params.txt', 'r')
    param_values = [float(l) if '.' in l else int(l) for l in f]
    f.close()
    params = dict(zip(param_names, param_values))

    ga.Salesman.diploid = True
    starters = ga.mfp(params['size'])
    v1 = params['v1']  # velocity 1 in Poland
    v2 = params['v2']  # velocity 2 in Poland
    t = params['t']  # period of change of velocity in Poland
    n = params['n']  # number of generations
    pm = params['pm']  # probabilty of mutation (per gene)
    pc = params['pc']  # probability of crossover
    tournsize = params['tournsize']

    start_time = time.time()
    salesmen = starters
    ga.Salesman.velocity_pol = v1
    path_s = ga.findbest(salesmen).fitness
    print('first population best: ' + str(round(1 / path_s, 2)) + ' hours')

    results = [[0, path_s]]
    counter = 0
    for i in range(n):
        if counter == t // 2 - 1:
            ga.Salesman.velocity_pol = v1 if ga.Salesman.velocity_pol == v2 \
                else v2
            counter = 0
        counter += 1
        salesmen = ga.evolution(salesmen, pm, pc, tournsize)
        path = ga.findbest(salesmen).fitness
        results.append([i + 1, path])

    path_d = ga.findbest(salesmen).fitness
    path_d_seq = ga.findbest(salesmen).best_seq
    print(str(n) + '-th population best (diploidal): ' +
          str(round(1 / path_d, 2)) + ' hours')
    print([decoder[x] for x in path_d_seq])
    print("Time elapsed: " + str(time.time() - start_time) + 's')

    start_time = time.time()
    salesmen = starters
    ga.Salesman.diploid = False
    ga.Salesman.velocity_pol = v1

    results2 = [[0, path_s]]
    counter = 0
    for i in range(n):
        if counter == t // 2 - 1:
            ga.Salesman.velocity_pol = v1 if ga.Salesman.velocity_pol == v2 \
                else v2
            counter = 0
        counter += 1
        salesmen = ga.evolution(salesmen, pm, pc, tournsize)
        path = ga.findbest(salesmen).fitness
        results2.append([i + 1, path])

    path_h = ga.findbest(salesmen).fitness
    path_h_seq = ga.findbest(salesmen).city_seq
    print(str(n) + '-th population best (haploidal): ' +
          str(round(1 / path_h, 2)) + ' hours')
    print([decoder[x] for x in path_h_seq])
    print("Time elapsed: " + str(time.time() - start_time) + 's')

    # plot fitnesses:
    results = np.asarray(results)
    results2 = np.asarray(results2)
    plt.plot(results[:, 0], results[:, 1], 'b-', label='diploidal')
    plt.plot(results2[:, 0], results2[:, 1], 'g-', label='haploidal')
    plt.legend(loc=4)
    plt.show()

    # plot paths:
    fig, ax = plt.subplots(1)

    starters_best_seq = ga.findbest(starters).city_seq
    starters_best_seq += [starters_best_seq[0]]  # close the loop
    starters_best_seq = np.asarray(starters_best_seq)
    plt.plot(starters_best_seq[:, 0], starters_best_seq[:, 1], 'r-', alpha=0.2)

    labels = cities_indices
    cities = np.asarray(list(ga.cities.values()))

    plt.scatter(cities[:, 0], cities[:, 1], color='r')
    for label, x, y in zip(labels, cities[:, 0], cities[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(-6, -12),
                     textcoords='offset points')
    poland_c = hf.equirectangular_projection(52, 19, 50, 9)
    poland = plt.Circle(poland_c, .047, color='r', alpha=0.3)
    ax.add_artist(poland)

    path_d_seq = path_d_seq + [path_d_seq[0]]
    path_d_seq = np.asarray(path_d_seq)

    path_h_seq = path_h_seq + [path_h_seq[0]]
    path_h_seq = np.asarray(path_h_seq)

    plt.plot(path_h_seq[:, 0],
             path_h_seq[:, 1], 'g-', label='haploidal')
    plt.plot(path_d_seq[:, 0],
             path_d_seq[:, 1], 'b-', label='diploidal')

    legend = "Legend:\n"
    legend += "\n".join([str(ii) + ': ' + name
                         for ii, name in enumerate(cities_names)])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(-0.15, 0.95, legend,
            transform=ax.transAxes,
            fontsize=14, verticalalignment='top', bbox=props)

    plt.axis('off')
    plt.legend(loc=4)
    plt.show()

if __name__ == "__main__":
    main()
