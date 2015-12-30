"""Genetic algorithms methods for the travelling salesman problem."""

import random
import handies as hf
from collections import OrderedDict


cities = OrderedDict()
# cities_names = []
# cities_indices = []

EARTH_RADIUS = 6371


class Salesman(object):
    """salesman travelling around European Union.

    Attr.:
    city_seq - order in which the salesman visits cities
    fitness - 1.0 / <time salesman needs to visit all cities>
    (and come back to starting point without visitting any city more than once)
    velocity_eu - velocity everywhere in EU except for Poland
    velocity_pol_min - min. velocity in Poland
    velocity_pol_max - max. velocity in Poland
    frequency - frequency of change between vpmin, vpmax
    diploid - whether we use haploidal or diploidal encoding (boolean)
    """
    velocity_eu = 70
    velocity_pol_min = 50
    velocity_pol_max = 80
    frequency = 0
    diploid = False

    def __init__(self, city_seq, temp=False, city_seq2=None):
        """initialize a salesman.

        city_seq - sequence of cities for salesman's journey
        city = point in space
        temp - temporary salesman, i.e. dont calculate fitness
        we dont need fitness being recalculated after crossover for example
        we just need it for the selection process, and so we can save some time
        """
        self.city_seq = city_seq
        if not self.diploid:
            if not temp:
                self.fitness = self._fitness()
        else:
            if not city_seq2:
                self.city_seq2 = random.sample(
                    list(cities.values()), len(cities))
            else:
                self.city_seq2 = city_seq2
            if not temp:
                self.fitness1, self.fitness2 = self._fitnesses()
                self.fitness = max(self.fitness1, self.fitness2)

    def _fitness(self):
        """return fitness for the salesman."""
        t = hf.timelength(self.city_seq, f=self.frequency, v1=self.velocity_eu,
                          v2=self.velocity_pol_min, v3=self.velocity_pol_max)
        if t != 0:
            return 1.0 / t
        else:
            return float('inf')

    def _fitnesses(self):
        """return fitnesses for city_seq and city_seq2."""
        t1 = hf.timelength(self.city_seq, f=self.frequency,
                           v1=self.velocity_eu, v2=self.velocity_pol_min,
                           v3=self.velocity_pol_max)
        t2 = hf.timelength(self.city_seq2, f=self.frequency,
                           v1=self.velocity_eu, v2=self.velocity_pol_min,
                           v3=self.velocity_pol_max)
        if t1 != 0:
            t1 = 1.0 / t1
        else:
            t1 = float('inf')
        if t2 != 0:
            t2 = 1.0 / t2
        else:
            t2 = float('inf')
        return t1, t2


def mfp(n=10):
    """make first population of n salesmen."""
    if n % 2 != 0:
        n += 1
    salesmen = []
    sapp = salesmen.append
    v = list(cities.values())
    for _ in range(n):
        nu_v = random.sample(v, len(v))
        sapp(Salesman(nu_v))
    return salesmen


def crossingover(population=None, pc=0.9):
    """return children of salesmen."""
    if population is None:
        return None
    else:
        children = []
        children_app = children.append
        parents = [random.sample(population, 2)
                   for _ in range(len(population) // 2)]
        for p1, p2 in parents:
            r = random.random()
            if pc > r:
                if Salesman.diploid:
                    if p1.fitness1 < p1.fitness2:
                        p1.city_seq, p1.city_seq2 = p1.city_seq2, p1.city_seq
                    if p2.fitness1 < p2.fitness2:
                        p2.city_seq, p2.city_seq2 = p2.city_seq2, p2.city_seq

                    # dominant-dominant
                    c1, c2 = cxOX(p1.city_seq, p2.city_seq)
                    # recessive-recessive
                    c3, c4 = cxOX(p1.city_seq2, p2.city_seq2)

                    # dont recalc. fitness: temp=True
                    children_app(Salesman(c1, city_seq2=c3, temp=True))
                    children_app(Salesman(c2, city_seq2=c4, temp=True))
                else:
                    c1, c2 = cxOX(p1.city_seq, p2.city_seq)
                    children_app(Salesman(c1, temp=True))
                    children_app(Salesman(c2, temp=True))
            else:
                children_app(p1)
                children_app(p2)
        return children


def mutation(population=None, pm=1e-4):
    """return mutated population."""
    if population is None:
        return None
    else:
        mutants = []
        mutants_app = mutants.append
        for s in population:
            city_seq = mutate(s.city_seq, pm)
            if Salesman.diploid:
                city_seq2 = mutate(s.city_seq2, pm)
                mutants_app(Salesman(city_seq, city_seq2=city_seq2))
            else:
                mutants_app(Salesman(city_seq))
        return mutants


def cxOX(sequence1, sequence2):
    """ordered crossover method."""
    r1 = random.randint(0, len(sequence1))
    # not len()-1 because i'll be slicing
    r2 = random.randint(0, len(sequence2))
    if r1 > r2:
        r1, r2 = r2, r1
    elif r1 == r2:
        if r2 < len(sequence1):
            r2 += 1
        else:
            r1 -= 1
    middle1 = sequence1[r1:r2]
    middle2 = sequence2[r1:r2]

    # child no. 1
    c1 = [city for city in sequence2[:r1]
          if city not in middle1 and city not in middle2]
    c1 += middle2
    c1 += [city for city in sequence2[r2:]
           if city not in middle1 and city not in c1]
    c1 += [city for city in middle1 if city not in c1]

    # child no. 2
    c2 = [city for city in sequence1[:r1]
          if city not in middle1 and city not in middle2]
    c2 += middle1
    c2 += [city for city in sequence1[r2:]
           if city not in middle1 and city not in c2]
    c2 += [city for city in middle2 if city not in c2]

    return c1, c2


def mutate(city_sequence, indpm=0.05):
    """mutate sequence, return mutant."""
    city_seq = city_sequence
    for ii in range(len(city_seq)):
        if indpm > random.random():
            jj = random.randint(0, len(city_seq) - 1)
            if ii == jj:
                if jj < len(city_seq) - 1:
                    jj += 1
                else:
                    jj -= 1
            city_seq[ii], city_seq[jj] = city_seq[jj], city_seq[ii]
    return city_seq


def selTournament(individuals, k, tournsize):
    """return k new individuals selected via tournament method.

    copied from DEAP (Distributed Evolutionary Algorithms in Python)
        https://github.com/DEAP/deap
    and modified a bit so that it works with my Salesman class
    """
    chosen = []
    for i in range(k):
        aspirants = [random.choice(individuals) for i in range(tournsize)]
        chosen.append(max(aspirants, key=lambda x: x.fitness))
    return chosen


def evolution(population, pm=1e-4, pc=0.9, tournsize=3):
    """evolve system, return changed population."""
    population = selTournament(population, k=len(population),
                               tournsize=tournsize)
    population = crossingover(population, pc=pc)
    population = mutation(population, pm=pm)

    return population


def findbest(population, k=1):
    """return k best individuals."""
    if k == 1:
        bf = 0
        ind = 0
        for ii, x in enumerate(population):
            if x.fitness > bf:
                bf = x.fitness
                ind = ii
        return population[ind]
    elif k < 1:
        return None
    else:
        return sorted(population, key=lambda s: s.fitness, reverse=True)[:k]
