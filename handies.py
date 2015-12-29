"""contains handy methods."""

import math
import matplotlib.pyplot as plt
import numpy


def savepic(name='test.png', path='/home/luke/'):
    """save figure to a file."""
    figure = plt.gcf()  # get current figure
    plt.savefig(path + name)  # save
    plt.clf()  # clear plot


def equirectangular_projection(latitude, longtitude,  phi_0=0, l_0=0):
    """project latitude, longtitude to x, y plane."""
    latitude = math.radians(latitude)
    longtitude = math.radians(longtitude)
    phi_0 = math.radians(phi_0)
    l_0 = math.radians(l_0)
    x = (longtitude - l_0) * math.cos(phi_0)
    y = latitude
    return (x, y)


def drange(start, stop, step):
    """similar to range but can take float arguments."""
    r = start
    while r < stop:
        yield r
        r += step


def mrange(start, stop, step):
    """generate geometric sequence: a_n+1 = a_n * step."""
    r = start
    while r < stop:
        yield r
        r *= step


def pathlength(sequence):
    """return path(loop) length for given sequence of points(tuples/lists)."""
    return sum([math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                for p1, p2 in zip(sequence, sequence[1:] + sequence[0:1])])


def distance(p1, p2):
    """return distance between 2 points (tuples/lists)."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def distance_km(p1, p2):
    """return distance in kilometers assuming equirectangular projection."""
    r = 6371  # earth radius in km
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1]) * r


def timelength(sequence, v1, v2):
    """return time of travel with given velocity.

    sequence - set of cities
    v1 - velocity everywhere except for Poland
    v2 - velocity in Poland"""
    v_c = v2  # speed through the inside of the circle [km/h]
    v = v1  # speed everywhere except for inside of the circle [km/h]
    # geographical center of Poland:
    q = numpy.asarray(equirectangular_projection(52, 19, 50, 9))
    r = .047  # * EARTH_RADIUS = ~300km
    times = []
    times_app = times.append
    for p1, p2 in zip(sequence, sequence[1:] + sequence[0:1]):
        time = 0
        p1, p2 = numpy.asarray(p1), numpy.asarray(p2)
        l = p2 - p1
        # we can write line l as p1 + t(p2-p1), t - parameter [0,1]
        # (p1 + t*l-q) (dot) (p1 + t*l-q) = r^2
        # leads to quadratic equation at^2 +bt+c=0
        a = numpy.dot(l, l)
        b = 2 * numpy.dot(l, p1 - q)
        c = numpy.dot(p1, p1) + numpy.dot(q, q) - 2 * numpy.dot(p1, q) - r ** 2
        delta = b ** 2 - 4 * a * c
        if delta <= 0:  # no solutions or tangent
            if distance(p1, q) <= r:  # it's enough to check for only one
                time = distance_km(p1, p2) / v_c
            else:
                time = distance_km(p1, p2) / v
        else:
            t1 = (-b - math.sqrt(delta)) / (2 * a)
            t2 = (-b + math.sqrt(delta)) / (2 * a)
            if 0 < t1 < 1 or 0 < t2 < 1:
                if 0 < t2 < 1 and not(0 < t1 < 1):
                    t1 = t2
                p = p1 + t1 * (p2 - p1)
                dist1, dist2 = 0, 0
                if distance(p1, q) <= r:
                    dist1 = distance_km(p1, p)  # inside circle
                    dist2 = distance_km(p2, p)
                else:
                    dist1 = distance_km(p2, p)  # inside circle
                    dist2 = distance_km(p1, p)
                time = dist1 / v_c + dist2 / v
            elif 0 < t1 < 1 and 0 < t2 < 1:
                p3 = p1 + t1 * (p2 - p1)
                p4 = p1 + t2 * (p2 - p1)
                dist3 = distance_km(p3, p4)  # inside circle
                dist1 = distance_km(p1, p3)
                dist2 = distance_km(p1, p4)
                if dist1 > dist2:
                    dist1 = dist2
                    dist2 = distance_km(p2, p3)
                else:
                    dist2 = distance(p2, p4)
                time = dist1 / v + dist2 / v + dist3 / v_c
            else:  # would intersect circle if extended
                time = distance_km(p1, p2) / v
        # note: in 'ifs' above we dont care about situations involving any
        # of the points is on the edge of the circle, that is because
        # points are cities and circle is a country within which
        # we claim to be moving slower (or faster)
        # in our case that is Poland and also we consider
        # capital cities of EU countries and we know that none of them
        # is on the edge of a country
        times_app(time)
    return sum(times)
