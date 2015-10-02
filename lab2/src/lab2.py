from cvxopt.solvers import qp
from cvxopt.base import matrix

import numpy, pylab, random, math

def kernel_linear(x, y):
    return x[0]*y[0] + x[1]*y[1] + 1
def kernel_radial(x, y):
    tup = (x[0] - y[0], x[1] - y[1])
    sigma = 2
    return math.exp(-(numpy.dot(tup, tup))/(2*pow(sigma, 2)))

def p_matrix(data, func):
    matrix = [[0 for x in range(len(data))] for x in range(len(data))]
    for i in range(len(data)):
        for j in range(len(data)):
            matrix[i][j] = data[i][2] * data[j][2] * func(data[i], data[j])
    return matrix

def indicator(datapoint, newalpha, func):
    notsum = 0.0
    for a in newalpha:
        notsum += a[1] * a[0][2] * func(datapoint, a[0])
    return notsum

kernel_func = kernel_radial

def main():
    classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)] + \
            [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)]
    classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(10)]

    data = classA + classB
    random.shuffle(data)

    q = matrix(-1, (len(data), 1), 'd')
    h = matrix(0, (len(data), 1), 'd')
    G = [[-1.0 if i == j else 0 for i in range(len(data))] for j in range(len(data))]

    r = qp(matrix(p_matrix(data, kernel_func)), q, matrix(G), h)
    alpha = list(r['x'])
    newalpha = []
    for i, a in enumerate(alpha):
        if a > 0.00001:
            newalpha.append((data[i], a))

    # print alpha
    # print len(alpha)
    # print len(newalpha)
    # print newalpha

    xrange = numpy.arange(-4, 4, 0.05)
    yrange = numpy.arange(-4, 4, 0.05)

    grid = matrix([[indicator((x, y), newalpha, kernel_func) for y in yrange] for x in xrange])
    pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors = ('red', 'black', 'blue'), linewidths = (1, 3, 1))
    pylab.hold(True)

    pylab.plot([p[0] for p in classA],
            [p[1] for p in classA],
            'bo')
    pylab.plot([p[0] for p in classB],
            [p[1] for p in classB],
            'ro')
    pylab.show()

main()
