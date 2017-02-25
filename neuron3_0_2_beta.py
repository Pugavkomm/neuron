import random
import numpy as np
import matplotlib.pyplot as plt
import pylab

n = int(input('Количество точек'))
a = 1
alpha = .2
d = .45
r = .9
b = 4.95

x1 = np.zeros(n)
x2 = np.zeros(n)
x3 = np.zeros(n)
step = np.zeros(n)
x1[0] = .4

x2[0] = .45
x3[0] = .47
x1_start = np.zeros(n)
x2_start = np.zeros(n)
x3_start = np.zeros(n)


def F(x):
    if (x <= a):
        return alpha * x
    if (x > a):
        return alpha * x + alpha * (b - a)


print(F(1))


def f1(x1, x2, x3):
    return F(x1) + d * (x2 + x3 - 2 * x1)


def f2(x1, x2, x3):
    return F(x2) + d * (x1 - x2 + r * (x3 - x2))


def f3(x1, x2, x3):
    return F(x3) + d * (x1 - x3 + r * (x2 - x3))


for i in range(n - 1):

    x1[i + 1] = f1(x1[i], x2[i], x3[i])
    x2[i + 1] = f2(x1[i], x2[i], x3[i])
    x3[i + 1] = f3(x1[i], x2[i], x3[i])
    if x1[i + 1] > a:
        x1_start[i + 1] = 2
    if x2[i + 1] > a:
        x2_start[i + 1] = 3
    if x3[i + 1] > a:
        x3_start[i + 1] = 4
    step[i] = i
step[n - 1] = n
'''
plt.plot(step, x1, color = 'red')
plt.show()
plt.plot(step, x2, color = 'red')
plt.show()
plt.plot(step, x3, color = 'red')
plt.show()'''


pylab.subplot(2, 2, 1)
pylab.plot(step, x1, color = '#008080')
pylab.title("x1")
pylab.axis([100, n, -3, 3])

pylab.subplot(2, 2, 3)
pylab.plot(step, x2, color = '#008080')
pylab.title("x2")
pylab.axis([100, n, -3, 3])

pylab.subplot(2, 2, 4)
pylab.plot(step, x3, color = '#008080')
pylab.title("x3")
pylab.axis([100, n, -3, 3])

pylab.show()
plt.grid(True)
plt.plot(step, x2)
plt.show()


'''
def plot(step, xn, i):
    plt.plot(step, xn, color = '#008080')
    plt.title('x' + str(i))
    plt.grid(True)
    plt.axis([100, n, -3, 3])
    plt.xlabel('n')
    plt.show()


plot(step, x1, 1)
plot(step, x2, 2)
plot(step, x3, 3)
'''


plt.plot(step, x1_start, '.', color='green', label="x1")
plt.plot(step, x2_start, '.', color='red', label="x2")
plt.plot(step, x3_start, '.', color='blue', label="x3")
plt.grid(True)
plt.axis([100, n, 1, 7])
plt.legend()
plt.show()
