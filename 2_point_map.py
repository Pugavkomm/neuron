import random
import numpy as np
import matplotlib.pyplot as plt
import pylab

n = 1000
a = 1
alpha = .2

d = .45
r = 1
b =4.95
x1 = np.zeros(n)
x2 = np.zeros(n)
x3 = np.zeros(n)
step = np.zeros(n)
x1[0] = 1.3
x2[0] = 0.3
x3[0] = .6
x1_start = np.zeros(n)
x2_start = np.zeros(n)
x3_start = np.zeros(n)


def F(x, quast = 0):
    if (x <= a):
        return alpha * x
    if (x > a):
        return alpha * x + alpha * (b - a)
    if quast == 2:
        if (x<=a):
            return (alpha  - .1)* x
        if (x > a):
            return (alpha  - .1)* x + (alpha - .1) * (b - a)





def f1(x1, x2, x3):
    return (F(x1) + d * (x2 + x3 - 2 * x1))


def f2(x1, x2, x3):
    return F(x2) + d * (x1 - x2 + r * (x3 - x2))


def f3(x1, x2, x3):
    return F(x3, quast = 3) + d * (x1 - x3 + r * (x2 - x3))



for i in range(n - 1):

    x1[i + 1] = f1(x1[i], x2[i], x3[i])
    x2[i + 1] = f2(x1[i], x2[i], x3[i])
    x3[i + 1] = f3(x1[i], x2[i], x3[i])
    if x1[i + 1] > a:
        x1_start[i + 1] = 1
    if x2[i + 1] > a:
        x2_start[i + 1] = 1
    if x3[i + 1] > a:
        x3_start[i + 1] = 1
    step[i] = i
step[n - 1] = n
def per(x):
    t  = x
    for i in range(1, n):
        if i%2 != 0:
            print(t[i])
            t[i] = 0
    return t
x1 = per(x1)
x2 = per(x2)
x3 = per(x3)
print('это: \n ',x1, '\n', x2, '\n', x3)
'''
plt.plot(step, x1, color = 'red')
plt.show()
plt.plot(step, x2, color = 'red')
plt.show()
plt.plot(step, x3, color = 'red')
plt.show()'''

pylab.subplot(2, 2, 1)
pylab.plot(step, x1, color='#008080')
pylab.plot(step, x1, ".")
pylab.plot([0, n], [a, a], '--', color='black')
pylab.plot([0, n], [0,0], '--', color='black')
pylab.title("x1")
pylab.axis([0, n, min(x1) - .1, max(x1) + .1])

pylab.subplot(2, 2, 3)
pylab.plot(step, x2, color='#008080')
pylab.plot(step, x2, ".")

pylab.plot([0, n], [a, a], '--', color='black')
pylab.plot([0, n], [0,0], '--', color='black')
pylab.title("x2")
pylab.axis([0, n, min(x2) - .1, max(x2) + .1])

pylab.subplot(2, 2, 4)
pylab.plot(step, x3, color='#008080')
pylab.plot(step, x3, ".")
pylab.plot([0, n], [a, a], '--', color='black')
pylab.plot([0, n], [0,0], '--', color='black')
pylab.title("x3")
pylab.axis([0, n, min(x3) - .1, max(x3) + .1])

pylab.show()
plt.grid(True)

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
''' ''''''
'''
plt.plot(step, x1_start, linestyle='-', color='black', label="x1")
plt.plot(step, x2_start, linestyle='-', color='red', label="x2")
plt.plot(step, x3_start, linestyle='-', color='blue', label="x3")
plt.grid(True)
plt.axis([100, n, 0, 7])
plt.legend()
plt.show()
'''
two = [0, 1]


def plot_end(x, col, n):
    if x == a:
        plt.plot([n, n],[0, 1], color=col, linestyle = '-')


for i in range(n):
    plot_end(x1_start[i], 'green', i)
    plot_end(x2_start[i], 'black', i)
    plot_end(x3_start[i], 'blue', i)
    if i == n - 1:
        plt.plot([i, i], [0, 0], color = 'green', label = 'x1')
        plt.plot([i, i], [0, 0], color = 'black', label = 'x2')
        plt.plot([i, i], [0, 0], color='blue', label='x3')
plt.axis([100, n, 0, 7])
plt.legend()
plt.show()
print('x1:', x1[:19000], '\n x2: ', x2[:19000], '\n x3: ', x3[:19000])