import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import histogram
import pprint

n = int(input('Количество точек'))
a = 1
alpha = .2
d = .45
r = 0.9
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

x1_step = []
x2_step = []
x3_step = []


t1 = 0
t2 = 0
t3 = 0
for i in range(n - 1):

    x1[i + 1] = f1(x1[i], x2[i], x3[i])
    x2[i + 1] = f2(x1[i], x2[i], x3[i])
    x3[i + 1] = f3(x1[i], x2[i], x3[i])
    if x1[i + 1] > a:
        x1_start[i + 1] = 1
        x1_step.append((i+1) - t1)
        t1 = i + 1
    if x2[i + 1] > a:
        x2_start[i + 1] = 1
        x2_step.append((i + 1) - t2)
        t2 = i + 1
    if x3[i + 1] > a:
        x3_start[i + 1] = 1
        x3_step.append((i + 1) - t3)
        t3 = i + 1
    step[i] = i
step[n - 1] = n

print(x3_step)
'''
for i in range(n):
    if x1_start[i] == 1:

        plt.plot([i, i], [0, 1], linestyle = '-', color = 'red')
plt.axis([100, n, 0, 7])
plt.show()
tau = np.histogram(x3_step)
print('max  ', min(x2_step))
print(tau)
plt.hist(x1_step,1000,  color = 'black')
plt.xlabel('interval')
plt.show()
plt.hist(x2_step,1000,  color = 'black')
plt.xlabel('interval')
plt.show()
'''
def apr(x):
    a1 = 1068
    b1 = 27.42
    c1 = 10.25
    a2 = 446.1
    b2 = 21.41
    c2 = 115.2
    return a1*np.exp(-((x - b1)/c1)**2) + a2*np.exp(-((x-b2)/c2)**2)
ap = np.zeros(300-11)
ax_st = np.zeros(300-11)
for i in range(11 ,300, 1):
    ap[i-11] = apr(i)
    ax_st[i-11] = i
l = plt.hist(x3_step,'auto',  color = 'black', alpha = .8)
pprint.pprint(l)
file_1 = open("snake.txt", "w")


file_1.write(str(l[0]))
file_1.close()
plt.xlabel('interval')

plt.plot(ax_st, ap, linestyle = '--', color = 'red', alpha = .9)
plt.show()

