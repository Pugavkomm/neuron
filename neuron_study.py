import random
import numpy as np
import matplotlib.pyplot as plt

# задаём параметры: a, alpha, d. Задаём колиество итераций quantity
a = 1
b = 4.95  # float(input(('b ')))
alpha = .2  # float(input('alpha '))
d = .72  # float(input('d '))
quantity = int(input('количество итераций(точек): '))  # Quantity points

for l in range(5):
    y1 = np.zeros(quantity)
    y2 = np.zeros(quantity)
    # y1[0] = float(input('y1'))                      #start point y1
    # y2[0] = float(input('y2'))                      #start point y2
    # y2[0] = 0.1*(l+1)
    # y1[0] = 0.02*(3*l + 2)
    y1[0] = random.random()
    y2[0] = random.random()
    # loop and condition
    for i in range(quantity - 1):
        if y2[i] > 2 * a and y1[i] < y2[i] - 2 * a and y1[i] > 2 * a - y2[i]:
            b1 = 0
            b2 = 2 * alpha * (b - a)

        elif y2[i] < 2 * a and y1[i] > y2[i] - 2 * a and y1[i] < 2 * a - y2[i]:
            b1 = 0
            b2 = 0
        elif y1[i] > 0 and y2[i] > 2 * a - y1[i] and y2[i] < 2 * a + y1[i]:
            b1 = alpha * (b - a)
            b2 = alpha * (b - a)
        elif y1[i] < 0 and y2[i] < 2 * a - y1[i] and y2[i] > 2 * a + y1[i]:
            b1 = -alpha * (b - a)
            b2 = alpha * (b - a)
        y1[i + 1] = y1[i] * (alpha - 2 * d) + b1
        y2[i + 1] = y2[i] * alpha + b2

    print(y1, '\n', y2)  # display a points
    plt.plot(y1, y2)  # plot plane y1, y2
    plt.plot(y1[0], y2[0], '.', color = 'red')
    # plt.axis([-3, 3, 0, .7])
    plt.ylabel('y2')
    plt.xlabel('y1')
    plt.title('y1=' + str(y1[0]) + '  y2 =' + str(y2[0]) + ' b =' + str(b) + ' d=' + str(d) + ' alpha=' + str(
        alpha) + ' a=' + str(a))
    plt.text(1, .1, str(quantity))


