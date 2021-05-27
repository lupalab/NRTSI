import numpy as np
import pdb
import time
import random
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
lim = 0.8828
max_time = 200
N_train = 12000
N_test = 1000
data = np.zeros((N_train, max_time, 3))
random.seed(0)
np.random.seed(0)
# use seed 0 to create train
# use seed 1 to create test
def f(x, y, dx, dy, lim=0.8828):
    #theta = theta % (2*np.math.pi)
    theta = np.arctan(np.abs(dy/dx))
    if dy > 0 and dx > 0:
        theta = theta
    elif dy > 0 and dx < 0:
        theta = np.math.pi - theta
    elif dy < 0 and dx < 0:
        theta = np.math.pi + theta
    else:
        theta = 2 * np.math.pi - theta
    
    A = np.math.sin(theta)
    B = - np.math.cos(theta)
    C = np.math.sin(theta) * x - np.math.cos(theta) * y
    intersect = []
    dist = []
    if theta < np.math.pi / 2:
        candidates =  [(1,0,lim),(0, 1, lim)]
    elif theta < np.math.pi:
        candidates =  [(-1,0,lim),(0, 1, lim)]
    elif theta < np.math.pi * 3 / 2:
        candidates =  [(-1,0,lim),(0, -1, lim)]
    else:
        candidates =  [(1,0,lim),(0, -1, lim)]
    for a,b,c in candidates:
        aa = np.array([[A, B], [a, b]])
        bb = np.array([C, c])
        xx = np.linalg.solve(aa, bb)
        intersect.append(xx)
        dist.append(np.sqrt((xx[0] - x) ** 2 + (xx[1] - y) ** 2))
    next_x, next_y = intersect[np.argmin(dist)]
    if np.argmin(dist) == 0:
        next_dx = - dx
        next_dy = dy
    else:
        next_dx = dx
        next_dy = - dy

    plt.plot((x,next_x),(y, next_y))
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])
    return next_x, next_y, next_dx, next_dy

for j in tqdm(range(N_train)):
    x0 = random.uniform(-lim, lim)
    y0 = random.uniform(-lim, lim)
    theta0 = random.uniform(0, 2 * np.math.pi)
    v = random.uniform(0.0018, 0.1075)
    dx = np.cos(theta0)
    dy = np.sin(theta0)
    #x0 = 0
    #y0 = 0
    #theta0 = 3.14

    plt.figure()
    t = 0
    inter = [(x0, y0, t)]
    while t < max_time:
        old_x0, old_y0, old_dx, old_dy = x0, y0, dx, dy

        x0, y0, dx, dy = f(x0, y0, dx, dy, lim)
        t += np.sqrt((x0 - old_x0) ** 2 + (y0 - old_y0) ** 2) / v
        inter.append((x0, y0, t))
    irr_time = [random.uniform(0, max_time) for i in range(max_time)]
    for i, t in enumerate(irr_time):
        k = 0
        while k < len(inter):
            if t > inter[k][-1]:
                k += 1
            else:
                k -= 1
                break
        cur_x = inter[k][0] + (inter[k+1][0] - inter[k][0]) * (t - inter[k][2]) / (inter[k+1][2] - inter[k][2])
        cur_y = inter[k][1] + (inter[k+1][1] - inter[k][1]) * (t - inter[k][2]) / (inter[k+1][2] - inter[k][2])
        data[j, i, 0] = cur_x
        data[j, i, 1] = cur_y
        data[j, i, 2] = t
    if False:
        plt.scatter(data[j, :,0], data[j, :,1])
        plt.savefig('./data/irr_billiard/test_%d.png' % j)
        plt.close()
    
np.save('./data/irr_billiard_train_large.npy', data)

data = np.zeros((N_test, max_time, 3))
for j in tqdm(range(N_test)):
    x0 = random.uniform(-lim, lim)
    y0 = random.uniform(-lim, lim)
    theta0 = random.uniform(0, 2 * np.math.pi)
    v = random.uniform(0.0018, 0.1075)
    dx = np.cos(theta0)
    dy = np.sin(theta0)
    #x0 = 0
    #y0 = 0
    #theta0 = 3.14

    plt.figure()
    t = 0
    inter = [(x0, y0, t)]
    while t < max_time:
        old_x0, old_y0, old_dx, old_dy = x0, y0, dx, dy

        x0, y0, dx, dy = f(x0, y0, dx, dy, lim)
        t += np.sqrt((x0 - old_x0) ** 2 + (y0 - old_y0) ** 2) / v
        inter.append((x0, y0, t))
    irr_time = [random.uniform(0, max_time) for i in range(max_time)]
    for i, t in enumerate(irr_time):
        k = 0
        while k < len(inter):
            if t > inter[k][-1]:
                k += 1
            else:
                k -= 1
                break
        cur_x = inter[k][0] + (inter[k+1][0] - inter[k][0]) * (t - inter[k][2]) / (inter[k+1][2] - inter[k][2])
        cur_y = inter[k][1] + (inter[k+1][1] - inter[k][1]) * (t - inter[k][2]) / (inter[k+1][2] - inter[k][2])
        data[j, i, 0] = cur_x
        data[j, i, 1] = cur_y
        data[j, i, 2] = t
    if False:
        plt.scatter(data[j, :,0], data[j, :,1])
        plt.savefig('./data/irr_billiard/test_%d.png' % j)
        plt.close()
    
np.save('./data/irr_billiard_test.npy', data)
    


