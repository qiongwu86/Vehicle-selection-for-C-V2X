# coding: utf-8
import tensorflow as tf
import os
import numpy as np
import ipdb as pdb
import math
import random
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  # 导入FontProperties
from tensorflow.examples.tutorials.mnist import input_data


C_max = 100
T_max = 1200
Q_max = 2000
T_compare_queue2 = np.zeros(T_max)

file = open('distance_data_100', 'r')
d = file.read().split()
distance_1 = list(float(i) for i in d)
distance_2 = []
for i in range(len(distance_1)):
    distance_2.append(np.array(distance_1[i]))
dhk = np.array(distance_2) 



for t in range(T_max):
    departure = random.random()
    if departure < 0.95:
        departure = 10 * 10 * random.random()
    else:
        departure = 0


    D = [500 for i in range(C_max)]
    Position = abs(D - dhk)

    min_10 = sorted(Position)[:10]
    min_10_number = [i for i, num in enumerate(Position) if num in min_10]

    min_80 = sorted(Position)[:80]
    min_80_number = [i for i, num in enumerate(Position) if num in min_80]
    print(Position)
    print("距离RSU最近的80个车辆：", min_80_number)
    print("距离RSU最近的10个车辆：", min_10_number)


    # 创建一个存储删除索引的列表
    indices_to_delete = []

    for i in range(C_max-1):
        dhk[i] += 1
        if dhk[i] >= 1000:
            indices_to_delete.append(i)

    # 根据索引列表删除元素
    dhk = np.delete(dhk, indices_to_delete)
    C_max = C_max - len(indices_to_delete)
    print(len(min_10_number))
    if t == 0:
        T_compare_queue2[t] = 0
    else:
        T_compare_queue2[t] = max(T_compare_queue2[t - 1] + len(min_10_number) * 10 - departure, 0)
