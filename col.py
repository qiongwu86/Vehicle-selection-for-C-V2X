import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import math
import copy
import scipy.stats as stats



C_max = 50  # 车辆数
T_max = 1200
Q_max = 2000
L = 20
d_sen = 500
S = 2
a = 0
b = 0
c = 0
d = 0
e = 0
N_bn = 0
N_oa = 0
N_A = 0
N_D = 0
N_ccr = 0


file = open('distance_data_50', 'r')
d = file.read().split()
distance_1 = list(float(i) for i in d)
distance_2 = []
for i in range(len(distance_1)):
    distance_2.append(np.array(distance_1[i]))
dhk = np.array(distance_2) 


if L==100:
    f = 10
    Rl = 5
    Rh = 15
elif L==50:
    f = 20
    Rl = 10
    Rh = 30
elif L==20:
    f = 50
    Rl = 25
    Rh = 75

C_priority = np.zeros(C_max)
C_fairness_random = np.zeros(C_max)
C_fairness_proposed = np.zeros(C_max)
T_queue_proposed = np.zeros(T_max)
T_queue_random = np.zeros(T_max)
T_compare_queue1 = np.zeros(T_max)
T_compare_queue2 = np.zeros(T_max)
T_client_choice_proposed = np.zeros(T_max)
T_client_choice_random = np.zeros(T_max)
T_accuracy_proposed = np.zeros(T_max)
T_accuracy_random = np.zeros(T_max)
T_departure = np.zeros(T_max)
T_alive_client_proposed = np.zeros(T_max)
T_alive_client_random = np.zeros(T_max)
s_star_proposed = []
s_star_random = []
s_star_static = []
Total_Data_Proposed = 0
Total_Data_Random = 0
copy_T_queue_proposed = []
copy_T_queue_random = []
epsilon_1 = np.zeros(C_max)
a_E_1 = np.zeros(C_max)
r_1 = np.zeros(C_max)
Delay = np.zeros(C_max)
d_kr = np.zeros(C_max)
d_ir = np.zeros(C_max)
d_ki = np.zeros(C_max)
K_S = np.zeros(C_max)
d_1 = np.zeros(C_max)
d_2 = np.zeros(C_max)
d_2 = np.zeros(C_max)
K_C = np.zeros(C_max-1)
P_same = np.zeros(C_max-1)
C_n = [10 for i in range(C_max)]
C_distance = np.zeros(C_max)


fo = open('int_50_50_23_2.txt', 'r')



P_col = np.zeros(C_max)
for i in range(C_max):
    d_kr[i] = abs(500 - dhk[i])
for j in range(C_max): 
    d_ir = copy.deepcopy(d_kr)   
    d_ir[j] = 0 
    d_ki = abs(dhk-dhk[j])
    d_ki = np.delete(d_ki, j)
    N_total = (1000*S)/f
    N_lc = 0.2*N_total
    P_rc0 = 1/(Rh-Rl)
    a = np.sum(d_ki<=d_sen)
    K_S[j] = a
    for k in range(C_max-1):
        d_1 = copy.deepcopy(dhk)
        d_1 = np.delete(d_1, j)
        d_2 = copy.deepcopy(dhk)
        b = d_2[j] + d_sen
        c = d_1[k] + d_sen
        d = d_2[j] - d_sen
        e = d_1[k] - d_sen
        if b > 1000:
            b = 1000
        if d < 0:
            d = 0
        if c > 1000:
            c = 1000
        if e < 0:
            e = 0
        g = 0    
        for l in range(C_max-2):
            d_3 = copy.deepcopy(d_1)
            d_3 = np.delete(d_3, k)
            if max(d,e)<d_3[l]<min(b,c):
                g = g + 1
            K_C[k] = g
    for m in range(C_max-1):
        N_bn = N_total*(1-(1-(1/N_total))**K_C[m])
        N_oa = N_total*(1-(1-(1/N_total))**K_S[j])
        N_A = N_oa - N_bn
        print(N_A)
        N_D = N_total - N_bn
        if N_D >= 1:
            N_ccr = (N_D-N_A)*(1-(1/N_D))**N_A
            print(N_ccr)
        else:
            N_ccr = 0
        if d_ki[m]<=d_sen:
            h = (P_rc0*N_ccr)/(N_lc*N_lc)
            P_same[m] = h
        elif d_ki[m]>d_sen:
            h = N_ccr/(N_lc*N_lc)
            P_same[m] = h     
    P_int = []
    line = fo.readline()
    P_int_1 = []
    P_int_2 = line.split(',')
    for x in P_int_2:
        newx = float(x)
        P_int_1.append(newx)
    P_int_1 = np.array(P_int_1)
    P_int = np.array(P_int_1)
    col = np.zeros(C_max)
    col = P_same*P_int
    prod = 0
    prod = 1 - np.prod(1-col)
    P_col[j] = prod
# print(P_col)