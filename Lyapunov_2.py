import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import math
import copy
import scipy.stats as stats



C_max = 100  # 车辆数
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


# Data
datasize = 1  # MegaByte 平均数据大小
dataPerClient = 1500  # 总训练数据100000 每个客户接受1500
datachunk = 10  # 单位时间传输10

static_selection_client_number = 4  # static选择方法

V = 10 ** 9

# Client number data amount
CN_data_amount = np.arange(C_max + 1)  
CN_data_amount = CN_data_amount * datachunk

CN_accuracy = np.arange(Q_max + 1, dtype='f')
l_rate = 100
d_rate = -0.3
for i in range(1, Q_max + 1):
    CN_accuracy[i] = 100 - l_rate * (pow(CN_accuracy[i], d_rate))


# Data
C_data_proposed = np.zeros(C_max)
C_data_proposed = np.full(C_max, dataPerClient, dtype=int)
C_data_random = C_data_proposed.copy()


# Survivability
file = open('distance_data_100', 'r')
d = file.read().split()
distance_1 = list(float(i) for i in d)
distance_2 = []
for i in range(len(distance_1)):
    distance_2.append(np.array(distance_1[i]))
dhk = np.array(distance_2) 
D = [1000 for i in range(C_max)]
Uhk = 10
C_S = ((D - dhk) / Uhk)
C_S_random = C_S.copy()
C_S_static = C_S.copy()


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


fo = open('int_100_50_23_2.txt', 'r')

for t in range(T_max):

    departure = random.random()
    if departure < 0.95:
        departure = 10 * datachunk * random.random()
    else:
        departure = 0

# delay
    for y in range(C_max):
        sigma_2 = 7.5
        C_distance[y] = abs(500 - dhk[y])
        epsilon_1[y] = 1.238 * (10 ** (-6)) * (C_distance[y] * C_distance[y]) + (-0.001807) * C_distance[y] + 1.002
        a_E_1[y] = (epsilon_1[y] ** (1 / C_distance[y]))
        r_1[y] = sigma_2 * a_E_1[y]
        Delay[y] = C_n[y]/r_1[y]

# collision
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
            N_D = N_total - N_bn
            if N_D >= 1:
                N_ccr = (N_D-N_A)*(1-(1/N_D))**N_A
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



    # Proposed Number Decision
    if t == 0:
        max_choice_client_Proposed = C_max
        max_choice_data_Proposed = max_choice_client_Proposed * datachunk

        max_choice_client_Random = C_max
        max_choice_data_Random = max_choice_client_Random * datachunk

        T_queue_proposed[t] = 0
        T_queue_random[t] = 0
        T_compare_queue1[t] = 0
        T_compare_queue2[t] = 0
    else:
        val_proposed = -9999999999
        val_temp_proposed = val_proposed - 1
        for i in range(C_max + 1):
            if 0 < (T_queue_proposed[t - 1] + CN_data_amount[i]) and (
                    T_queue_proposed[t - 1] + CN_data_amount[i]) < Q_max + 1:
                val_temp_proposed = V * CN_accuracy[(int)(T_queue_proposed[t - 1] + CN_data_amount[i])] + T_queue_proposed[t - 1] * CN_data_amount[i]
                if val_temp_proposed > val_proposed:
                    val_proposed = val_temp_proposed
                    max_choice_client_Proposed = i
        if T_queue_proposed[t - 1] == 0 and (t - 1) != 0:
            copy_T_queue_proposed.append(t - 1)
        # for a in range(len(T_queue_proposed)):
        #     copy_T_queue_proposed[a] = T_queue_proposed[t-1]

        val_random = -9999999999
        val_temp_random = val_random - 1
        for i in range(C_max + 1):
            if 0 < (T_queue_random[t - 1] + CN_data_amount[i]) and (
                    T_queue_random[t - 1] + CN_data_amount[i]) < Q_max + 1:
                val_temp_random = V * CN_accuracy[(int)(T_queue_random[t - 1] + CN_data_amount[i])] + T_queue_random[t - 1] * CN_data_amount[i]
                if val_temp_random > val_random:
                    val_random = val_temp_random
                    max_choice_client_Random = i

        # num_alive_client = sum(x[0] > 0 and x[1] > 0 for x in enumerate(zip(C_power, C_data)))
        # for a in range(t):
        #     if T_queue_random[t-1] == 0:
        #         g = t-1
        #     if g != 0:
        #         print(g)
        #     break;
        # for b in range(t):
        #     copy_T_queue_random[a] = T_queue_random[t-1]
        if T_queue_random[t - 1] == 0 and (t - 1) != 0:
            copy_T_queue_random.append(t - 1)

        # Proposed Selection
        num_alive_client_Proposed = 0
        for x in zip(C_S, C_data_proposed):
            if x[0] > 0 and x[1] > 0:
                num_alive_client_Proposed += 1
        max_choice_client_Proposed = min(max_choice_client_Proposed, num_alive_client_Proposed)
        alive_client_Proposed = np.argwhere((C_S > 0) & (C_data_proposed > 0))
        alive_client_Proposed = alive_client_Proposed.reshape(num_alive_client_Proposed)

        for i in range(C_max):
            if (C_S[i] <= 0) or (C_data_proposed[i] <= 0) or (P_col[i] == 0) or (Delay[i] ==0):
                C_priority[i] = 0
            else:
                C_priority[i] = C_data_proposed[i] / (Delay[i]*P_col[i]*C_S[i])
        client_choice_Proposed = C_priority.argsort()[::-1][:max_choice_client_Proposed]


        for i in client_choice_Proposed:
            C_data_proposed[i] -= datachunk
            C_fairness_proposed[i] += 1
            
 
        for i in client_choice_Proposed:
            pd = random.random()
            if pd <= (P_col[i]+0.15):
                client_choice_Proposed = np.delete(client_choice_Proposed, np.where(client_choice_Proposed==i))
                max_choice_client_Proposed = max_choice_client_Proposed - 1
            else:
                Total_Data_Proposed += datachunk    

        s_star_proposed.append(max_choice_client_Proposed)

        # Random Select
        num_alive_client_Random = 0
        for x in zip(C_S_random, C_data_random):
            if x[0] > 0 and x[1] > 0:
                num_alive_client_Random += 1
        max_choice_client_Random = min(max_choice_client_Random, num_alive_client_Random)
        alive_client_Random = np.argwhere((C_S_random > 0) & (C_data_random > 0))
        alive_client_Random = alive_client_Random.reshape(num_alive_client_Random)
        client_choice_Random = random.sample(list(alive_client_Random), max_choice_client_Random)


        for i in client_choice_Random:
            C_data_random[i] -= datachunk
            C_fairness_random[i] += 1


        for i in client_choice_Random:
            pd = random.random()
            if pd <= (P_col[i]+0.15):
                client_choice_Random = np.delete(client_choice_Random, np.where(client_choice_Random==i))
                max_choice_client_Random = max_choice_client_Random - 1
            else:
                Total_Data_Random += datachunk

        s_star_random.append(max_choice_client_Random)


        # Static Select
        num_alive_client_Static = 0
        for x in C_S_static:
            if x > 0:
                num_alive_client_Static += 1
        static_selection_client_number = min(static_selection_client_number, num_alive_client_Static)
        s_star_static.append(static_selection_client_number)
        for i in range(C_max):
            C_S[i] -= 0.1
            C_S_random[i] -= 0.1
            C_S_static[i] -= 0.1
            dhk[i] -= 1

        max_choice_data_Proposed = max_choice_client_Proposed * datachunk
        max_choice_data_Random = max_choice_client_Random * datachunk

        T_queue_proposed[t] = max(T_queue_proposed[t - 1] + max_choice_data_Proposed - departure, 0)
        T_queue_random[t] = max(T_queue_random[t - 1] + max_choice_data_Random - departure, 0)
        T_compare_queue1[t] = T_compare_queue1[t - 1] + (C_max * datachunk) - departure
        T_compare_queue2[t] = max(T_compare_queue2[t - 1] + static_selection_client_number * datachunk - departure, 0)
        T_client_choice_proposed[t] = max_choice_client_Proposed
        T_client_choice_random[t] = max_choice_client_Random
        T_accuracy_proposed[t] = CN_accuracy[(int)(T_queue_proposed[t])]
        T_accuracy_random[t] = CN_accuracy[(int)(T_queue_random[t])]
        T_departure[t] = departure
        T_alive_client_proposed[t] = num_alive_client_Proposed
        T_alive_client_random[t] = num_alive_client_Random


f1 = open('Proposed_power.csv', 'w')
wr = csv.writer(f1)
wr.writerow(T_queue_proposed)

f2 = open('Random_power.csv', 'w')
wr = csv.writer(f2)
wr.writerow(T_queue_random)

f3 = open('Full_power.csv', 'w')
wr = csv.writer(f3)
wr.writerow(T_compare_queue1)

f4 = open('Static_power.csv', 'w')
wr = csv.writer(f4)
wr.writerow(T_compare_queue2)

f5 = open('Fairness_proposed.csv', 'w')
wr = csv.writer(f5)
wr.writerow(C_fairness_proposed)

f6 = open('Fairness_random.csv', 'w')
wr = csv.writer(f6)
wr.writerow(C_fairness_random)

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()

interval = 3


g = copy_T_queue_proposed[0]
h = copy_T_queue_random[0]
print(len(s_star_proposed))
print(len(s_star_random))
print(h)
print(g)


name = '100_50_23_2/' + 'three'
np.savez(name, T_queue_proposed)
# name = 'data/' + 'two'
# np.savez(name, CN_accuracy, T_queue_proposed, T_queue_random, T_compare_queue1, T_compare_queue2,
#          Total_Data_Proposed, Total_Data_Random, s_star_proposed, s_star_random)

# plt.legend()
print("Total Data : ", dataPerClient * C_max)
print("Total_Data_Proposed : ", Total_Data_Proposed)
print("Total_Data_Random : ", Total_Data_Random)


def return_s_star_proposed():
    return s_star_proposed, g


def return_s_star_random():
    return s_star_random, h


plt.figure(2)
plt.axhline(y=Q_max, color='k', linestyle='--', linewidth=3.0)
plt.ylim(0, Q_max * 5)
plt.xlim(0, T_max)

# # print(T_queue_proposed)
line1, = plt.plot(np.arange(len(T_queue_proposed)), T_queue_proposed[:], label='Proposed')
line2, = plt.plot(np.arange(len(T_queue_random)), T_queue_random[:], label='Random')
line3, = plt.plot(np.arange(len(T_compare_queue1)), T_compare_queue1[:], label='full')
line4, = plt.plot(np.arange(len(T_compare_queue2)), T_compare_queue2[:], label='static')

plt.setp(line1, color='magenta', linewidth=3.0)
# plt.setp(line2, color='b', linewidth=1.5)
# plt.setp(line3, color='red', linewidth=3.0)
# plt.setp(line4, color='lime', linewidth=3.0)
plt.legend(handles=(line1, line2, line3, line4), labels=('Control Algorithm + Weight Selection', 'Control Algorithm + Random Selection', 'Full Selection', 'Static Selection'), prop={'size':30})
plt.xlabel('Time Slot (sec)', fontsize=20)
plt.ylabel('Queue Backlog (MB)', fontsize=20)
plt.grid(True)


plt.show()
