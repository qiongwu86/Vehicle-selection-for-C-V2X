    # coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import ipdb as pdb
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  # 导入FontProperties
# import torch.nn as nn
from Lyapunov_2 import *

np.set_printoptions(threshold=np.inf)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
def output_avg(dir):
    dir_path = dir
    fileList = os.listdir(dir_path) 
    fileList = [name for name in fileList if '.npz' in name]
    avg_rs_1 = []
    for name in fileList:
        path = dir_path + name
        res = np.load(path)
        temp_rs_1 = np.array(res['arr_0'])
        avg_rs_1.append(temp_rs_1)
    avg_rs_1 = np.mean(avg_rs_1, axis=0, keepdims=True)[0]
    # avg_rs_1 = moving_average(np.mean(avg_rs_1, axis=0, keepdims=True)[0],10)
    return avg_rs_1

if __name__ == '__main__':
    # 读入数据
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # x为训练图像的占位符、y_为训练图像标签的占位符
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # 将单张图片从784维向量重新还原为28x28的矩阵图片
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 第一层卷积层
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积层
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层，输出为1024维的向量
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # 使用Dropout，keep_prob是一个占位符，训练时为0.5，测试时为1
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 把1024维的向量转换成10维，对应10个类别
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

   
    # 我们不采用先Softmax再计算交叉熵的方法，而是直接用tf.nn.softmax_cross_entropy_with_logits直接计算
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    # 同样定义train_step
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # 定义测试的准确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #用训练好的模型进行训练
    # res_path = 'localmodel_1/'     
    # Local_model_1 = output_avg(res_path)


    # 创建Session和变量初始化
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # W_fc2 = Local_model_1
    # 训练100次每次200步
    Accuracy_1 = []
    Loss_1 = []
    s_star,stop_proposed = return_s_star_proposed()
    print(stop_proposed)
    for j in range(len(s_star)):
        if s_star[j] > 0:
            for i in range(s_star[j]):
                batch = mnist.train.next_batch(1000)
                if i % 1 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0})
                    print("number %d, training accuracy %g" % (i+1, train_accuracy))
                _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                # if i == 0:
                #     W_glob = sess.run(W_fc2)
                # W_glob = sess.run(W_fc2) + W_glob
            # # 训练结束后报告在测试集上的准确度
            #     W_glob = W_glob / s_star[j]
                if i + 1 == s_star[j]:
                    # total_accuracy = train_accuracy
                    print("step %d, train_accuracy %g" % (j+1, train_accuracy))
                    print("loss %g" % (loss))
                    Accuracy_1.append(train_accuracy)
                    Loss_1.append(loss)
  
        # print("total accuracy %g" % accuracy.eval(feed_dict={
        #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
            # print(W_glob)
        # else:
        #     print("The number of training is not reached!")
        if s_star[j] == 0:
            print("step %d, train_accuracy %g" % (j+1, train_accuracy))
            print("loss %g" % (loss))
            Accuracy_1.append(train_accuracy)
            Loss_1.append(loss)
        if j == stop_proposed:
            W_glob_1 = sess.run(W_fc2)
            break


    # np.savez('train_K50_S4_L100_6.npz', d_1=Accuracy_1, d_2=Loss_1)
    name = 'traindata_1_1/' + 'two'
    np.savez(name, Accuracy_1, Loss_1)
    # name = 'Train_data1/' + 'train_one_0'
    # np.savez(name, Accuracy_1, Loss_1)

 
    name = 'localmodel_1/' + 'two'
    np.savez(name, W_glob_1)

# plt.figure(6)
# plt.xlim(-100, 1700)
# plt.xlabel('Total Number of Client(Control Algorithm + Weight Selection)')
# plt.ylabel('Training Accuracy')
# client_number = np.arange(len(Accuracy_1))
# plt.plot(client_number, Accuracy_1, color='b', linewidth=1)
# plt.axhline(y=1, color='k', linestyle='--', linewidth=1)

# plt.figure(7)
# plt.xlim(-100, 1700)
# plt.xlabel('Total Number of Client(Control Algorithm + Weight Selection)')
# plt.ylabel('Loss')
# client_number = np.arange(len(Loss_1))
# plt.plot(client_number, Loss_1, color='b', linewidth=1)
# plt.axhline(y=0, color='k', linestyle='--', linewidth=1)

# plt.show()

# 分割
np.set_printoptions(threshold=np.inf)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def output_avg_1(dir):
    dir_path = dir
    fileList = os.listdir(dir_path) 
    fileList = [name for name in fileList if '.npz' in name]
    avg_rs = []
    for name in fileList:
        path = dir_path + name
        res = np.load(path)
        # print(res)
        temp_rs = np.array(res['arr_0'])
        avg_rs.append(temp_rs)
        # print(avg_rs_1)
    avg_rs = np.mean(avg_rs, axis=0, keepdims=True)[0]
    # avg_rs_1 = moving_average(np.mean(avg_rs_1, axis=0, keepdims=True)[0],10)
    return avg_rs


if __name__ == '__main__':
    # 读入数据
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # x为训练图像的占位符、y_为训练图像标签的占位符
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # 将单张图片从784维向量重新还原为28x28的矩阵图片
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 第一层卷积层
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积层
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层，输出为1024维的向量
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # 使用Dropout，keep_prob是一个占位符，训练时为0.5，测试时为1
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 把1024维的向量转换成10维，对应10个类别
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


    # 我们不采用先Softmax再计算交叉熵的方法，而是直接用tf.nn.softmax_cross_entropy_with_logits直接计算
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    # 同样定义train_step
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # 定义测试的准确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # res_path = 'localmodel_2/'
    # Local_model = output_avg_1(res_path)

    # 创建Session和变量初始化
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # W_fc2 = Local_model
    # 训练100次每次200步
    Accuracy = []
    Loss = []
    s_star,stop_random = return_s_star_random()
    print(stop_random)
    for a in range(len(s_star)):
        if s_star[a] > 0:
            for b in range(s_star[a]):
                batch = mnist.train.next_batch(1000)
                if b % 1 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0})
                    print("number %d, training accuracy %g" % (b+1, train_accuracy))
                _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                # if i == 0:
                #     W_glob = sess.run(W_fc2)
                # W_glob = sess.run(W_fc2) + W_glob
            # # 训练结束后报告在测试集上的准确度
            #     W_glob = W_glob / s_star[j]
                if b + 1 == s_star[a]:
                    # total_accuracy = train_accuracy
                    # total_accuracy_0 = total_accuracy % s_star[a]
                    print("step %d, train_accuracy %g" % (a+1, train_accuracy))
                    print("loss %g" % (loss))
                    Accuracy.append(train_accuracy)
                    Loss.append(loss)
        # print("total accuracy %g" % accuracy.eval(feed_dict={
        #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
            # print(W_glob)
        if s_star[a] == 0:
            print("step %d, train_accuracy %g" % (a+1, train_accuracy))
            print("loss %g" % (loss))
            Accuracy.append(train_accuracy)
            Loss.append(loss)
        if a == stop_random:
            W_glob = sess.run(W_fc2)
            break

    name = 'traindata_1_2/' + 'two'
    np.savez(name, Accuracy, Loss)


    name = 'localmodel_2/' + 'two'
    np.savez(name, W_glob)

# plt.figure(8)
# plt.xlim(-100, 1700)
# plt.xlabel('Total Number of Client(Control Algorithm + Random Selection)')
# plt.ylabel('Training Accuracy')
# client_number = np.arange(len(Accuracy))
# plt.plot(client_number, Accuracy, color='red', linewidth=1)
# plt.axhline(y=1, color='k', linestyle='--', linewidth=1)

# plt.figure(9)
# plt.xlim(-100, 1700)
# plt.xlabel('Total Number of Client(Control Algorithm + Random Selection)')
# plt.ylabel('Loss')
# client_number = np.arange(len(Loss))
# plt.plot(client_number, Loss, color='red', linewidth=1)
# plt.axhline(y=0, color='k', linestyle='--', linewidth=1)

# plt.figure(1)
# plt.xlim(-100, 1600)
# plt.xlabel('Time Slot (50ms)')
# plt.ylabel('Training Accuracy')
# # plt.title('Control Algorithm + Random Selection', fontsize=20)
# client_number_1 = np.arange(len(Accuracy_1))
# client_number_2 = np.arange(len(Accuracy))
# line1, = plt.plot(client_number_1, Accuracy_1, color='b', linewidth=1.5)
# line2, = plt.plot(client_number_2, Accuracy, color='r', linewidth=1.5, linestyle=':')
# plt.setp(line1, color='b', linewidth=1.5)
# plt.setp(line2, color='r', linewidth=1.5)
# plt.legend(handles=(line1, line2), labels=('Control Algorithm + Weigth Selection', 'Control Algorithm + Random Selection'), prop={'size':10})
# plt.axhline(y=1, color='k', linestyle='--', linewidth=1)

# plt.figure(2)
# plt.xlim(-100, 1600)
# plt.xlabel('Time Slot (50ms)')
# plt.ylabel('Loss')
# # plt.title('Control Algorithm + Random Selection', fontsize=20)
# client_number_1 = np.arange(len(Loss_1))
# client_number_2 = np.arange(len(Loss))
# line1, = plt.plot(client_number_1, Loss_1, color='b', linewidth=1.5)
# line2, = plt.plot(client_number_2, Loss, color='r', linewidth=1.5, linestyle=':')
# plt.setp(line1, color='b', linewidth=1.5)
# plt.setp(line2, color='r', linewidth=1.5)
# plt.legend(handles=(line1, line2), labels=('Control Algorithm + Weigth Selection', 'Control Algorithm + Random Selection'), prop={'size':10})
# plt.axhline(y=0, color='k', linestyle='--', linewidth=1)

plt.show()
