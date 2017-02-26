# coding: utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def add_layer(inputs, n_features, neurons, activation_function=None):
    '''一个神经层包括输入值，输入大小，输出大小，激励函数'''
    # 定义一个in_size*out_size的服从正态分布的随机张量
    Weights = tf.Variable(tf.random_normal([n_features, neurons]))
    # 定义一个1*out_size的偏置，由于不推荐为0，加0.1
    biases = tf.Variable(tf.zeros([1, neurons]) + 0.1)
    # (m_samples, n_features)*(n_features, neurons)+(1, neurons)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs

# -1到1的300个数据
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
# 高斯噪声，均值0，方差0.05
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = x_data**2 + x_data**3 + noise

# 定义两个占位符
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)
# 计算预测值prediction和真实值的误差，对二者差的平方求和再取平均。
loss = tf.reduce_mean(
    tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

sess.run(init)
for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(1)
