# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 定义参数服务器ps（主要是保存和更新参数的节点）
# 和工作服务器worker（主要负责参数计算的节点）
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
# 定义工作名字和任务序列号
tf.app.flags.DEFINE_string("job_name", "",
                           "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0,
                            "Index of task within the job")

FLAGS = tf.app.flags.FLAGS


def main(_):
    # 得到参数服务器和工作服务器IP和端口号
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    # 创建参数服务器和工作服务器集群，包含所有的ps节点和worker节点
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # 设置一个服务器，通过设置FLAGS参数，决定执行的任务是ps还是worker
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        # 如果是参数服务器，就join，为了避免进程退出，等待其他worker节点给他提交数据
        server.join()
    elif FLAGS.job_name == "worker":
        # 如果是工作服务器，就执行下面的计算任务
        """
        下面使用between-graph replication（图间的拷贝）
        在这种情况下，每一个任务(/job:worker)都是通过独立客户端单独声明的。
        其相互之间结构类似，每一个客户端都会建立一个相似的图结构，该结构中包含的参数均通过ps
        作业(/job:ps)进行声明并使用tf.train.replica_device_setter()方法将参数映射到
        不同的任务中。模型中每一个独立的计算单元都会映射到/job:worker的本地的任务中。
        """
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            # 设置一个记录全局训练步骤的单值，以及使用minimize操作，该操作不仅可以优化更新
            # 训练的模型参数，也可以为全局步骤（global step）计数
            global_step = tf.Variable(0, name='global_step', trainable=False)
            input = tf.placeholder("float")
            label = tf.placeholder("float")
            weight = tf.get_variable("weight", [1], tf.float32,
                                     initializer=tf.random_normal_initializer())
            biase = tf.get_variable("biase", [1], tf.float32,
                                    initializer=tf.random_normal_initializer())
            pred = tf.multiply(input, weight) + biase
            loss = tf.square(label - pred)

            train_op = tf.train.AdagradOptimizer(0.01).minimize(
                loss, global_step=global_step)

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()
        # 定义一个监督者，因为分布式，很多机器都在运行，如参数初始化，保存模型，写summary等，
        # Supervisor帮忙一起弄起来了，免得手工做这些事情
        # is_chief比较重要，设置一个主节点负责初始化参数，模型的保存，summary的保存。
        # logdir就是保存和装载模型的路径。
        # 主worker节点负责模型参数初始化等工作，在这个过程中，其他worker节点等待主节点完成
        # 初始化工作，等主节点初始化完成后，大家就开始一起跑数据
        # global_step是所有计算节点共享的，在执行optimizer的minimize的时候，会自动加1，
        # 所以可以通过这个可以知道所有的计算节点一共计算了多少步。
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="/tmp/train_logs",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)
        with sv.managed_session(server.target) as sess:
            step = 0
            while not sv.should_stop() and step < 1000:
                train_x = np.random.randn(1)
                train_y = 2 * train_x + np.random.randn(1) * 0.33 + 10
                _, loss_v, step = sess.run([train_op, loss, global_step],
                                           feed_dict={input: train_x, label: train_y})
                if step % 50 == 0:
                    w, b = sess.run([weight, biase])
                    print("step: %d, weight: %f, biase: %f, loss: %f"
                          % (step, w, b, loss_v))
        sv.stop()

if __name__ == "__main__":
    tf.app.run()
