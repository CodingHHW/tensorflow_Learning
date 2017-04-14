import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "",
                           "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0,
                            "Index of task within the job")
FLAGS = tf.app.flags.FLAGS


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
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
