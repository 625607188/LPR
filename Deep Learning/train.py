# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf
# 加载inference.py中定义的常量和前向传播的函数。
import inference
import tfrecord


# 定义神经网络结构相关的参数
BATCH_SIZE = 500                    # 一个训练batch中的训练数据个数。数字越小时，训练过程越接近
                                    # 随机梯度下降；数字越大时，训练越接近梯度下降。
LEARNING_RATE_BASE = 0.8            # 基础的学习率。
LEARNING_RATE_DECAY = 0.99          # 学习率的衰减率
REGULARAZTION_RATE = 0.0001         # 描述模型复杂度的正则化项在损失函数中的系数。
TRAINING_STEPS = 100000             # 训练的轮数。
MOVING_AVERAGE_DECAY = 0.99         # 滑动平均衰减率。

# 模型保存的路径和文件名。
MODEL_SAVE_PATH = "E:/model/"
MODEL_NAME = "model.kpt"

CAPACITY = 3000 + 3 * BATCH_SIZE


def train(image, label):

    # 定义输入输出placeholder。
    x = tf.placeholder(
        tf.float32, [None, inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(
        tf.float32, [None, inference.OUTPUT_NODE], name='y-input')

    # 计算L2正则化损失函数。
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 直接使用inference.py中定义的前向传播过程。
    y = inference.inference(x, regularizer)
    # 定义存储训练轮数的变量。这个变量不需要计算滑动平均值，所以这里指定这个变量为
    # 不可训练的变量（trainable=False)。在使用TensorFlow训练神经网络时，
    # 一般会将代表训练轮数的变量指定为不可训练的参数。
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。
    # 给定训练轮数的变量可以加快训练早期变量的更新速度。
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均。
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables())
    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数。这里使用了TensorFlow中提
    # 供的sparse_softmax_cross_entropy_with_logits函数来计算交叉熵。当分类
    # 问题只有一个正确答案时，可以使用这个函数来加速交叉熵的计算。
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=y, labels=y_)
    # 计算在当前batch中所有样例的交叉熵平均值。
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 总损失等于交叉熵损失和正则化损失的和。
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,             # 基础的学习率，随着迭代的进行，更新变量时使用的
                                        # 学习率在这个基础上递减。
        global_step,                    # 当前迭代的轮数。
        100,                            # 迭代次数。
        LEARNING_RATE_DECAY)            # 学习率衰减速度。
    # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数。注意这里损失函数
    # 包含了交叉熵损失和L2正则化损失。
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss, global_step=global_step)

    # 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神将网络中的参数，
    # 又要更新每一个参数的滑动平均值。为了一次完成多个操作，TensorFlow提供了
    # tf.control_dependencies和tf.group两种机制。下面两个程序和
    # train_op = tf.group(train_step, variable_averages_op)是等价的。
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                sess.run(init_op)
                xs, ys = sess.run([image, label])
                # 迭代地训练神经网络。
                for i in range(TRAINING_STEPS):
                    _, loss_value, step = sess.run([train_op, loss, global_step],
                                                   feed_dict={x: xs, y_: ys})
                    # 每10轮输出一次在验证数据集上的测试结果。
                    if (i+1) % 1000 == 0:
                        print("After %d training step(s), loss on training "
                              "batch is %g " % (step, loss_value))
                        saver.save(
                            sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                            global_step=global_step)
                break
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)


# 主程序入口。
def main(argv = None):
    image, label = tfrecord.read_and_decode("E:/iCloudDrive/毕业设计-吴沈青/车牌识别-python/train.tfrecords")
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=BATCH_SIZE, capacity=CAPACITY, min_after_dequeue=500, num_threads=1)
    label_batch = tf.reshape(label_batch, [BATCH_SIZE, tfrecord.Length])
    train(image_batch, label_batch)


# tf.app.run()会调用上面定义的main函数。
if __name__ == '__main__':
    tf.app.run()
