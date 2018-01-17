# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import tfrecord
import argparse
import sys
import tempfile


BATCH_SIZE = 5000                    # 一个训练batch中的训练数据个数。数字越小时，训练过程越接近
CAPACITY = 10000 + 3 * BATCH_SIZE

# 配置神经网络的参数。
INPUT_NODE = 20*20
OUTPUT_NODE = 35

IMAGE_SIZE = 20
NUM_CHANNELS = 1
NUM_LABLES = 35

# 第一层卷积层的尺寸和深度。
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷积层的尺寸和深度。
CONV2_DEEP = 64
CONV2_SIZE = 5
# 全连接层的节点个数。
FC_SIZE = 1024

MODEL_SAVE_PATH = "D:/final work/FinalWork-Ms.Wu/Project/Model/character/"
MODEL_NAME = "model.kpt"


# 定义卷积神经网络的前向传播过程
def deepnn(x):
    # 将输入转化为卷积层的输入格式。
    # 通过使用不同的命名空间来隔离不同层的变量，这可以让每一层中的变量命名只需要
    # 考虑在当前层的作用，而不需担心重名的问题。
    # 其中第一维表示一个batch中样例的个数；第二维和第三维表示图片的尺寸；第四维
    # 表示图片的深度。当参数为-1时，根据剩下的维度计算出数组的另外一个shape属性值。
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

    # 声明第一层卷积层的前向传播过程。定义的卷积层输入为20*20*1的图像像素。因为卷积
    # 层中使用了全0填充，所以输出为20*20*32。
    with tf.name_scope('layer1-conv1'):
        W_conv1 = weight_variable([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP])
        b_conv1 = bias_variable([CONV1_DEEP])

        # 使用边长为5，深度为32的过滤器，过滤器移动的步长为1，且使用全0填充。
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # 声明第二层池化层的前向传播过程。这里选用最大池化层，池化层过滤器的边长为2，
    # 使用全0填充且移动的步长为2。这一层的输入是上一层的输出，也就是20*20*32的矩
    # 阵。输出为10*10*32的矩阵。
    with tf.name_scope('layer2-pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # 声明第三层卷积层的变量并实现前向传播过程。这一层的输入为10*10*32的矩阵。输
    # 出为10*10*64的矩阵。
    with tf.name_scope('layer3-conv2'):
        W_conv2 = weight_variable([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP])
        b_conv2 = bias_variable([CONV2_DEEP])

        # 使用边长为5，深度为64的过滤器，过滤器的步长为1，且使用全0填充。
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # 实现第四层池化层的前向传播过程。这一层和第三层的结构是一样的。这一层的输入
    # 为10*10*64的矩阵，输出为5*5*64的矩阵。
    with tf.name_scope('layer4-pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # 将第四层池化层的输出转化为第五层全连接层的输入格式。第四层的输出为为5*5*64
    # 的矩阵，然而第五层全连接层需要的输入格式为向量，所以在这里需要将这个5*5*64
    # 的矩阵拉直成一个向量。h_pool2.get_shape函数可以得到第四层输出矩阵的维度而不
    # 需要手工计算。注意因为每一层神经网络的输入输出都为一个batch的矩阵，所以这里
    # 得到的维度也包含了一个batch中数据的个数。
    pool_shape = h_pool2.get_shape().as_list()

    # 计算将矩阵拉直成向量之后的长度，这个长度就是矩阵长宽及深度的乘积。注意这里
    # pool_shape[0]为一个batch中数据的个数。
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    # 通过tf.reshape函数将第四层的输出变为一个batch的变量。
    reshaped = tf.reshape(h_pool2, [-1, nodes])

    # 声明第五层全连接层的变量并实现前向传播过程。这一层的输入为5*5*64的矩阵，输出
    # 为一组长度为1024的向量。这一层
    with tf.name_scope('layer5-fc1'):
        W_fc1 = weight_variable([nodes, FC_SIZE])
        b_fc1 = bias_variable([FC_SIZE])

        h_fc1 = tf.nn.relu(tf.matmul(reshaped, W_fc1) + b_fc1)

    # 引入了dropout的概念。dropout在训练时会随机将部分节点的输出改为0。dropout可以
    # 避免过拟合问题，从而使得模型在测试数据上的效果更好。
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 声明第六层全连接层的变量并实现前向传播过程。这一层的输入为一组长度为1024的向
    # 量，输出为一组长度为2的向量。这一层的输出通过softmax之后就得到了最后的分类结
    # 果。
    with tf.name_scope('layer6-fc2'):
        W_fc2 = weight_variable([FC_SIZE, NUM_LABLES])
        b_fc2 = bias_variable([NUM_LABLES])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_conv, keep_prob


# tf.nn.conv2d 提供了一个非常方便的函数来实现卷积层前向传播算法。这个函数的第一个输
# 入为当前层的节点矩阵。注意这个矩阵是一个四维矩阵，后面三个维度对应一个节点矩阵，第一
# 维对应一个输入batch。比如在输入层，input[0,:,:,:]表示第一张图片，input[1,:,:,:]
# 表示第二张图片，以此类推。tf.nn.conv2d第二个参数提供了卷积层的权重，第三个参数为不
# 同维度上的步长。虽然第三个参数提供的是一个长度为4 的数组，但是第一维和最后一维的数字
# 要求一定是1。这是因为卷积层的步长只对只对矩阵的长和宽有效。最后一个参数是填充，
# “VALID”表示不添加。
def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# tf.nn.max_pool实现了最大池化层的前向传播过程，它的参数和tf.nn.conv2d函数类似。
# ksize提供了过滤器的尺寸，strides提供了步长信息，padding提供了是否使用全0填充。
def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    # tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是
    # 均值，stddev是标准差。这个函数产生正太分布，均值和标准差自己设定。这是一个
    # 截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的
    # 标准差，那就重新生成。和一般的正太分布的产生随机数据比起来，这个函数产生的
    # 随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # Import data
    image_train, label_train = tfrecord.read_and_decode(
        "D:/final work/FinalWork-Ms.Wu/Project/character train.tfrecords")
    image_train_batch, label_train_batch = tf.train.shuffle_batch(
        [image_train, label_train], batch_size=BATCH_SIZE, capacity=CAPACITY, min_after_dequeue=500, num_threads=1)

    image_test, label_test = tfrecord.read_and_decode(
        "D:/final work/FinalWork-Ms.Wu/Project/character test.tfrecords")
    image_test_batch, label_test_batch = tf.train.shuffle_batch(
        [image_test, label_test], batch_size=BATCH_SIZE, capacity=CAPACITY, min_after_dequeue=500, num_threads=1)

    # 定义神经网络的输入。
    x = tf.placeholder(
        tf.float32, [None, INPUT_NODE], name='x-input')

    # 定义神经网络的输出。
    y_ = tf.placeholder(
        tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 调用卷积神经网络。
    y_conv, keep_prob = deepnn(x)

    # 定义交叉熵损失函数。
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 设置指数衰减学习率。学习率 = learning_rate * decay_rate^(global_step/decay_steps)
    with tf.name_scope('adam_optimizer'):
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            learning_rate=1e-5, global_step=global_step, decay_steps=5000, decay_rate=0.98,
            staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_mean,
                                                                    global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            # 加载模型。
            saver.restore(sess, ckpt.model_checkpoint_path)
            # 通过文件名得到模型保存时迭代的轮数。
            global_step = int(ckpt.model_checkpoint_path\
                              .split('/')[-1].split('-')[-1])
            print("Loading success, global_step is %d " % global_step)
        else:
            global_step = 0
            print('No checkpoint file found')
        try:
            while not coord.should_stop():
                global_step = 1 + global_step
                xs, ys = sess.run([image_train_batch, label_train_batch])
                if (global_step % 10 == 0) and global_step != 0:
                    xs_test, ys_test = sess.run([image_test_batch, label_test_batch])
                    train_accuracy = accuracy.eval(feed_dict={
                        x: xs_test, y_: ys_test, keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (global_step, train_accuracy))
                    saver.save(
                        sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                        global_step=global_step)
                _, test_accuracy = sess.run([train_step, accuracy], feed_dict={
                    x: xs, y_: ys, keep_prob: 0.5})
                print('test accuracy %g' % test_accuracy)
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])
