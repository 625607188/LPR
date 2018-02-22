# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import cv2

Character = 0
License = 1
Train = 0
Test = 1

character_test_path = '../Train/ann/'
character_train_path = '../Train/annGray/'
license_test_path = ['../Train/svm/has/test',
                     '../Train/svm/no/test']
license_train_path = ['../Train/svm/has/train',
                      '../Train/svm/no/train']

character_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                     'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                     'W', 'X', 'Y', 'Z', 'zh_zhe']
character_length = len(character_classes)
character_label = [[0 for col in range(character_length)] for row in range(character_length)]
for i in range(character_length):
    character_label[i][i] = 1
license_classes = ['0', '1']
license_length = len(license_classes)
license_label = [[1, 0], [0, 1]]


# 生成训练所需的训练集tfrecords
def create_record(character_or_license, train_or_test):
    temp = ""
    path = ""
    if character_or_license == Character:
        classes = character_classes
        label = character_label
        if train_or_test == Train:
            temp = "train"
            print("Creating TFRecord Of Character Train...")
            path = character_train_path
        elif train_or_test == Test:
            temp = "test"
            print("Creating TFRecord Of Character Test...")
            path = character_test_path
        writer = tf.python_io.TFRecordWriter("../" + "character " + temp + ".tfrecords")
        for index, name in enumerate(classes):                                      # 枚举classes中所有项目
            print("Creating TFRecord Of ", name)
            class_path = path + name + '/'
            for img_name in os.listdir(class_path):
                img_path = class_path + img_name                                    # 记录照片路径
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)                    # 打开照片
                img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
                img_raw = img.tobytes()                                             # 将图片转化为原生bytes
                example = tf.train.Example(features=tf.train.Features(
                            feature={
                                     '_label': tf.train.Feature(int64_list=tf.train.Int64List(value=label[index])),
                                     '_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                                     }))
                writer.write(example.SerializeToString())
        writer.close()
    elif character_or_license == License:
        label = license_label
        if train_or_test == Train:
            temp = "train"
            print("Creating TFRecord Of License Train...")
            path = license_train_path
        elif train_or_test == Test:
            temp = "test"
            print("Creating TFRecord Of License Test...")
            path = license_test_path
        writer = tf.python_io.TFRecordWriter("../" + "license " + temp + ".tfrecords")
        for index, name in enumerate(path):                                             # 枚举classes中所有项目
            print("Creating TFRecord Of ", index)
            class_path = name + '/'
            for img_name in os.listdir(class_path):
                img_path = class_path + img_name                                        # 记录照片路径
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)                            # 打开照片
                img = cv2.resize(img, (40, 12), interpolation=cv2.INTER_AREA)
                img_raw = img.tobytes()                                                 # 将图片转化为原生bytes
                print(class_path + "     " + str(label[index]))
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        '_label': tf.train.Feature(int64_list=tf.train.Int64List(value=label[index])),
                        '_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }))
                writer.write(example.SerializeToString())
        writer.close()


# 读取tfrecords文件
def read_and_decode(filename, character_or_license):
    length = 0
    area = 0
    if character_or_license == Character:
        length = character_length
        area = 20 * 20
    elif character_or_license == License:
        length = license_length
        area = 40 * 12 * 3
    # 创建一个文件队列来维护文件列表,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # 创建一个reader用于读取队列
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            # TensorFlow提供两种不同的属性解析方法。一种是方法tf.FixedLenFeature，
            # 这种方法解析的结果为一个Tensor。另一种方法是tf.VarLenFeature，这种方法
            # 得到的解析结果为SparseTensor，用于处理稀疏数据。这里解析数据的格式需要和
            # 上面程序写入数据的格式一样。
            '_label': tf.FixedLenFeature([length], tf.int64),
            '_image': tf.FixedLenFeature([], tf.string)
        }
    )
    # tf,decode_raw可以将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['_image'], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [area])
    label = tf.cast(features['_label'], tf.int64)
    return image, label


def main(_):
    create_record(License, Train)
    create_record(License, Test)
    '''create_record(Character, Test)
    create_record(License, Train)
    create_record(License, Test)'''
    '''image, label = read_and_decode("E:/iCloudDrive/车牌识别-python/train.tfrecords")
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label], batch_size=5, capacity=train.CAPACITY, min_after_dequeue=100, num_threads=1)
    label_batch = tf.reshape(label_batch, [5, Length])

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                for i in range(5):
                    xs, ys = sess.run([image_batch, label_batch])
                    print(xs)
                    print(ys)
                    time.sleep(5)
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)'''


if __name__ == '__main__':
    tf.app.run()
