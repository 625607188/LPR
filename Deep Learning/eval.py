# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# 加载inference.py和train.py中定义的常量和函数。
import CharacterRecognition
import inference
import tfrecord


def image_to_character(image_path):
    img = cv2.imread(image_path, 0)
    blur = img
    ret3, image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)          # 二值化处理
    image = cv2.equalizeHist(image)                                                         # 均值化处理

    (x, y) = image.shape

    image = list(image)                             # 行处理
    line = [0] * x
    sum = 0
    for i in range(x):
        for l in range(y):
            line[i] += image[i][l]
        sum += line[i]

    for i in range(x):
        if line[x - i - 1] < (sum / x /2):
            del image[x - i - 1]
    image = np.array(image)

    (x, y) = image.shape                            # 列处理
    line = [0] * y
    para = [[]]
    section = 0
    top = None
    for i in range(y):
        line[i] = 0
        for l in range(x):
            line[i] += image[l][i]
        if (line[i] < 300) and (top != None):
            bottom = i
            for m in range(x):
                para[section].append(image[m][top:bottom])
            para.append([])
            section += 1
            top = None
        elif (line[i] > 300) and (top == None):
            top = i

    '''for i in range(section):
        plt.subplot(4, 4, i + 1), plt.imshow(para[i])
    plt.show()'''

    for i in range(section):
        para[i] = np.array(para[i]).reshape(x, -1)

    l = 0
    for i in range(section):
        temp = para[section - i - 1].sum()
        if temp < 5000:
            del para[section - i - 1]
            l = l + 1
    section = section - l

    image = para

    return section, image


def evaluate_one_character(image_path):
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)

    with tf.Graph().as_default():
        x = tf.placeholder(
            tf.float32, [None, inference.INPUT_NODE], name='x-input')
        # 直接使用inference.py中定义的前向传播过程。
        y_conv, keep_prob = CharacterRecognition.deepnn(x)
        y_soft = tf.nn.softmax(y_conv)
        pre = tf.argmax(y_soft, 1)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(
                CharacterRecognition.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型。
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 通过文件名得到模型保存时迭代的轮数。
                global_step = ckpt.model_checkpoint_path\
                    .split('/')[-1].split('-')[-1]
                print("Loading success, global_step is %s " % global_step)
                image = tf.reshape(img, [20 * 20])
                xs = sess.run([image])
                prediction = int(pre.eval(feed_dict={x: xs, keep_prob: 1.0}))
                print(tfrecord.classes[prediction])
            else:
                print('No checkpoint file found')

        plt.subplot(1, 1, 1)
        plt.imshow(img)
        plt.show()


def get_one_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((20, 20))
    image = np.array(image)
    image = 255 - image
    return image


def evaluate_one_image(image_path):
    section, para = image_to_character(image_path)
    result = ""
    with tf.Graph().as_default():
        para.remove([])
        for i in range(section):
            (x, y) = para[i].shape
            if x > y:
                para[i] = cv2.copyMakeBorder(para[i], 0, 0, int((x - y) / 2), int((x - y) / 2), cv2.BORDER_CONSTANT,
                                             value=[0, 0, 0])
            elif x < y:
                para[i] = cv2.copyMakeBorder(para[i], int((y - x) / 2), int((y - x) / 2), 0, 0, cv2.BORDER_CONSTANT,
                                             value=[0, 0, 0])
            elif x == y:
                para[i] = para[i]
            para[i] = cv2.resize(para[i], (20, 20), interpolation=cv2.INTER_AREA)
            para[i] = tf.reshape(para[i], [20*20])

        x = tf.placeholder(
            tf.float32, [None, inference.INPUT_NODE], name='x-input')
        # 直接使用inference.py中定义的前向传播过程。
        y_conv, keep_prob = CharacterRecognition.deepnn(x)
        y_soft = tf.nn.softmax(y_conv)
        pre = tf.argmax(y_soft, 1)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(CharacterRecognition.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型。
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 通过文件名得到模型保存时迭代的轮数。
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print("Loading success, global_step is %s " % global_step)
                for i in range(section):
                    xs = sess.run([para[i]])
                    prediction = int(pre.eval(feed_dict={x: xs, keep_prob: 1.0}))
                    if 'zh_zhe' == tfrecord.character_classes[prediction]:
                        result = result + '浙'
                    else:
                        result = result + tfrecord.character_classes[prediction]
                    print(tfrecord.character_classes[prediction])
            else:
                print('No checkpoint file found')
    return result


def main(argv = None):
    for i in range(1917):
        path = "C:/Users/Hao/Desktop/test/" + str(i) + ".jpg"
        image = evaluate_one_image(path)
        

if __name__ == '__main__':
    tf.app.run()

