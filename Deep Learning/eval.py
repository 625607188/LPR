# -*- coding: utf-8 -*-
import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# 加载inference.py和train.py中定义的常量和函数。
import CharacterRecognition
import LicenseRecognition
import inference
import tfrecord


def image_to_character1(image):
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)          # 二值化处理
    image = cv2.equalizeHist(image)                                                       # 均值化处理

    (y, x) = image.shape

    line = list(map(sum, image))  # 行处理
    total = sum(line)

    for i in range(y - 1, -1 - 1):
        if line[i] < (total / y / 2):
            del image[i]

    (y, x) = image.shape  # 列处理
    para = []
    start = None
    column = list(map(sum, zip(*image)))
    for i in range(x):
        if (column[i] < 300) and (start is not None):
            end = i
            para.append([temp[start:end] for temp in image])
            start = None
        elif (column[i] > 300) and (start is None):
            start = i

    for i in range(len(para) - 1, -1, -1):
        temp = sum(sum(para[i]))
        if temp < 1000:
            del para[i]

    for i in range(len(para)):
        para[i] = np.array(para[i])
        (y, x) = para[i].shape
        if y > x:
            para[i] = cv2.copyMakeBorder(para[i], 0, 0, int((y - x) / 2), int((y - x) / 2), cv2.BORDER_CONSTANT,
                                         value=[0, 0, 0])
        elif y < x:
            para[i] = cv2.copyMakeBorder(para[i], int((x - y) / 2), int((x - y) / 2), 0, 0, cv2.BORDER_CONSTANT,
                                         value=[0, 0, 0])
        elif x == y:
            para[i] = para[i]
        para[i] = cv2.resize(para[i], (20, 20), interpolation=cv2.INTER_AREA)

    '''for i in range(section):
            plt.subplot(4, 4, i + 1), plt.imshow(para[i])
    plt.show()'''

    return para


def image_to_character2(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)            # 二值化处理
    image0 = cv2.equalizeHist(image)                                                        # 均值化处理

    (y, x) = image0.shape
    column = list(map(sum, image0))  # 删除头和尾的行空白
    for i in range(y):
        if column[i] > 255 * 5:
            start = i
            break
    for i in list(range(y - 1, -1, -1)):
        if column[i] > 255 * 5:
            end = i
            break
    delete = list(range(0, start)) + list(range(end, y))
    for i in range(y):
        if column[i] > 255 * x * 4 / 5:
            delete.append(i)
    image1 = np.delete(image0, list(set(delete)), axis=0)

    '''(y, x) = image1.shape
    column = list(map(sum, zip(*image1)))  # 删除头和尾的列空白
    for i in range(x):
        if column[i] > 0:
            start = i
            break
    for i in list(range(x - 1, -1, -1)):
        if column[i] > 0:
            end = i
            break
    delete = list(range(0, start)) + list(range(end, x))
    for i in range(x):
        if column[i] > 255 * y * 9 / 10:
            delete.append(i)
    image2 = np.delete(image1, list(set(delete)), axis=1)'''

    image2 = image1
    (y, x) = image2.shape
    column = list(map(sum, zip(*image2)))  # 分割字符
    total = sum(column)
    start, end = (0, x - 1)
    para = []
    while start < x - 1:
        flag = 0
        for i in range(start, end):
            if column[i] > 255 * 1:
                flag = 1
            if column[i] < 255 * 1 and flag:
                end = i + 1
                if end > x:
                    end = x
                break
        para.append([temp[start:end] for temp in image2])
        start, end = (end, x - 1)
        para[-1] = np.array(para[-1])
        if para[-1].sum() < total / 7 / 5:
            del para[-1]

    for i in range(len(para)):
        (y, x) = para[i].shape
        if y > x:
            para[i] = cv2.copyMakeBorder(para[i], 0, 0, int((y - x) / 2), int((y - x) / 2), cv2.BORDER_CONSTANT,
                                         value=[0, 0, 0])
        elif y < x:
            para[i] = cv2.copyMakeBorder(para[i], int((x - y) / 2), int((x - y) / 2), 0, 0, cv2.BORDER_CONSTANT,
                                         value=[0, 0, 0])
        elif x == y:
            para[i] = para[i]
        para[i] = cv2.resize(para[i], (20, 20), interpolation=cv2.INTER_AREA)

    '''for i in range(section):
            plt.subplot(4, 4, i + 1), plt.imshow(para[i])
    plt.show()'''

    return para


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


def evaluate_characters(paragraphs):
    result = ""
    with tf.Graph().as_default():
        for i in range(len(paragraphs)):
            paragraphs[i] = tf.reshape(paragraphs[i], [20*20])

        x = tf.placeholder(
            tf.float32, [None, CharacterRecognition.INPUT_NODE], name='x-input')
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
                for i in range(len(paragraphs)):
                    xs = sess.run([paragraphs[i]])
                    prediction = int(pre.eval(feed_dict={x: xs, keep_prob: 1.0}))
                    if 'zh_zhe' == tfrecord.character_classes[prediction]:
                        result = result + '浙'
                    else:
                        result = result + tfrecord.character_classes[prediction]
                    print(tfrecord.character_classes[prediction])
            else:
                print('No checkpoint file found')
    return result


def evaluate_one_photo(image):
    x_sum = 0
    y_sum = 0
    num = 0
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None, LicenseRecognition.INPUT_NODE], name='x-input')
        # 直接使用inference.py中定义的前向传播过程。
        y_conv, keep_prob = LicenseRecognition.deepnn(x)
        y_soft = tf.nn.softmax(y_conv)
        pre = tf.argmax(y_soft, 1)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(LicenseRecognition.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型。
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 通过文件名得到模型保存时迭代的轮数。
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print("Loading success, global_step is %s " % global_step)
                for i in range(120,  170, 5):
                    for l in range(100,  200,  10):
                        img = image[i:(i+12*3), l:(l+40*3)]
                        img = cv2.resize(img, (40, 12), interpolation=cv2.INTER_AREA)
                        img = np.array(img)
                        img = tf.reshape(img, [40 * 12 * 3])

                        xs = sess.run([img])
                        prediction = int(pre.eval(feed_dict={x: xs, keep_prob: 1.0}))
                        # print("x is %d, y is %d, pre is %d" % (i, l, prediction))
                        if '0' == tfrecord.license_classes[prediction]:
                            num += 1
                            x_sum += l
                            y_sum += i
            else:
                print('No checkpoint file found')
                return 0
    x_sum = int(x_sum / num)
    y_sum = int(y_sum / num)
    image = image[y_sum:(y_sum + 48), x_sum:(x_sum + 160)]
    image = cv2.resize(np.array(image), (136, 36))
    return image


def eval(image_path):
    path = "C:/Users/Hao/Desktop/temp/"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    y_, x_ = image.shape
    x_sum = 0
    y_sum = 0
    num = 0
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None, LicenseRecognition.INPUT_NODE], name='x-input')
        # 直接使用inference.py中定义的前向传播过程。
        y_conv, keep_prob = LicenseRecognition.deepnn(x)
        y_soft = tf.nn.softmax(y_conv)
        pre = tf.argmax(y_soft, 1)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(LicenseRecognition.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型。
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 通过文件名得到模型保存时迭代的轮数。
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print("Loading success, global_step is %s " % global_step)
                for i in range(int(y_ - 48)):
                    for l in range(int(x_ - 160)):
                        img = image[i:(i+48), l:(l+160)]
                        img = cv2.resize(img, (40, 12), interpolation=cv2.INTER_AREA)
                        img = np.array(img)
                        img = tf.reshape(img, [40 * 12 * 3])

                        xs = sess.run([img])
                        prediction = int(pre.eval(feed_dict={x: xs, keep_prob: 1.0}))
                        '''if '0' == tfrecord.license_classes[prediction]:
                            print("x is %d, y is %d, pre is %d" % i, l, prediction)
                        else:
                            print("x is %d, y is %d, pre is %d" % i, l, prediction)'''
                        print("x is %d, y is %d, pre is %d" % (i, l, prediction))
                        if '0' == tfrecord.license_classes[prediction]:
                            num += 1
                            x_sum += (i+44)*10
                            y_sum += (l*10)

                x_sum = int(x_sum/num)
                y_sum = int(y_sum/num)
                image = image[x_sum:(x_sum + 120), y_sum:(y_sum + 480)]
                image = cv2.resize(image, (136, 36), interpolation=cv2.INTER_AREA)
                cv2.imwrite(path + "0.jpg", image)
            else:
                print('No checkpoint file found')


def main(_):
    #evaluate_one_image("D:/final work/FinalWork-Ms.Wu/Project/Train/svm/has/test/0.jpg")
    #eval("C:/Users/Hao/Desktop/temp/test.jpg")
    #image_to_character("C:/Users/Hao/Desktop/temp/0.jpg")

    '''image_path = "C:/Users/Hao/Desktop/test/"
    path = "D:/final work/FinalWork-Ms.Wu/Project/Train/temp/"
    a = 0
    for name in os.listdir(image_path):
        if ".jpg" in name:
            print(image_path + name)
            para = image_to_character2(image_path + name)
            for i in para:
                cv2.imwrite(path + str(a) + '.jpg', i)
                print(path + str(a) + '.jpg')
                a = a + 1'''

    image_path = "C:/Users/Hao/Desktop/test/"
    path = "D:/final work/FinalWork-Ms.Wu/Project/Train/temp/"
    result_path = "D:/final work/FinalWork-Ms.Wu/temp/"
    for name in os.listdir(path):
        if ".jpg" in name:
            para = cv2.imread(path + name, cv2.IMREAD_GRAYSCALE)
            result = evaluate_characters([para])
            cv2.imwrite(result_path + result + '/' + name, para)
            print(result_path + result + '/' + name)


if __name__ == '__main__':
    tf.app.run()

