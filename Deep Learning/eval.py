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

window1 = []
window2 = []
window3 = []
window4 = []
window5 = []
window6 = []


def show_debug_window():
    global window1, window2, window3, window4, window5, window6
    window1 = cv2.cvtColor(window1, cv2.COLOR_RGB2BGR)
    plt.subplot(3, 2, 1), plt.imshow(window1)
    plt.axis('off')
    window2 = cv2.cvtColor(window2, cv2.COLOR_GRAY2BGR)
    plt.subplot(3, 2, 3), plt.imshow(window2)
    plt.axis('off')
    plt.subplot(3, 2, 5), plt.imshow(window3)
    plt.axis('off')
    plt.subplot(3, 2, 2), plt.imshow(window4)
    plt.axis('off')
    window5 = cv2.cvtColor(window5, cv2.COLOR_GRAY2BGR)
    plt.subplot(3, 2, 4), plt.imshow(window5)
    plt.axis('off')
    window6 = cv2.cvtColor(window6, cv2.COLOR_GRAY2BGR)
    plt.subplot(3, 2, 6), plt.imshow(window6)
    plt.axis('off')
    plt.show()


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
    global window4, window5, window6
    window4 = image.copy()
    '''cv2.imshow("window4", window4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.GaussianBlur(image, (3, 3), 1)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)            # 二值化处理
    image0 = cv2.equalizeHist(image)                                                        # 均值化处理
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)            # 二值化处理

    window5 = image0.copy()
    '''cv2.imshow("window5", window5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    delete = []
    (y, x) = image0.shape
    column = list(map(sum, image0))                                                         # 删除头和尾的行空白和白边
    for i in range(y):
        if column[i] < 255 * 5:
            delete.append(i)
        else:
            temp = 0
            for l in range(x):
                if image0[i][l] != 0:
                    temp += 1
                else:
                    temp = 0
                if temp > x * 2.5 / 14:
                    delete.append(i)
                    break
    image1 = np.delete(image0, delete, axis=0)

    delete = []
    (y, x) = image1.shape
    column = list(map(sum, zip(*image1)))                                                   # 删除头和尾的列空白
    need = list(range(int(x/14))) + list(range(int(x*13/14), x))
    for i in need:
        if column[i] < 255 * 5: # or column[i] > 255 * y * 9 / 10:
            print(i)
            delete.append(i)
    image2 = np.delete(image1, delete, axis=1)

    window6 = image2.copy()
    '''cv2.imshow("window6", window6)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    (y, x) = image2.shape
    column = list(map(sum, zip(*image2)))  # 分割字符
    total = sum(column)
    start, end = (0, x - 1)
    para = []
    while start < x - 1:
        flag = 0
        for i in range(start, end):
            if column[i] > 255 * 2:
                flag = 1
            if column[i] < 255 * 2 and flag:
                end = i + 1
                if end > x:
                    end = x
                break
        para.append([temp[start:end] for temp in image2])
        start, end = (end, x - 1)
        while column[start] < 255 * 2 and start < end:
            start += 1
        para[-1] = np.array(para[-1])
        if para[-1].sum() < total / 7 / 3:
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

    '''for i in range(len(para)):
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
                    if 'zh_zhe' == tfrecord.character_classes[prediction] or i == 0:
                        result = result + '浙'
                    else:
                        result = result + tfrecord.character_classes[prediction]
                    print(tfrecord.character_classes[prediction])
            else:
                print('No checkpoint file found')
    return result


def evaluate_one_photo(image):

    image1 = image.copy()
    #image1 = cv2.GaussianBlur(image1, (3, 3), 1)

    global window1, window2, window3
    window1 = image1.copy()
    '''cv2.imshow("window1", window1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])

    lower_white = np.array([0, 0, 170])
    upper_white = np.array([180, 30, 255])

    hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    # get mask
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv, lower_white, upper_white)

    end = cv2.bitwise_or(mask1, mask2)

    window2 = end.copy()
    window3 = window2.copy()
    window3 = cv2.cvtColor(window3, cv2.COLOR_GRAY2BGR)

    '''cv2.imshow("window2", window2)
    cv2.imshow("window3", window3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    img, contours, hierarchy = cv2.findContours(end, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coordinate = []

    for i in contours:
        if len(i) > 50:
            x, y, w, h = cv2.boundingRect(i)
            if (w/h > 440/160) and (w/h < 440/100):
                coordinate.append([x+3, y+5, w-6, h-10])
                cv2.rectangle(window3, (x+3, y+5), (x+w-3, y+h-5), (255, 0, 0), 5)

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
                if coordinate:
                    for i in coordinate:
                        (x_, y_, w_, h_) = i
                        img = image[y_:(y_ + h_), x_:(x_ + w_)]
                        img = cv2.resize(img, (40, 12), interpolation=cv2.INTER_AREA)
                        img = np.array(img)
                        img = tf.reshape(img, [40 * 12 * 3])

                        xs = sess.run([img])
                        prediction = int(pre.eval(feed_dict={x: xs, keep_prob: 1.0}))

                        if '0' == tfrecord.license_classes[prediction]:
                            licence = image[y_:(y_ + h_), x_:(x_ + w_)]
                            licence = cv2.resize(licence, (136, 36), interpolation=cv2.INTER_AREA)
                            return x_, y_, w_, h_, licence
            else:
                print('No checkpoint file found')
                return 0, 0, 0, 0, np.array([])
    return 0, 0, 0, 0, np.array([])


def eval(image_path):
    path = "C:/Users/Hao/Desktop/temp/"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    y_, x_, _ = image.shape
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
    eval("C:/Users/Hao/Desktop/temp/test.jpg")
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
                a = a + 1

    image_path = "C:/Users/Hao/Desktop/test/"
    path = "D:/final work/FinalWork-Ms.Wu/Project/Train/temp/"
    result_path = "D:/final work/FinalWork-Ms.Wu/temp/"
    for name in os.listdir(path):
        if ".jpg" in name:
            para = cv2.imread(path + name, cv2.IMREAD_GRAYSCALE)
            result = evaluate_characters([para])
            cv2.imwrite(result_path + result + '/' + name, para)
            print(result_path + result + '/' + name)'''


if __name__ == '__main__':
    tf.app.run()

