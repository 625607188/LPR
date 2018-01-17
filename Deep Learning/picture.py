import matplotlib.pyplot as plt
import tensorflow as tf

image_data = tf.gfile.FastGFile("/train/ann.7z/0/4-3.jpg", 'r').read()