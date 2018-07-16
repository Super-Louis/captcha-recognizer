# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: predict_with_model.py
# Python  : python3.6
# Time    : 18-6-10 13:26
# Github  : https://github.com/Super-Louis

from captcha_handle.gen_sample import *
import tensorflow as tf
from PIL import Image
import re
from model import cnn_utils


def image_handler(file):
    img = Image.open(file)
    l_img = img.convert('L')
    r_img = l_img.resize((120, 48),Image.ANTIALIAS)
    img_array = np.array(r_img)
    letters = re.findall('(\w{4})_?\d?\.\w+', file)[0]
    return img_array, ''.join(letters)

def predict(from_local=False, file_path=''):
    """
    :param file: input captcha path
    :return: predicted letters
    """
    # y_conv, X, keep_prob = tf_variables()
    if from_local and file_path:
        X, letters = image_handler(file_path)
    else:
        X, letters = create_data(test=True)
    x = X.reshape([48, 120, 1])
    image = cnn_utils.preprocess_for_train(x) # 不要忘了数据预处理！！！
    saver = tf.train.import_meta_graph('./model/cnn_full_model_0.92.ckpt.meta')
    graph = tf.get_default_graph()
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, './model/cnn_full_model_0.92.ckpt')
        im = sess.run(image) # 先要得到真实的输入，不然只是tensor
        y_predict = sess.run(graph.get_tensor_by_name('y_pred_11:0'),
                             feed_dict={graph.get_tensor_by_name('x-input_11:0'): np.expand_dims(im, 0), # 三维转四维
                                        graph.get_tensor_by_name('Placeholder_11:0'): 1})
        y_label = decode_label(y_predict[0])
        print("true letters: {}; predict letters: {}".format(letters, y_label))

if __name__ == '__main__':
    letters = predict(from_local=False, file_path='t6fb_1.jpg')

