# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: predict_with_model.py
# Python  : python3.6
# Time    : 18-6-10 13:26
# Github  : https://github.com/Super-Louis

import pickle
import numpy as np
from captcha_handle.captcha_handler import CaptchaHandler
import tensorflow as tf


ch = CaptchaHandler()

def load_ml_model():
    # load the model from local file
    model = pickle.load(open('model/random_forest_model', 'rb'))
    le = pickle.load(open('model/label_encoder', 'rb'))
    return model, le

def predict(file, method='ml', test=True):
    """
    :param file: input captcha path
    :return: predicted letters
    """
    array, letters = ch.convert(file, threshold='manual', test=test)
    if not array:
        print('convert failed!')
        return []
    if method == 'ml':
        model, le = load_ml_model()
        letters_pred = model.predict(array)
        letters_pred_transform = le.inverse_transform(letters_pred)
        print(letters_pred_transform)
        return letters_pred_transform
    else:
        # y_conv, X, keep_prob = tf_variables()
        saver = tf.train.import_meta_graph('./model/cnn_model_0.99.ckpt.meta')
        graph = tf.get_default_graph()
        with tf.Session(graph=graph) as sess:
            # ckpt = tf.train.get_checkpoint_state('./model/')
            # print(ckpt.model_checkpoint_path)
            # saver.restore(sess, ckpt.model_checkpoint_path)
            saver.restore(sess, './model/cnn_model_0.99.ckpt')
            le = pickle.load(open('./data/le_cnn', 'rb'))
            mm = pickle.load(open('./data/mm_cnn', 'rb'))
            x_mm = mm.transform(array)
            x_re = np.reshape(x_mm, [-1, 24, 18, 1])
            y_predict = sess.run(graph.get_tensor_by_name('Softmax_1:0'),
                                 feed_dict={graph.get_tensor_by_name('x-input_1:0'): x_re,
                                            graph.get_tensor_by_name('Placeholder_1:0'): 1})
            y_label = np.argmax(y_predict, 1)
            predict_letters = le.inverse_transform(y_label)
            print(predict_letters)
            return predict_letters

if __name__ == '__main__':
    letters = predict('AaZ2_1.png', method='cnn')

