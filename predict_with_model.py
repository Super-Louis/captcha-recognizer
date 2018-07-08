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

def tf_variables():
    # 输入与输出
    X = tf.placeholder(tf.float32, shape=[None, 24, 18, 1], name='x-input')
    y_ = tf.placeholder(tf.int32, shape=[None, 47], name='y-input')
    y_ = tf.cast(y_, 'float')

    # 卷积层1
    w_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], stddev=0.1))
    bias_conv1 = tf.Variable(tf.constant(0.1, shape=[6]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(X, w_conv1, padding='SAME', strides=[1, 1, 1, 1]) + bias_conv1)
    # 池化层1
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 卷积层2
    w_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], stddev=0.1))
    bias_conv2 = tf.Variable(tf.constant(0.1, shape=[16]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, padding='VALID', strides=[1, 1, 1, 1]) + bias_conv2)
    # 池化层2
    # h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 全连接层1
    W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 5 * 16, 120], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[120]))
    h_conv2_flat = tf.reshape(h_conv2, [-1, 8 * 5 * 16])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    w1_loss = tf.contrib.layers.l2_regularizer(0.001)(W_fc1)

    # 全连接层2
    W_fc2 = tf.Variable(tf.truncated_normal(shape=[120, 84], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[84]))
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    w2_loss = tf.contrib.layers.l2_regularizer(0.001)(W_fc2)
    # keep_prob2 = tf.placeholder(tf.float32)
    # h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob2)

    # 输出层
    W_fc3 = tf.Variable(tf.truncated_normal(shape=[84, 47], stddev=0.1))
    b_fc3 = tf.Variable(tf.constant(0.1, shape=[47]))
    y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)
    w3_loss = tf.contrib.layers.l2_regularizer(0.001)(W_fc3)

    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(
    # 0.0001,
    # global_step,
    # X_train.shape[0] / 100,
    # 0.99
    # )
    # 损失函数与准确率
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
    losses = cross_entropy + w1_loss + w2_loss + w3_loss
    train_step = tf.train.AdamOptimizer(1e-3).minimize(losses)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return y_conv, X, keep_prob

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

