# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: cnn_utils.py
# Python  : python3.6
# Time    : 18-7-12 21:33
# Github  : https://github.com/Super-Louis
import tensorflow as tf

def preprocess_for_train(image):
    image = tf.image.resize_images(image, [48, 120]) # todo: 图片必须是三维或4维
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.per_image_standardization(image)
    return image

def parser(record):
    featrues = tf.parse_single_example(
        record,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        }
    )
    decoded_image = tf.decode_raw(featrues['image_raw'], tf.uint8)
    decoded_image = tf.reshape(decoded_image, [48, 120, 1]) # todo: 图片在测试前应resize，此处resize只是将一维转为三维
    label = tf.decode_raw(featrues['label'], tf.int64)
    return decoded_image, label

def inference(input_tensor, keep_prob):
    conv1_weights = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], stddev=0.1), name='conv1_weights')
    conv1_bias = tf.Variable(tf.constant(0.1, shape=[6]), name='conv1_bias')
    conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1,1,1,1], padding='SAME')
    relu1 = tf.nn.relu(tf.add(conv1, conv1_bias))
    pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    conv2_weights = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], stddev=0.1), name='conv2_weights')
    conv2_bias = tf.Variable(tf.constant(0.1, shape=[16]), name='conv2_bias')
    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
    relu2 = tf.nn.relu(tf.add(conv2, conv2_bias))
    pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    conv3_weights = tf.Variable(tf.truncated_normal(shape=[5, 5, 16, 30], stddev=0.1), name='conv3_weights')
    conv3_bias = tf.Variable(tf.constant(0.1, shape=[30]), name='conv3_bias')
    conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='VALID')
    relu3 = tf.nn.relu(tf.add(conv3, conv3_bias))
    pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    fc1_weights = tf.Variable(tf.truncated_normal(shape=[3*12*30, 500], stddev=0.1), name='fc1_weights')
    fc1_bias = tf.Variable(tf.constant(0.1, shape=[500]), name='fc1_bias')
    pool3_flatten = tf.reshape(pool3, [-1, 3*12*30])
    fo1 = tf.nn.relu(tf.matmul(pool3_flatten, fc1_weights) + fc1_bias)
    fo1 = tf.nn.dropout(fo1, keep_prob=keep_prob)
    fw1_loss = tf.contrib.layers.l2_regularizer(0.001)(fc1_weights)
    tf.add_to_collection('losses', fw1_loss)

    fc2_weights = tf.Variable(tf.truncated_normal(shape=[500, 208], stddev=0.1), name='fc2_weights')
    fc2_bias = tf.Variable(tf.constant(0.1, shape=[208]), name='fc2_bias')
    y_out = tf.nn.softmax(tf.matmul(fo1, fc2_weights) + fc2_bias, name='y_pred')
    fw2_loss = tf.contrib.layers.l2_regularizer(0.001)(fc2_weights)
    tf.add_to_collection('losses', fw2_loss)
    return y_out

def get_iterator(params, path):
    batch_size, shuffle_buffer, num_epochs = params
    dataset = tf.data.TFRecordDataset([path])
    dataset = dataset.map(parser)
    dataset = dataset.map(lambda image, label:(preprocess_for_train(image), label)) # 返回要加括号，不然会报错
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_initializable_iterator()
    return iterator

def calc_accuracy(y_pred, y_):
    slices = []
    for i in range(4):
        y_pred_slice = tf.slice(y_pred, [0, 52*i], [100, 52])
        y__slice = tf.slice(y_, [0, 52*i], [100, 52])
        y_equal_slice = tf.equal(tf.argmax(y__slice, 1), tf.argmax(y_pred_slice, 1))
        slices.append(tf.expand_dims(y_equal_slice,1))
    concat_ = slices[0]
    for s in slices[1:]:
        concat_ = tf.concat([concat_, s], axis=1)
    equal_sum_row = tf.reduce_sum(tf.cast(concat_, tf.float32), axis=1)
    equal_4 = tf.constant(4, shape=[100], dtype=tf.float32)
    equal_final = tf.equal(equal_sum_row, equal_4)
    accuracy = tf.reduce_mean(tf.cast(equal_final, "float"))
    return accuracy




