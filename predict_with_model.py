# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: predict_with_model.py
# Python  : python3.6
# Time    : 18-6-10 13:26
# Github  : https://github.com/Super-Louis

import pickle
import numpy as np
from captcha_handle.captcha_handler import CaptchaHandler

ch = CaptchaHandler()

def load_model():
    # load the model from local file
    model = pickle.load(open('model/random_forest_model', 'rb'))
    le = pickle.load(open('model/label_encoder', 'rb'))
    return model, le

def predict(file):
    """
    :param file: input captcha path
    :return: predicted letters
    """
    array, letters = ch.convert(file, threshold='manual')
    if not array:
        print('convert failed!')
        return []
    model, le = load_model()
    letters_pred = model.predict(array)
    letters_pred_transform = le.inverse_transform(letters_pred)
    print(letters_pred_transform)
    return letters_pred_transform

if __name__ == '__main__':
    letters = predict('A12X.jpg')

