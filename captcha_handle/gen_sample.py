# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: gen_sample.py
# Python  : python3.6
# Time    : 18-7-11 19:28
# Github  : https://github.com/Super-Louis

from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random

import string
characters = string.digits + string.ascii_uppercase + string.ascii_lowercase
letter_exclude = ['l','1','O', 'w', 'c', 'k', 's', 'v', 'x', 'z']
characters_list = list(characters)
for l in letter_exclude:
    characters_list.remove(l)
characters = ''.join(characters_list)
def create_data(test=False):
    width, height, n_len, n_class = 120, 48, 4, len(characters)
    generator = ImageCaptcha(width=width, height=height)
    random_str = ''.join([random.choice(characters) for _ in range(4)])
    img = generator.generate_image(random_str)
    # plt.imshow(img)
    # plt.title(random_str)
    # plt.show()
    img_L = img.convert('L')
    X = np.array(img_L)
    y = encode_label(random_str)
    if test:
        img.show()
        return X, random_str
    return X, np.array(y)

def encode_label(labels):
    y = []
    for l in labels:
        array = [0 for _ in range(52)]
        index = characters.index(l)
        array[index] = 1
        y += array
    return y

def decode_label(y):
    letters = ''
    for i in range(4):
        array = y[52*i:52*(i+1)]
        max_index = int(np.argmax(array))
        letter = characters[max_index]
        letters += letter
    return letters

#
# if __name__ == '__main__':
#     X, y = create_data()
#     print(X.shape)
#     print(y.shape)