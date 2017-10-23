# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import


import torch
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split


def load_data(solve, train_size=55000, valid_size=5000, test_size=10000, seed=1234):
    images = np.load('./data/image.npy')
    labels = np.load('./data/labels.npy')

    # print(images)
    ##############################################################
    #
    # 全データを（学習＋検証）：テストに分割する

    ##############################################################
    train_x, test_x, train_y, test_y = train_test_split(images, labels,
                                                        test_size=1/7,
                                                        random_state=seed)

    ##############################################################
    #
    # （学習＋検証）データを学習：検証：に分割する
    #
    ##############################################################
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y,
                                                          test_size= 1/12,
                                                          random_state=seed)

    # 応用
    if solve == 'q2': # 数字を偶数・奇数の2値分類する場合
        return (train_x, label2ans2(train_y)), (valid_x, label2ans2(valid_y)), (test_x, label2ans2(test_y))
    elif solve == 'q3': # 数字を0-3, 4-6, 7-9の3値分類する場合
        return (train_x, label2ans3(train_y)), (valid_x, label2ans3(valid_y)), (test_x, label2ans3(test_y))
    # 10クラス分類する場合
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def label2ans2(label):
      #  受け取ったラベル（0-9）を偶数・奇数の2値に振り直す
    return np.array([1 if l % 2 == 1 else 0 for l in label])


def label2ans3(label):
     #   受け取ったラベル（0-9）を0-3, 4-6, 7-9の3値に振り直す
    new_label = []
    for l in label:
        if l <= 3:
            tmp = 0
        elif l <= 6:
            tmp = 1
        else:
            tmp = 2
        new_label.append(tmp)
    return np.array(new_label)

if __name__ == '__main__':
    (x, y),  _, _ = load_data()
    print(label2ans3(y))
