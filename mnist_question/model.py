# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class MNISTClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        """
            MNISTの問題を解くためのネットワーク構造を定義

            引数：
                input_size: 入力される画像のピクセル数．
                    例：mnistの場合，28 x 28 = 784
                output_size: 出力される次元数．
                    例：10クラス分類問題の場合，10
            field：
                self.net: 自身のネットワーク構造
        """
        #############################################################
        # 最初にnn.Moduleを継承するため，親クラスの__init__を呼び出す
        #############################################################
        super(MNISTClassifier, self).__init__()
        #############################################################
        # 引数で受け取ったinput_sizeで始まるネットワーク．
        # 出力はoutput_sizeで，日活性化関数は，n値分類問題ならば，
        # nn.LogSoftmax()を使う．
        #############################################################

        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(400, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, output_size)
        self.dense = nn.Linear(28*28, 10)

        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(400, 50)
        # self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
            順伝搬計算．

            引数：
                x: 入力データの1バッチ
                    Shape: (Batch size, Input dim)
                    例：32 x 784
        """
        #############################################################
        #
        # 引数で受け取ったxをself.netに入力して，
        # 出力結果をoutputに代入
        #        x = self.conv1(x)
        #############################################################
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 400)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        x = self.dense(x)
        output = F.log_softmax(x)

        # h = self.conv1(x)
        # h = F.relu(h)
        # h = F.max_pool2d(h,2)
        # h = self.conv2(h)
        # h = F.relu(h)
        # h = F.max_pool2d(h,2)
        # h = h.view(-1, 400)
        # h = self.fc1(h)
        # h = F.relu(h)
        # h = self.fc2(h)
        # h = F.relu(h)
        # h = F.LogSoftmax(h)
        # output = h
        return output
