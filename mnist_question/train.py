# -*- coding: utf-8 -*-


from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from tqdm import tqdm

import data
from model import MNISTClassifier
import pdb


def train(args):

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data.load_data(args.solve)

    x_train = torch.from_numpy(x_train/255.)
    x_valid = torch.from_numpy(x_valid/255.)
    x_test = torch.from_numpy(x_test/255.)

    ######################################################
    # ラベルyもxと同様にtorch.Tensorに変換する
    ######################################################
    y_train = torch.LongTensor(y_train)
    y_valid = torch.LongTensor(y_valid)
    y_test = torch.LongTensor(y_test)

    output_dim = 10
    if args.solve == 'q2':
        print('solve q2')
        output_dim = 2
    elif args.solve == 'q3':
        print('solve q3')
        output_dim = 3

    ############################################################################
    # 入力次元が784, 出力次元がoutput_dimなMNISTClassifierのオブジェクトを作る
    ############################################################################
    net = MNISTClassifier(784, output_dim)
    print(net)


    ############################################################################
    # netのパラメータ（list(net.parameters()))を更新するため，
    # それらをoptimizerに渡す.今回はAdam optimizerを利用する
    ############################################################################
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    ############################################################################
    # args.epochsに回す回数が設定されているので，その回数だけ回す
    ############################################################################
    for epoch in range(args.epochs):
        print("Epoch: {}, ".format(epoch), end='')
        train_epoch(epoch, net, optimizer, args, x_train, y_train, (x_valid, y_valid))
        test_epoch(epoch, net, x_test, y_test)


def train_epoch(epoch, net, optimizer, args, x_train, y_train, valid_data=None):
    losses = 0
    batch_idx = np.random.permutation(list(range(x_train.size(0))))

    ######################################################
    # 損失関数を定義します．多クラス分類問題の場合は？
    ######################################################
    creterion = nn.CrossEntropyLoss()

    for i in tqdm(range(x_train.size(0) // args.batch_size)):
        #########################################################
        # まずoptimizerでパラメータのgradをゼロにする
        #########################################################
        optimizer.zero_grad()

        x_batch = Variable(x_train[i * args.batch_size: (i+1)*args.batch_size])
        y_batch = Variable(y_train[i * args.batch_size: (i+1)*args.batch_size])


        #########################################################
        # x_batchをネットワークに入力して，結果を出力させる
        #########################################################
        # x_batch = x_batch.view(32, 1, 28, 28)
        y_pred = net(x_batch)

        loss = F.nll_loss(y_pred, y_batch)

        #########################################################
        # 計算した損失関数を偏微分する
        #########################################################
        loss.backward()
        #########################################################
        # パラメータをoptimizerに更新させる
        #########################################################
        optimizer.step()

        losses += loss.data[0]
    print("Train Loss: {0:.3f} ".format(losses/(i+1)), end='')

    if not valid_data is None:
        x_batch = Variable(valid_data[0], requires_grad=False)
        # x_batch = x_batch.view(5000, 1, 28, 28)
        y_pred = net(x_batch)
        print("Valid loss: {0:.3f}".format(creterion(y_pred, Variable(valid_data[1])).data[0]))

    return losses / (i+1)


def test_epoch(epoch, net, x_test, y_test):
    # x_test = x_test.view(10000, 1, 28, 28)
    x = Variable(x_test, requires_grad=False)
    y_pred = net(x)
    loss = torch.mean(F.nll_loss(y_pred, Variable(y_test)))
    y_pred = torch.max(y_pred, 1)[1]
    print("Test loss: {0:.3f}, Acc: {1:.3f}".format(loss.data[0], torch.mean((y_pred.data == y_test).float())))

if __name__ == '__main__':
    train()
