# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import


import argparse
from train import train

parser = argparse.ArgumentParser(description='MNIST TRAINING')
parser.add_argument('--epochs', type=int, default=10,
                    help='Epoch')
parser.add_argument('--gpu', type=bool, default=False,
                    help='Use gpu or not')
parser.add_argument('--lr', type=float, default=0.0003,
                    help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--solve', type=str, default='q1',
                    help='the question to solve')

args = parser.parse_args()

train(args)
