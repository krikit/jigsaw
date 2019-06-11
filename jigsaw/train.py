#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
train program
__author__ = 'krikit (krikit@naver.com)'
__copyright__ = 'No copyright, just copyleft!'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
import logging

import dataset
from trainer import Trainer


#############
# functions #
#############
def run(args: Namespace):
    """
    actual function which is doing some task
    Args:
        args:  program arguments
    """
    Trainer(args, dataset.load(args.train)).run()


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='cat')
    parser.add_argument('-t', '--train', help='train dataset', metavar='FILE', required=True)
    parser.add_argument('--debug', help='enable debug', action='store_true')
    parser.add_argument('--epoch', help='epoch number <default: 100>', metavar='NUM', type=int,
                        default=100)
    parser.add_argument('--patience', help='patience number for early stopping <default: 5>',
                        metavar='NUM', type=int, default=5)
    parser.add_argument('--batch-size', help='batch size <default: 36>', metavar='SIZE', type=int,
                        default=36)
    parser.add_argument('--model-out', help='model output path <default: ./toxicity.model>',
                        metavar='FILE', default='./toxicity.model')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()
