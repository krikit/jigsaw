#!/usr/bin/env python3


"""
predict program
__author__ = 'krikit (krikit@naver.com)'
__copyright__ = 'No copyright, just copyleft!'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
import logging
import sys

from predictor import Predictor


#############
# functions #
#############
def run(args: Namespace):
    """
    actual function which is doing some task
    Args:
        args:  program arguments
    """
    pred = Predictor(args)
    if args.test:
        pred.predict_test(args.test, args.batch_size, sys.stdout)
    else:
        for line_num, line in enumerate(sys.stdin, start=1):
            if line_num % 1000 == 0:
                logging.info('%dk-th line..')
            line = line.rstrip('\r\n')
            if not line:
                print()
                continue
            print(pred.predict(line))


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='predict program')
    parser.add_argument('-m', '--model-path', help='model path', metavar='FILE', required=True)
    parser.add_argument('--bert-path', help='bert model path', metavar='FILE')
    parser.add_argument('--test', help='test dataset path', metavar='FILE')
    parser.add_argument('--batch-size', help='batch size <default: 128>', metavar='SIZE', type=int,
                        default=128)
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE')
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.input:
        sys.stdin = open(args.input, 'r', encoding='UTF-8')
    if args.output:
        sys.stdout = open(args.output, 'w', encoding='UTF-8')
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()
